from __future__ import annotations

import asyncio
import base64
import logging
import queue
import threading
import time

import numpy as np

from tools.voice.asr.base import BaseASR

logger = logging.getLogger(__name__)

_WS_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"

SUPPORTED_MODELS = [
    "qwen3-asr-flash-realtime",
]


try:
    import dashscope
    from dashscope.audio.qwen_omni import (
        MultiModality,
        OmniRealtimeCallback,
        OmniRealtimeConversation,
    )
    from dashscope.audio.qwen_omni.omni_realtime import TranscriptionParams

    _DASHSCOPE_AVAILABLE = True
except Exception:
    _DASHSCOPE_AVAILABLE = False
    dashscope = None  # type: ignore[assignment]
    OmniRealtimeCallback = object  # type: ignore[assignment,misc]
    OmniRealtimeConversation = None  # type: ignore[assignment]
    TranscriptionParams = None  # type: ignore[assignment]
    MultiModality = None  # type: ignore[assignment]


class _ASRCallback(OmniRealtimeCallback):
    """Collects transcription results from OmniRealtimeConversation."""

    def __init__(self) -> None:
        self.final_text = ""
        self._partial_text = ""
        self.sentence_done = threading.Event()
        self.session_ready = threading.Event()
        self.speech_stopped = threading.Event()
        self.error: Exception | None = None

    def on_open(self) -> None:
        logger.debug("ASR WebSocket opened")

    def on_close(self, code: int, msg: str) -> None:
        logger.debug("ASR WebSocket closed: code=%s msg=%s", code, msg)
        self.sentence_done.set()

    def on_event(self, response: dict) -> None:
        evt_type = response.get("type", "")

        if evt_type == "session.created":
            self.session_ready.set()

        elif evt_type == "conversation.item.input_audio_transcription.completed":
            text = response.get("transcript", "")
            if text:
                self.final_text = text
            self.sentence_done.set()

        elif evt_type == "conversation.item.input_audio_transcription.text":
            text = response.get("text", "") + response.get("stash", "")
            if text:
                self._partial_text = text

        elif evt_type == "input_audio_buffer.speech_stopped":
            logger.debug("VAD detected speech end")
            self.speech_stopped.set()

        elif evt_type == "error":
            err_msg = response.get("error", {}).get("message", str(response))
            self.error = RuntimeError(f"ASR error: {err_msg}")
            self.sentence_done.set()

        elif evt_type == "session.finished":
            if not self.final_text and self._partial_text:
                self.final_text = self._partial_text
            self.sentence_done.set()


class DashScopeASR(BaseASR):
    """DashScope ASR using Qwen3-ASR-Flash-Realtime via OmniRealtime API.

    The model has built-in VAD — it detects when the user stops speaking,
    so the caller does not need silence detection.

    This implementation consumes live audio from a :class:`queue.Queue`
    provided by :class:`tools.voice.audio_capture.AudioCapture`, avoiding
    the microphone hand-off gap that previously caused dropped words.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "qwen3-asr-flash-realtime",
        sample_rate: int = 16000,
        audio_format: str = "pcm",
    ) -> None:
        if not _DASHSCOPE_AVAILABLE:
            raise RuntimeError(
                "dashscope is not installed. Run: pip install dashscope"
            )
        if model not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model: {model}. Choose from {SUPPORTED_MODELS}"
            )
        dashscope.api_key = api_key
        self.model = model
        self.sample_rate = sample_rate
        self.audio_format = audio_format

    def _create_conversation(self, callback: _ASRCallback) -> OmniRealtimeConversation:
        return OmniRealtimeConversation(
            model=self.model,
            url=_WS_URL,
            callback=callback,
        )

    def _configure_session(self, conv: OmniRealtimeConversation) -> None:
        conv.update_session(
            output_modalities=[MultiModality.TEXT],
            enable_input_audio_transcription=True,
            transcription_params=TranscriptionParams(
                language="zh",
                sample_rate=self.sample_rate,
                input_audio_format=self.audio_format,
            ),
        )

    async def recognize(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        """Recognize speech from pre-recorded audio bytes."""
        loop = asyncio.get_event_loop()
        callback = _ASRCallback()

        def _run() -> str:
            conv = self._create_conversation(callback)
            conv.connect()

            if not callback.session_ready.wait(timeout=10):
                raise RuntimeError("ASR session creation timed out")

            self._configure_session(conv)

            # Send audio in ~100ms chunks
            chunk_size = (sample_rate * 2) // 10  # 100ms of 16-bit PCM
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i : i + chunk_size]
                conv.append_audio(base64.b64encode(chunk).decode("ascii"))
                time.sleep(0.05)

            conv.end_session()
            callback.sentence_done.wait(timeout=30)
            conv.close()
            return callback.final_text

        text = await loop.run_in_executor(None, _run)

        if callback.error:
            raise callback.error
        return text

    async def recognize_from_microphone(
        self, pre_audio: bytes = b"", audio_queue: queue.Queue | None = None
    ) -> str:
        """Stream audio from a shared Queue, let the model detect when the user stops.

        Args:
            pre_audio: PCM bytes (int16, mono) captured before the mic stream
                starts (e.g. from a wake-word ring buffer).
            audio_queue: Live audio chunk queue provided by AudioCapture.

        Returns the recognized text once the model signals sentence end.
        """
        if audio_queue is None:
            raise ValueError("audio_queue is required for recognize_from_microphone")

        loop = asyncio.get_event_loop()
        callback = _ASRCallback()

        def _stream() -> None:
            conv = self._create_conversation(callback)
            conv.connect()

            if not callback.session_ready.wait(timeout=10):
                raise RuntimeError("ASR session creation timed out")

            self._configure_session(conv)

            # Send pre-buffered audio first (from wake word ring buffer)
            if pre_audio:
                chunk_size = self.sample_rate * 2 // 10  # 100ms chunks
                for i in range(0, len(pre_audio), chunk_size):
                    chunk = pre_audio[i : i + chunk_size]
                    conv.append_audio(
                        base64.b64encode(chunk).decode("ascii")
                    )
                logger.debug(
                    "Sent %d bytes of pre-buffered audio to ASR",
                    len(pre_audio),
                )

            # Consume live audio from the shared queue
            while not callback.sentence_done.is_set():
                # When server VAD detects speech end, tell it we're done
                # so it returns the final transcript.
                if callback.speech_stopped.is_set():
                    try:
                        conv.end_session()
                    except Exception:
                        pass
                    callback.speech_stopped.clear()

                try:
                    chunk = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                conv.append_audio(
                    base64.b64encode(chunk).decode("ascii")
                )

            try:
                conv.close()
            except Exception:
                pass

        await loop.run_in_executor(None, _stream)

        if callback.error:
            raise callback.error
        return callback.final_text

    async def test_connection(self) -> bool:
        try:
            silent_pcm = b"\x00\x00" * self.sample_rate
            await self.recognize(silent_pcm)
            return True
        except Exception:
            logger.exception("ASR connection test failed")
            return False


def check_dashscope_asr_requirements() -> bool:
    """Check if DashScope ASR dependencies are available."""
    if not _DASHSCOPE_AVAILABLE:
        logger.warning("dashscope not installed. Run: uv pip install dashscope")
        return False
    return True
