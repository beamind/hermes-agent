from __future__ import annotations

import asyncio
import logging
import queue
from pathlib import Path
from typing import Any, Optional

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.session import SessionSource
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# Fixed chat identifier for the local voice session
_VOICE_CHAT_ID = "local_voice"


def _load_voice_config() -> dict[str, Any]:
    """Load voice gateway configuration from ~/.hermes/config.yaml.

    Uses the ``voice_gateway`` key to avoid collision with the CLI's
    ``voice`` section (push-to-talk / recording settings).
    """
    config_path = get_hermes_home() / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("voice_gateway", {}) or {}
    except Exception:
        logger.exception("Failed to load voice_gateway config from %s", config_path)
        return {}


def _expand_path(path: str) -> str:
    """Expand user home and env vars in a path string."""
    return __import__("os").path.expandvars(__import__("os").path.expanduser(path))


class VoiceAdapter(BasePlatformAdapter):
    """Local voice gateway adapter.

    Listens for a wake word on the microphone, streams audio to ASR,
    injects the transcribed text into the Hermes agent pipeline, and
    plays TTS responses through the local speaker.

    Uses a single persistent :class:`tools.voice.audio_capture.AudioCapture`
    process so that the microphone is never released between wake-word
    detection and ASR, eliminating the audio drop-out that previously
    caused missing words (e.g. "play Wang Fei's Red Bean" → "Dou").
    """

    MAX_MESSAGE_LENGTH = 4000

    def __init__(self, config: PlatformConfig) -> None:
        super().__init__(config, Platform.VOICE)
        self._voice_cfg = _load_voice_config()
        self._audio_capture: Any = None
        self._wake_detector: Any = None
        self._asr: Any = None
        self._audio_player: Any = None
        self._session_manager: Any = None
        self._wake_task: asyncio.Task | None = None
        self._connected = False

    # ------------------------------------------------------------------
    # BasePlatformAdapter interface
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Initialize voice hardware and start wake-word listening."""
        if self._connected:
            return True

        # 1. Audio player (must work for TTS playback)
        from tools.voice.audio_player import AudioPlayer, check_audio_player_requirements

        if not check_audio_player_requirements():
            logger.error("Audio player requirements not met (sounddevice missing)")
            return False

        self._audio_player = AudioPlayer(
            on_play_start=self._pause_music,
        )

        # 2. ASR
        asr_provider = self._voice_cfg.get("asr", {}).get("provider", "dashscope")
        if asr_provider == "dashscope":
            from tools.voice.asr.dashscope_asr import DashScopeASR, check_dashscope_asr_requirements

            if not check_dashscope_asr_requirements():
                logger.error("DashScope ASR requirements not met")
                return False

            api_key = __import__("os").getenv("ASR_DASHSCOPE_API_KEY", "").strip()
            if not api_key:
                logger.error(
                    "ASR_DASHSCOPE_API_KEY not configured. "
                    "Run: hermes setup gateway  → select 'Local Voice' to configure."
                )
                return False

            self._asr = DashScopeASR(
                api_key=api_key,
                model=self._voice_cfg.get("asr", {}).get("dashscope", {}).get("model", "qwen3-asr-flash-realtime"),
                sample_rate=self._voice_cfg.get("asr", {}).get("dashscope", {}).get("sample_rate", 16000),
            )
        else:
            logger.error("Unsupported ASR provider: %s", asr_provider)
            return False

        # 3. Session manager
        from tools.voice.session_manager import VoiceSessionManager

        self._session_manager = VoiceSessionManager(
            on_wake=self._on_wake,
            on_listen_complete=self._on_listen_complete,
            on_speaking_complete=self._on_speaking_complete,
            continuation_timeout=float(self._voice_cfg.get("session", {}).get("continuation_timeout", 30)),
            max_continuation_rounds=int(self._voice_cfg.get("session", {}).get("max_continuation_rounds", 3)),
        )

        # 4. Unified audio capture (must start before wake-word detector)
        from tools.voice.audio_capture import AudioCapture

        self._audio_capture = AudioCapture(
            sample_rate=16000,
            alsadevice=self._voice_cfg.get("audio", {}).get("mic_device", ""),
            mic_gain=float(self._voice_cfg.get("wake_word", {}).get("mic_gain", 1.0)),
        )
        self._audio_capture.start()

        # 5. Wake word detector
        from tools.voice.wake_word import WakeWordDetector, check_wake_word_requirements

        if not check_wake_word_requirements():
            logger.error("Wake word requirements not met")
            return False

        ww_cfg = self._voice_cfg.get("wake_word", {})
        model_dir = _expand_path(ww_cfg.get("model_dir", "~/.hermes/models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"))
        model_dir = Path(model_dir)

        # Resolve model files
        tokens = ww_cfg.get("tokens") or str(model_dir / "tokens.txt")
        encoder = ww_cfg.get("encoder") or str(next(model_dir.glob("encoder-*.onnx"), ""))
        decoder = ww_cfg.get("decoder") or str(next(model_dir.glob("decoder-*.onnx"), ""))
        joiner = ww_cfg.get("joiner") or str(next(model_dir.glob("joiner-*.onnx"), ""))
        keywords_file = ww_cfg.get("keywords_file") or str(model_dir / "keywords.txt")

        for req_file in (tokens, encoder, decoder, joiner, keywords_file):
            if not req_file or not Path(req_file).exists():
                logger.error("Wake word model file not found: %s", req_file)
                return False

        self._wake_detector = WakeWordDetector(
            tokens=tokens,
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            keywords_file=keywords_file,
            keywords_threshold=float(ww_cfg.get("keywords_threshold", 0.1)),
            keywords_score=float(ww_cfg.get("keywords_score", 2.5)),
            num_threads=int(ww_cfg.get("num_threads", 2)),
            provider=ww_cfg.get("provider", "cpu"),
            sample_rate=16000,
            pre_buffer_seconds=1.0,
            mic_gain=float(ww_cfg.get("mic_gain", 1.0)),
        )

        self._connected = True
        self._mark_connected()

        # Start wake-word loop in background
        self._wake_task = asyncio.create_task(self._wake_loop())
        logger.info("VoiceAdapter connected — listening for wake word")
        return True

    async def disconnect(self) -> None:
        """Stop listening and release hardware."""
        self._connected = False
        self._mark_disconnected()

        if self._wake_task and not self._wake_task.done():
            self._wake_task.cancel()
            try:
                await self._wake_task
            except asyncio.CancelledError:
                pass

        if self._wake_detector:
            self._wake_detector.stop()

        if self._session_manager:
            await self._session_manager.stop()

        if self._audio_capture:
            self._audio_capture.stop()

        if self._audio_player:
            self._audio_player.stop()

        logger.info("VoiceAdapter disconnected")

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[dict] = None,
        **kwargs: Any,
    ) -> SendResult:
        """Local voice adapter has no remote delivery; log only."""
        logger.debug("[voice] send to %s: %s", chat_id, content[:200])
        return SendResult(success=True)

    async def send_typing(self, chat_id: str, **kwargs: Any) -> None:
        """No typing indicator for voice."""
        pass

    async def play_tts(self, chat_id: str, audio_path: str, **kwargs: Any) -> SendResult:
        """Play TTS audio through the local speaker."""
        if not self._audio_player:
            logger.warning("Audio player not initialized")
            return SendResult(success=False, error="Audio player not initialized")

        # Release microphone so ALSA device is free for TTS playback.
        # arecord holds the hw: device exclusively; sounddevice cannot
        # open the output stream while arecord is running.
        if self._audio_capture and self._audio_capture.is_running:
            logger.debug("Pausing AudioCapture for TTS playback")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._audio_capture.stop)

        await self._session_manager.on_agent_response_received()

        try:
            await self._audio_player.play_file(audio_path)
        except Exception:
            logger.exception("Failed to play TTS audio")
            return SendResult(success=False, error="Playback failed")
        finally:
            # Resume microphone capture BEFORE notifying session manager.
            # Multi-turn continuation may immediately start ASR, which
            # needs AudioCapture to be producing chunks.
            if self._audio_capture and not self._audio_capture.is_running:
                logger.debug("Resuming AudioCapture after TTS")
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._audio_capture.start)
                # Discard any chunks captured during TTS (speaker echo)
                self._audio_capture.drain_queue()

            # Notify session manager that speaking is done
            try:
                await self._session_manager.on_speaking_complete()
            except Exception:
                logger.exception("Session manager on_speaking_complete failed")

        return SendResult(success=True)

    async def get_chat_info(self, chat_id: str) -> dict:
        return {"name": "Local Voice", "type": "dm", "chat_id": chat_id}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wake_consumer(self, audio_queue: queue.Queue, wake_detector: Any) -> None:
        """Background thread: feed audio chunks from AudioCapture to WakeWordDetector."""
        while self._connected and wake_detector.is_running:
            try:
                chunk = audio_queue.get(timeout=0.5)
                wake_detector.feed_chunk(chunk)
            except queue.Empty:
                continue
            except Exception:
                logger.exception("Wake-word consumer error")
                break

    async def _wake_loop(self) -> None:
        """Run wake-word detection in an asyncio-friendly way.

        A single persistent AudioCapture process feeds a Queue.  A
        background thread pulls from that Queue and pushes chunks into
        WakeWordDetector.  When the wake word fires the detector is
        stopped (so its consumer thread exits), but AudioCapture keeps
        running — ASR simply starts reading from the same Queue, giving
        us a zero-gap hand-off.
        """
        loop = asyncio.get_event_loop()

        def _on_wake_sync(pre_audio: bytes) -> None:
            # Stop the detector immediately so its consumer thread exits.
            # The Queue keeps filling from AudioCapture, so ASR will pick
            # up every chunk that follows without losing audio.
            if self._wake_detector and self._wake_detector.is_running:
                self._wake_detector.stop()
            asyncio.run_coroutine_threadsafe(
                self._session_manager.handle_wake(pre_audio), loop
            )

        try:
            while self._connected:
                if (
                    self._wake_detector
                    and not self._wake_detector.is_running
                    and self._session_manager.is_idle
                ):
                    logger.debug("Restarting wake-word detector")
                    # Discard stale audio from previous turn so the detector
                    # does not process old speech / TTS echo.
                    self._audio_capture.drain_queue()
                    self._wake_detector.start(on_wake=_on_wake_sync)
                    await loop.run_in_executor(
                        None,
                        self._wake_consumer,
                        self._audio_capture.get_queue(),
                        self._wake_detector,
                    )
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            if self._wake_detector:
                self._wake_detector.stop()

    async def _on_wake(self, pre_audio: bytes) -> None:
        """Callback: wake word detected → stop any playback and start ASR."""
        logger.info("Wake word detected — pausing music, stopping TTS")
        self._audio_player.stop()
        self._pause_music()
        await self._start_asr(pre_audio)

    async def _start_asr(self, pre_audio: bytes) -> None:
        """Start ASR listening and wire completion to session manager."""
        if not self._asr or not self._audio_capture:
            return

        async def _listen() -> str:
            return await self._asr.recognize_from_microphone(
                pre_audio, self._audio_capture.get_queue()
            )

        await self._session_manager.start_listening(_listen())

    async def _on_listen_complete(self, text: str) -> None:
        """Callback: ASR finished → inject text into Hermes gateway."""
        logger.info("ASR result: %r", text)

        source = self.build_source(
            chat_id=_VOICE_CHAT_ID,
            chat_name="Local Voice",
            chat_type="dm",
            user_id="voice_user",
            user_name="voice_user",
        )

        event = MessageEvent(
            text=text,
            message_type=MessageType.VOICE,
            source=source,
        )

        await self.handle_message(event)

    async def _on_speaking_complete(self) -> None:
        """Callback: TTS finished.

        If the session manager transitioned back to LISTENING (multi-turn
        continuation), we need to start ASR again without a wake word.
        """
        if self._session_manager.state == "listening":
            logger.info("Multi-turn continuation — starting ASR without wake word")
            await self._start_asr(pre_audio=b"")

    def _pause_music(self) -> None:
        """Pause the smart-speaker music player if it is running."""
        try:
            from plugins.smart_speaker.tools.music_tools import _get_player

            player = _get_player()
            status = player.get_status()
            if status.playing and not status.paused:
                player.pause()
                logger.debug("Music paused for voice interaction")
        except Exception:
            # Music player may not be installed or initialized — ignore
            pass


# ------------------------------------------------------------------
# Requirements check
# ------------------------------------------------------------------

def check_voice_requirements() -> bool:
    """Check if voice adapter dependencies are available."""
    from tools.voice.audio_player import check_audio_player_requirements
    from tools.voice.wake_word import check_wake_word_requirements

    ok = True
    if not check_audio_player_requirements():
        ok = False
    if not check_wake_word_requirements():
        ok = False

    # ASR is optional at check time (will fail at connect if unavailable)
    try:
        from tools.voice.asr.dashscope_asr import check_dashscope_asr_requirements
        if not check_dashscope_asr_requirements():
            logger.warning("DashScope ASR not available")
    except Exception:
        pass

    return ok
