from __future__ import annotations

import collections
import logging
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


try:
    import sherpa_onnx

    _SHERPA_AVAILABLE = True
except Exception:
    _SHERPA_AVAILABLE = False
    sherpa_onnx = None  # type: ignore[assignment]


class WakeWordDetector:
    """Sherpa-ONNX based keyword spotter that consumes audio from an external source.

    Previously opened ``arecord`` directly; now it receives mono int16 PCM chunks
    via :meth:`feed_chunk` from a shared :class:`tools.voice.audio_capture.AudioCapture`.
    This eliminates the microphone hand-off gap that caused dropped words after the
    wake word was detected.
    """

    def __init__(
        self,
        tokens: str,
        encoder: str,
        decoder: str,
        joiner: str,
        keywords_file: str,
        keywords_threshold: float = 0.25,
        keywords_score: float = 1.0,
        num_threads: int = 2,
        provider: str = "cpu",
        max_active_paths: int = 4,
        num_trailing_blanks: int = 1,
        sample_rate: int = 16000,
        pre_buffer_seconds: float = 1.0,
        mic_gain: float = 1.0,
    ) -> None:
        if not _SHERPA_AVAILABLE:
            raise RuntimeError(
                "sherpa-onnx is not installed. Run: pip install sherpa-onnx"
            )

        self._sample_rate = sample_rate
        self._mic_gain = mic_gain
        self._running = False
        self._on_wake: Callable[[bytes], None] | None = None
        self._stream = None

        # Ring buffer: stores raw int16 PCM bytes
        buffer_bytes = int(sample_rate * 2 * pre_buffer_seconds)  # 16-bit = 2 bytes/sample
        self._ring_buffer: collections.deque[bytes] = collections.deque(
            maxlen=max(1, buffer_bytes // (int(sample_rate * 0.1) * 2))
        )

        self._spotter = sherpa_onnx.KeywordSpotter(
            tokens=tokens,
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            keywords_file=keywords_file,
            keywords_threshold=keywords_threshold,
            keywords_score=keywords_score,
            num_threads=num_threads,
            provider=provider,
            max_active_paths=max_active_paths,
            num_trailing_blanks=num_trailing_blanks,
            sample_rate=sample_rate,
        )

    def start(self, on_wake: Callable[[bytes], None]) -> None:
        """Prepare the detector to receive chunks."""
        if self._running:
            return
        self._on_wake = on_wake
        self._running = True
        self._stream = self._spotter.create_stream()
        self._ring_buffer.clear()
        logger.info("Wake word detector started")

    def stop(self) -> None:
        """Stop detection and release state."""
        self._running = False
        self._on_wake = None
        self._stream = None
        logger.info("Wake word detector stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    def feed_chunk(self, pcm_int16_mono: bytes) -> None:
        """Feed one chunk of mono int16 PCM audio into the spotter.

        Args:
            pcm_int16_mono: Raw PCM bytes, little-endian int16, mono,
                length should correspond to the configured chunk duration.
        """
        if not self._running or self._stream is None:
            return

        # Store int16 PCM in ring buffer
        self._ring_buffer.append(pcm_int16_mono)

        # Convert to float32 for Sherpa-ONNX
        samples = np.frombuffer(pcm_int16_mono, dtype=np.int16).astype(np.float32)
        samples /= 32768.0

        if self._mic_gain != 1.0:
            samples = np.clip(samples * self._mic_gain, -1.0, 1.0)

        # accept_waveform is a method on the stream, not the spotter
        self._stream.accept_waveform(self._sample_rate, samples)

        while self._spotter.is_ready(self._stream):
            self._spotter.decode_stream(self._stream)

        result = self._spotter.get_result(self._stream)
        if result.strip():
            logger.info("Wake word detected: %s", result.strip())
            self._spotter.reset_stream(self._stream)

            # Drain buffer and pass to callback
            pre_audio = b"".join(self._ring_buffer)
            self._ring_buffer.clear()

            if self._on_wake:
                self._on_wake(pre_audio)


def check_wake_word_requirements() -> bool:
    """Check if wake word dependencies are available."""
    if not _SHERPA_AVAILABLE:
        logger.warning("sherpa-onnx not installed. Run: uv pip install sherpa-onnx")
        return False
    return True
