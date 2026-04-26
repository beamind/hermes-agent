from __future__ import annotations

import collections
import logging
import shutil
import struct
import subprocess
import threading
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
    """Sherpa-ONNX based keyword spotter that listens on the microphone.

    Uses ``arecord`` (ALSA) directly instead of PortAudio/sounddevice,
    because PortAudio inside Docker does not reliably enumerate USB
    microphones on some hosts.

    Maintains a ring buffer (default 1 second) so that audio captured just
    before the wake word is detected can be forwarded to ASR.
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
        alsadevice: str = "",
    ) -> None:
        if not _SHERPA_AVAILABLE:
            raise RuntimeError(
                "sherpa-onnx is not installed. Run: pip install sherpa-onnx"
            )

        self._sample_rate = sample_rate
        self._mic_gain = mic_gain
        self._alsadevice = alsadevice or "hw:1,0"
        self._running = False
        self._thread: threading.Thread | None = None
        self._on_wake: Callable[[bytes], None] | None = None
        self._arecord_proc: subprocess.Popen | None = None

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
        """Start listening for the wake word in a background thread."""
        if self._running:
            return
        self._on_wake = on_wake
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        logger.info("Wake word detector started (alsa=%s)", self._alsadevice)

    def stop(self) -> None:
        """Stop listening."""
        self._running = False
        if self._arecord_proc is not None:
            self._arecord_proc.terminate()
            try:
                self._arecord_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._arecord_proc.kill()
            self._arecord_proc = None
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=3)
        self._thread = None
        logger.info("Wake word detector stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    def _listen_loop(self) -> None:
        stream = self._spotter.create_stream()
        chunk_duration = 0.1  # 100ms chunks
        chunk_samples = int(self._sample_rate * chunk_duration)

        # ALSA device: hw:1,0 is 2-channel; read stereo and take left channel.
        # Each frame = 2 channels * 2 bytes (S16_LE) = 4 bytes.
        frame_bytes = 4
        chunk_bytes = chunk_samples * frame_bytes

        if not shutil.which("arecord"):
            logger.error("arecord not found in PATH")
            self._running = False
            return

        cmd = [
            "arecord",
            "-D", self._alsadevice,
            "-f", "S16_LE",
            "-r", str(self._sample_rate),
            "-c", "2",
            "-t", "raw",
            "-",
        ]

        try:
            self._arecord_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            logger.exception("Failed to start arecord")
            self._running = False
            return

        try:
            while self._running:
                proc = self._arecord_proc
                if proc is None:
                    break
                pcm = proc.stdout.read(chunk_bytes)
                if len(pcm) < chunk_bytes:
                    if not self._running:
                        break
                    continue

                # Unpack little-endian int16 stereo → take left channel (every 2nd sample starting at 0)
                fmt = "<" + "hh" * chunk_samples
                flat = struct.unpack(fmt, pcm)
                left = np.array(flat[::2], dtype=np.float32) / 32768.0

                if self._mic_gain != 1.0:
                    left = np.clip(left * self._mic_gain, -1.0, 1.0)

                # Store int16 PCM in ring buffer
                pcm_int16 = (left * 32767).astype(np.int16).tobytes()
                self._ring_buffer.append(pcm_int16)

                stream.accept_waveform(self._sample_rate, left)

                while self._spotter.is_ready(stream):
                    self._spotter.decode_stream(stream)

                result = self._spotter.get_result(stream)
                if result.strip():
                    logger.info("Wake word detected: %s", result.strip())
                    self._spotter.reset_stream(stream)

                    # Drain buffer and pass to callback
                    pre_audio = b"".join(self._ring_buffer)
                    self._ring_buffer.clear()

                    if self._on_wake:
                        self._on_wake(pre_audio)

        except Exception:
            logger.exception("Wake word detector error")
        finally:
            if self._arecord_proc is not None:
                self._arecord_proc.terminate()
                try:
                    self._arecord_proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._arecord_proc.kill()
                self._arecord_proc = None
            self._running = False


def check_wake_word_requirements() -> bool:
    """Check if wake word dependencies are available."""
    if not _SHERPA_AVAILABLE:
        logger.warning("sherpa-onnx not installed. Run: uv pip install sherpa-onnx")
        return False
    if not shutil.which("arecord"):
        logger.warning("arecord not found in PATH. Install alsa-utils.")
        return False
    return True
