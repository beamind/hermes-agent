from __future__ import annotations

import logging
import queue
import shutil
import struct
import subprocess
import threading
from typing import Callable

logger = logging.getLogger(__name__)


class AudioCapture:
    """Unified microphone capture that feeds audio to multiple consumers.

    Runs a single persistent ``arecord`` process and pushes mono int16 PCM
    chunks into a thread-safe Queue.  Consumers (wake-word detector, ASR)
    pull from the same Queue, eliminating the microphone hand-off gap that
    caused dropped words.

    Attributes:
        sample_rate: Audio sample rate in Hz (default 16000).
        chunk_duration: Duration of each chunk in seconds (default 0.1).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration: float = 0.1,
        alsadevice: str = "",
        mic_gain: float = 1.0,
    ) -> None:
        self._sample_rate = sample_rate
        self._chunk_duration = chunk_duration
        self._alsadevice = alsadevice or "hw:1,0"
        self._mic_gain = mic_gain

        self._running = False
        self._thread: threading.Thread | None = None
        self._arecord_proc: subprocess.Popen | None = None

        # Queue capacity ≈ 30 s of 100 ms chunks at 16 kHz mono int16
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=300)

    # -- public API --------------------------------------------------------

    def start(self) -> None:
        """Start the persistent arecord capture thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(
            "AudioCapture started (alsa=%s, rate=%d, chunk=%.0fms)",
            self._alsadevice,
            self._sample_rate,
            self._chunk_duration * 1000,
        )

    def stop(self) -> None:
        """Stop capture and release the microphone."""
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
        logger.info("AudioCapture stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    def get_queue(self) -> queue.Queue[bytes]:
        """Return the audio chunk queue for consumers."""
        return self._queue

    def drain_queue(self) -> None:
        """Discard all pending chunks (useful when switching consumers)."""
        drained = 0
        while True:
            try:
                self._queue.get_nowait()
                drained += 1
            except queue.Empty:
                break
        if drained:
            logger.debug("AudioCapture drained %d queued chunks", drained)

    # -- internal ----------------------------------------------------------

    def _capture_loop(self) -> None:
        chunk_samples = int(self._sample_rate * self._chunk_duration)
        # Stereo S16_LE: 2 channels × 2 bytes = 4 bytes/frame
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

                # Unpack stereo S16_LE → left channel mono int16 bytes
                fmt = "<" + "hh" * chunk_samples
                flat = struct.unpack(fmt, pcm)
                left = bytearray()
                for i in range(0, len(flat), 2):
                    sample = flat[i]
                    if self._mic_gain != 1.0:
                        sample = int(max(-32768, min(32767, sample * self._mic_gain)))
                    left.extend(struct.pack("<h", sample))

                mono_bytes = bytes(left)

                try:
                    self._queue.put(mono_bytes, block=False)
                except queue.Full:
                    # Back-pressure: drop oldest chunk to make room
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self._queue.put(mono_bytes, block=False)
                    except queue.Full:
                        pass

        except Exception:
            logger.exception("AudioCapture loop error")
        finally:
            if self._arecord_proc is not None:
                self._arecord_proc.terminate()
                try:
                    self._arecord_proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._arecord_proc.kill()
                self._arecord_proc = None
            self._running = False
