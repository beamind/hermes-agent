from __future__ import annotations

import asyncio
import io
import logging
import shutil
import subprocess
import threading
import wave
from pathlib import Path
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


try:
    import sounddevice as sd

    _SOUNDDEVICE_AVAILABLE = True
except Exception:
    _SOUNDDEVICE_AVAILABLE = False
    sd = None  # type: ignore[assignment]


class AudioPlayer:
    """Local audio playback via sounddevice.

    Supports WAV and raw PCM.  Provides hooks for pause/resume of external
    music players (e.g. the smart-speaker music tools from Phase 0).
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        on_play_start: Callable[[], None] | None = None,
        on_play_end: Callable[[], None] | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self._on_play_start = on_play_start
        self._on_play_end = on_play_end
        self._lock = asyncio.Lock()

    async def play_file(self, path: str | Path) -> None:
        """Play an audio file (wav/mp3/ogg) locally."""
        if not _SOUNDDEVICE_AVAILABLE:
            logger.error("sounddevice not installed. Run: pip install sounddevice")
            return

        path = Path(path)
        if not path.exists():
            logger.error("Audio file not found: %s", path)
            return

        async with self._lock:
            if self._on_play_start:
                try:
                    self._on_play_start()
                except Exception:
                    logger.exception("on_play_start hook failed")

            loop = asyncio.get_event_loop()
            done = asyncio.Event()

            def _play() -> None:
                try:
                    data = path.read_bytes()
                    # Try WAV first
                    if data[:4] == b"RIFF":
                        with io.BytesIO(data) as buf:
                            with wave.open(buf, "rb") as wf:
                                rate = wf.getframerate()
                                channels = wf.getnchannels()
                                frames = wf.readframes(wf.getnframes())
                                audio_np = np.frombuffer(frames, dtype=np.int16)
                                if channels > 1:
                                    audio_np = audio_np.reshape(-1, channels)
                    elif shutil.which("ffmpeg"):
                        # Decode OGG/MP3/etc. to WAV via ffmpeg
                        result = subprocess.run(
                            ["ffmpeg", "-i", str(path), "-f", "wav", "-"],
                            capture_output=True,
                        )
                        if result.returncode == 0 and len(result.stdout) > 44:
                            with io.BytesIO(result.stdout) as buf:
                                with wave.open(buf, "rb") as wf:
                                    rate = wf.getframerate()
                                    channels = wf.getnchannels()
                                    frames = wf.readframes(wf.getnframes())
                                    audio_np = np.frombuffer(frames, dtype=np.int16)
                                    if channels > 1:
                                        audio_np = audio_np.reshape(-1, channels)
                        else:
                            logger.error(
                                "ffmpeg decoding failed for %s: %s",
                                path,
                                result.stderr.decode()[:200],
                            )
                            return
                    else:
                        # Fallback: assume raw PCM mono int16 at default sample_rate
                        rate = self.sample_rate
                        audio_np = np.frombuffer(data, dtype=np.int16)

                    sd.play(audio_np, samplerate=rate)
                    sd.wait()
                except Exception:
                    logger.exception("Error playing audio file %s", path)
                finally:
                    loop.call_soon_threadsafe(done.set)

            threading.Thread(target=_play, daemon=True).start()
            await done.wait()

            if self._on_play_end:
                try:
                    self._on_play_end()
                except Exception:
                    logger.exception("on_play_end hook failed")

    async def play_pcm(self, audio_data: bytes, sample_rate: int | None = None) -> None:
        """Play raw PCM int16 mono audio."""
        if not _SOUNDDEVICE_AVAILABLE:
            logger.error("sounddevice not installed. Run: pip install sounddevice")
            return

        async with self._lock:
            if self._on_play_start:
                try:
                    self._on_play_start()
                except Exception:
                    logger.exception("on_play_start hook failed")

            loop = asyncio.get_event_loop()
            done = asyncio.Event()
            rate = sample_rate or self.sample_rate

            def _play() -> None:
                try:
                    audio_np = np.frombuffer(audio_data, dtype=np.int16)
                    sd.play(audio_np, samplerate=rate)
                    sd.wait()
                except Exception:
                    logger.exception("Error playing PCM audio")
                finally:
                    loop.call_soon_threadsafe(done.set)

            threading.Thread(target=_play, daemon=True).start()
            await done.wait()

            if self._on_play_end:
                try:
                    self._on_play_end()
                except Exception:
                    logger.exception("on_play_end hook failed")

    def stop(self) -> None:
        """Stop all playback immediately."""
        if _SOUNDDEVICE_AVAILABLE:
            sd.stop()


def check_audio_player_requirements() -> bool:
    """Check if audio playback dependencies are available."""
    if not _SOUNDDEVICE_AVAILABLE:
        logger.warning("sounddevice not installed. Run: uv pip install sounddevice")
        return False
    return True
