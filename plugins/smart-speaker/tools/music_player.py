"""
MusicPlayer — adapted from home-ai-assistant for Hermes smart-speaker plugin.

Wraps mpv for local music playback with playlist support.
The player instance lives at module level so it survives across
tool calls (Hermes calls each tool in isolation).
"""

from __future__ import annotations

import logging
import random
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# mpv import helper
# ---------------------------------------------------------------------------

def _import_mpv():
    try:
        import mpv
        return mpv
    except (ImportError, OSError) as e:
        raise RuntimeError(
            "mpv is required. Install mpv and ensure the library "
            "(mpv-1.dll / libmpv.so / libmpv.dylib) is on your system PATH."
        ) from e


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PlaybackStatus:
    playing: bool = False
    paused: bool = False
    current_file: str = ""
    title: str = ""
    duration: float = 0.0
    position: float = 0.0
    volume: int = 70


# ---------------------------------------------------------------------------
# MusicPlayer
# ---------------------------------------------------------------------------

class MusicPlayer:
    """Wraps mpv for music playback with playlist support."""

    _MUSIC_EXTENSIONS = {".mp3", ".flac", ".wav", ".ogg", ".m4a", ".wma", ".aac"}

    def __init__(self, default_volume: int = 70) -> None:
        mpv = _import_mpv()
        # Try PulseAudio first — on Linux+PipeWire this lets mpv share the
        # audio device with other apps (e.g. TTS) instead of fighting for
        # exclusive ALSA access via PortAudio.  Fall back to mpv default.
        try:
            self._player = mpv.MPV(ao="pulse")
        except Exception:
            self._player = mpv.MPV()
        self._player.volume = default_volume
        self._lock = threading.RLock()
        self._playlist: list[dict[str, str]] = []
        self._playlist_index: int = -1
        self._on_track_end: Callable[[], None] | None = None

        @self._player.event_callback("end-file")
        def _on_end(event):
            try:
                if event.data.reason == 0:  # EOF
                    if self._on_track_end:
                        self._on_track_end()
            except Exception:
                pass

    # -- callbacks --

    def set_on_track_end(self, callback: Callable[[], None] | None) -> None:
        self._on_track_end = callback

    # -- basic controls --

    def play(self, file_path: str) -> None:
        with self._lock:
            self._player.play(file_path)
            self._player.pause = False
            self._playlist.clear()
            self._playlist_index = -1

    def pause(self) -> None:
        with self._lock:
            self._player.pause = True

    def resume(self) -> None:
        with self._lock:
            # If currently paused, just unpause (preserves position)
            if self._player.pause:
                self._player.pause = False
                return
            # If idle but has a playlist position, restart from that track
            if self._player.core_idle and 0 <= self._playlist_index < len(self._playlist):
                self.play_index(self._playlist_index)
            else:
                self._player.pause = False

    def stop(self) -> None:
        with self._lock:
            self._player.stop()

    def set_volume(self, volume: int) -> None:
        vol = max(0, min(100, volume))
        with self._lock:
            self._player.volume = vol

    @property
    def volume(self) -> int:
        return int(self._player.volume or 0)

    # -- playlist --

    def clear_playlist(self) -> None:
        with self._lock:
            self._player.stop()
            self._playlist.clear()
            self._playlist_index = -1

    def load_playlist(self, songs: list[dict[str, str]]) -> None:
        with self._lock:
            self._playlist = list(songs)
            self._playlist_index = -1

    def play_index(self, index: int) -> bool:
        with self._lock:
            while 0 <= index < len(self._playlist):
                song = self._playlist[index]
                path = song.get("file_path", "")

                # Support HTTP URLs (Netease streaming)
                if path and (path.startswith("http://") or path.startswith("https://")):
                    self._playlist_index = index
                    logger.debug("MusicPlayer playing URL: %s", path)
                    self._player.play(path)
                    self._player.pause = False
                    return True

                if path and Path(path).exists():
                    self._playlist_index = index
                    logger.debug("MusicPlayer playing: %s", path)
                    self._player.play(path)
                    self._player.pause = False
                    return True
                logger.warning("Could not resolve %s, trying next", song)
                index += 1
            self._playlist_index = -1
            return False

    def next(self) -> bool:
        with self._lock:
            if self._playlist_index + 1 < len(self._playlist):
                return self.play_index(self._playlist_index + 1)
            return False

    def previous(self) -> bool:
        with self._lock:
            if self._playlist_index - 1 >= 0:
                return self.play_index(self._playlist_index - 1)
            return False

    # -- status --

    def get_status(self) -> PlaybackStatus:
        current_title = ""
        if 0 <= self._playlist_index < len(self._playlist):
            item = self._playlist[self._playlist_index]
            current_title = item.get("title", "")
        return PlaybackStatus(
            playing=not self._player.core_idle,
            paused=bool(self._player.pause),
            current_file=current_title,
            title=self._player.media_title or "",
            duration=self._player.duration or 0.0,
            position=self._player.time_pos or 0.0,
            volume=int(self._player.volume or 0),
        )

    def shutdown(self) -> None:
        self._player.terminate()

    # -- library search (static helper) --

    @classmethod
    def search_library(cls, library_path: str, query: str) -> list[str]:
        """Search local music library by filename matching.

        Returns list of file paths. If query is empty, returns shuffled library.
        """
        if not library_path or not Path(library_path).is_dir():
            return []

        keywords = [k.strip().lower() for k in query.split() if k.strip()]
        results = []

        for f in Path(library_path).rglob("*"):
            if f.is_file() and f.suffix.lower() in cls._MUSIC_EXTENSIONS:
                name_lower = f.name.lower()
                if not keywords or all(k in name_lower for k in keywords):
                    results.append(str(f))

        if not keywords and results:
            random.shuffle(results)

        return results
