"""
Smart Speaker music tools for Hermes.

Provides local music playback and playback control.
Adapted from home-ai-assistant's MusicAgent + MusicPlayer.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from .music_player import MusicPlayer, PlaybackStatus
from tools.registry import tool_error, tool_result

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global player instance — survives across tool calls
# ---------------------------------------------------------------------------

_player: MusicPlayer | None = None


def _get_player() -> MusicPlayer:
    """Return the global MusicPlayer, creating it on first call."""
    global _player
    if _player is None:
        _player = MusicPlayer(default_volume=70)
        logger.info("MusicPlayer initialized")
    return _player


_cached_library_path: str | None = None
_cached_library_path_loaded: bool = False


def _get_library_path() -> str | None:
    """Read music library path from config.yaml, falling back to env var."""
    global _cached_library_path, _cached_library_path_loaded
    if _cached_library_path_loaded:
        return _cached_library_path

    path = None
    try:
        from hermes_cli.config import load_config
        config = load_config()
        path = config.get("smart_speaker", {}).get("music_library_path")
        if path:
            path = str(path)
    except Exception:
        pass
    if not path:
        path = os.getenv("SMART_SPEAKER_MUSIC_LIBRARY_PATH")

    _cached_library_path = path
    _cached_library_path_loaded = True
    return path


def _try_play_intro(song: dict) -> None:
    """Generate and play a spoken intro before the song, if enabled.

    All failures are silently caught — the song plays regardless.
    Uses lazy imports so a missing dependency in song_intro.py
    does not break the music tools.
    """
    try:
        from .song_intro import intro_enabled, generate_and_play_intro
        if not intro_enabled():
            return
        generate_and_play_intro(
            title=song.get("title", ""),
            artist=song.get("artist", ""),
        )
    except ImportError:
        pass
    except Exception:
        logger.exception("Song intro failed, continuing to music")


def _check_music_available() -> bool:
    """Always return True so LLM knows about the music feature.

    The actual availability checks (library path, mpv) are done inside
    each tool handler so that the LLM can guide users to configure
    the plugin when it's not yet set up.
    """
    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _status_to_dict(status: PlaybackStatus) -> dict:
    return {
        "playing": status.playing,
        "paused": status.paused,
        "current_file": status.current_file,
        "title": status.title,
        "duration": round(status.duration, 1),
        "position": round(status.position, 1),
        "volume": status.volume,
    }


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _handle_play_music(args: dict, **kw) -> str:
    """Play music by query from local library or Netease Cloud Music."""
    library_path = _get_library_path()
    if not library_path:
        return tool_error(
            "Music library not configured. "
            "Run: hermes config set smart_speaker.music_library_path /path/to/your/music"
        )

    query = str(args.get("query") or "").strip()
    source = str(args.get("source") or "auto").strip().lower()

    if source not in {"local", "auto", "netease"}:
        return tool_error("source must be one of: local, auto, netease")

    # ---- local search ----
    if source in {"local", "auto"}:
        matches = MusicPlayer.search_library(library_path, query)
        if matches:
            player = _get_player()
            songs = []
            for fp in matches:
                p = Path(fp)
                songs.append({
                    "file_path": fp,
                    "title": p.stem,
                    "artist": "",
                })
            player.load_playlist(songs)
            if songs:
                _try_play_intro(songs[0])
            success = player.play_index(0)
            if success:
                status = player.get_status()
                return tool_result({
                    "success": True,
                    "action": "play",
                    "source": "local",
                    "query": query,
                    "matches_found": len(matches),
                    "now_playing": _status_to_dict(status),
                    "sensory_feedback": "audio",
                })
            return tool_error("Failed to start playback")

        if source == "local":
            return tool_error(f"No local matches for '{query}'")

    # ---- netease fallback ----
    from .netease_music import NeteaseMusic

    netease = NeteaseMusic()
    results = netease.search(query, limit=5)
    if not results:
        return tool_error(
            f"No matches found for '{query}' on Netease Cloud Music"
        )

    player = _get_player()
    songs = []
    for item in results:
        url = netease.get_play_url(item["id"])
        if not url:
            continue

        # Cache to local library
        safe_name = NeteaseMusic._sanitize_filename(
            f"{item['artist']} - {item['name']}"
        ) + ".mp3"
        cache_path = Path(library_path) / safe_name

        file_path: str = url
        if cache_path.exists():
            logger.info("Local cache hit: %s", cache_path)
            file_path = str(cache_path)
        elif netease.download_song(url, cache_path):
            file_path = str(cache_path)
        else:
            logger.warning(
                "Failed to download %s, fallback to streaming", item["name"]
            )

        songs.append({
            "file_path": file_path,
            "title": item["name"],
            "artist": item["artist"],
        })

    if not songs:
        return tool_error(
            f"Found songs on Netease but no playable URLs for '{query}' "
            "(may require login or be VIP-only)"
        )

    player.load_playlist(songs)
    if songs:
        _try_play_intro(songs[0])
    success = player.play_index(0)
    if success:
        status = player.get_status()
        return tool_result({
            "success": True,
            "action": "play",
            "source": "netease",
            "query": query,
            "matches_found": len(results),
            "playable": len(songs),
            "now_playing": _status_to_dict(status),
            "sensory_feedback": "audio",
        })
    return tool_error("Failed to start playback from Netease")


def _handle_control_playback(args: dict, **kw) -> str:
    """Control playback: pause, resume, stop, next, previous, volume."""
    library_path = _get_library_path()
    if not library_path:
        return tool_error(
            "Music library not configured. "
            "Run: hermes config set smart_speaker.music_library_path /path/to/your/music"
        )

    action = str(args.get("action") or "").strip().lower()
    valid_actions = {"pause", "resume", "stop", "next", "previous", "volume_set"}
    if action not in valid_actions:
        return tool_error(f"action must be one of: {', '.join(valid_actions)}")

    player = _get_player()

    if action == "pause":
        player.pause()
        return tool_result({
            "success": True,
            "action": "pause",
            "status": _status_to_dict(player.get_status()),
            "sensory_feedback": "audio",
        })

    if action == "resume":
        player.resume()
        return tool_result({
            "success": True,
            "action": "resume",
            "status": _status_to_dict(player.get_status()),
            "sensory_feedback": "audio",
        })

    if action == "stop":
        player.stop()
        return tool_result({
            "success": True,
            "action": "stop",
            "status": _status_to_dict(player.get_status()),
        })

    if action == "next":
        success = player.next()
        result = {
            "success": success,
            "status": _status_to_dict(player.get_status()),
        }
        if success:
            result["action"] = "next"
            result["sensory_feedback"] = "audio"
        return tool_result(result)

    if action == "previous":
        success = player.previous()
        result = {
            "success": success,
            "status": _status_to_dict(player.get_status()),
        }
        if success:
            result["action"] = "previous"
            result["sensory_feedback"] = "audio"
        return tool_result(result)

    if action == "volume_set":
        volume = args.get("volume")
        if volume is None:
            return tool_error("volume is required for action='volume_set'")
        try:
            vol = max(0, min(100, int(volume)))
        except (ValueError, TypeError):
            return tool_error("volume must be an integer between 0 and 100")
        player.set_volume(vol)
        return tool_result({
            "success": True,
            "action": "volume_set",
            "volume": vol,
            "status": _status_to_dict(player.get_status()),
            "sensory_feedback": "audio",
        })

    return tool_error(f"Unhandled action: {action}")


def _handle_get_playback_status(args: dict, **kw) -> str:
    """Get current playback status."""
    library_path = _get_library_path()
    if not library_path:
        return tool_error(
            "Music library not configured. "
            "Run: hermes config set smart_speaker.music_library_path /path/to/your/music"
        )

    player = _get_player()
    status = player.get_status()
    return tool_result({
        "status": _status_to_dict(status),
    })


# ---------------------------------------------------------------------------
# Schemas (OpenAI function-calling format)
# ---------------------------------------------------------------------------

PLAY_MUSIC_SCHEMA = {
    "name": "play_music",
    "description": (
        "Play music from local library or Netease Cloud Music. "
        "Searches local files by filename matching first (when source=auto). "
        "If no local match, falls back to Netease search, downloads to library, and plays. "
        "If query is empty, plays a random shuffle of the entire local library. "
        "Returns the currently playing track info."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search keywords (song name, artist, or both). Empty = shuffle all.",
            },
            "source": {
                "type": "string",
                "enum": ["local", "auto", "netease"],
                "description": "Where to search. 'local' = local files only. 'auto' = local first, then Netease fallback. 'netease' = Netease Cloud Music only.",
                "default": "auto",
            },
        },
        "required": [],
    },
}

CONTROL_PLAYBACK_SCHEMA = {
    "name": "control_playback",
    "description": (
        "Control music playback: pause, resume, stop, next track, previous track, or set volume."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["pause", "resume", "stop", "next", "previous", "volume_set"],
                "description": "Playback control action",
            },
            "volume": {
                "type": "integer",
                "description": "Volume level (0-100). Required only for action='volume_set'.",
            },
        },
        "required": ["action"],
    },
}

GET_PLAYBACK_STATUS_SCHEMA = {
    "name": "get_playback_status",
    "description": (
        "Get the current playback status: whether music is playing, paused, "
        "current track info, position, duration, and volume."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}
