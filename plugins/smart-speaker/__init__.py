"""
Smart Speaker plugin for Hermes.

Registers music playback tools and lifecycle hooks.
"""

from __future__ import annotations

import logging

from .tools.music_tools import (
    CONTROL_PLAYBACK_SCHEMA,
    GET_PLAYBACK_STATUS_SCHEMA,
    PLAY_MUSIC_SCHEMA,
    _check_music_available,
    _handle_control_playback,
    _handle_get_playback_status,
    _handle_play_music,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifecycle hooks
# ---------------------------------------------------------------------------

def _on_session_end(**kwargs) -> None:
    """Stop playback when the conversation session ends."""
    from plugins.smart_speaker.tools.music_tools import _get_player

    try:
        player = _get_player()
        player.stop()
        logger.info("Playback stopped on session end")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register all smart-speaker tools. Called once by the plugin loader."""
    _TOOLS = (
        ("play_music", PLAY_MUSIC_SCHEMA, _handle_play_music, "🎵", "action"),
        ("control_playback", CONTROL_PLAYBACK_SCHEMA, _handle_control_playback, "🎛️", "action"),
        ("get_playback_status", GET_PLAYBACK_STATUS_SCHEMA, _handle_get_playback_status, "📊", None),
    )

    for name, schema, handler, emoji, vhint in _TOOLS:
        ctx.register_tool(
            name=name,
            toolset="smart_speaker",
            schema=schema,
            handler=handler,
            check_fn=_check_music_available,
            emoji=emoji,
            voice_hint=vhint,
        )
        logger.info("Registered tool: %s", name)

    ctx.register_hook("on_session_end", _on_session_end)
    logger.info("Smart-speaker plugin registered")
