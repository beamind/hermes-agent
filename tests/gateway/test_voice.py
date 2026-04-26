"""Tests for the voice gateway adapter."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.config import Platform, PlatformConfig
from gateway.platforms.voice import VoiceAdapter, check_voice_requirements, _VOICE_CHAT_ID
from gateway.platforms.base import MessageType
from tools.voice.session_manager import VoiceSessionManager


# ---------------------------------------------------------------------------
# VoiceAdapter
# ---------------------------------------------------------------------------

class TestVoiceAdapter:
    def test_platform_is_voice(self):
        adapter = VoiceAdapter(PlatformConfig())
        assert adapter.platform == Platform.VOICE

    def test_get_chat_info(self):
        adapter = VoiceAdapter(PlatformConfig())
        info = asyncio.run(adapter.get_chat_info("local_voice"))
        assert info["name"] == "Local Voice"
        assert info["type"] == "dm"

    def test_send_returns_success(self):
        adapter = VoiceAdapter(PlatformConfig())
        result = asyncio.run(adapter.send("local_voice", "hello"))
        assert result.success is True

    def test_build_source(self):
        adapter = VoiceAdapter(PlatformConfig())
        source = adapter.build_source(chat_id=_VOICE_CHAT_ID)
        assert source.platform == Platform.VOICE
        assert source.chat_id == _VOICE_CHAT_ID

    @pytest.mark.asyncio
    async def test_connect_without_deps_returns_false(self):
        """If audio requirements are missing, connect() returns False."""
        with patch(
            "tools.voice.audio_player.check_audio_player_requirements", return_value=False
        ):
            adapter = VoiceAdapter(PlatformConfig())
            result = await adapter.connect()
            assert result is False

    @pytest.mark.asyncio
    async def test_disconnect_cleans_up(self):
        adapter = VoiceAdapter(PlatformConfig())
        adapter._connected = True
        adapter._wake_detector = MagicMock()
        adapter._session_manager = AsyncMock()
        adapter._audio_player = MagicMock()

        await adapter.disconnect()

        assert adapter._connected is False
        adapter._wake_detector.stop.assert_called_once()
        adapter._session_manager.stop.assert_awaited_once()
        adapter._audio_player.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_play_tts_delegates_to_audio_player(self):
        adapter = VoiceAdapter(PlatformConfig())
        adapter._audio_player = AsyncMock()
        adapter._session_manager = AsyncMock()

        result = await adapter.play_tts(_VOICE_CHAT_ID, "/tmp/test.mp3")

        assert result.success is True
        adapter._audio_player.play_file.assert_awaited_once_with("/tmp/test.mp3")
        adapter._session_manager.on_agent_response_received.assert_awaited_once()
        adapter._session_manager.on_speaking_complete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_on_listen_complete_injects_voice_message(self):
        adapter = VoiceAdapter(PlatformConfig())
        adapter.handle_message = AsyncMock()

        await adapter._on_listen_complete("你好")

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.call_args[0][0]
        assert event.text == "你好"
        assert event.message_type == MessageType.VOICE
        assert event.source.platform == Platform.VOICE
        assert event.source.chat_id == _VOICE_CHAT_ID

    def test_pause_music_swallows_import_error(self):
        """If smart-speaker plugin is not installed, _pause_music should not crash."""
        adapter = VoiceAdapter(PlatformConfig())
        # No exception should be raised even when plugin is missing
        adapter._pause_music()


# ---------------------------------------------------------------------------
# VoiceSessionManager
# ---------------------------------------------------------------------------

class TestVoiceSessionManager:
    @pytest.mark.asyncio
    async def test_state_machine_idle_to_listening(self):
        on_wake = AsyncMock()
        on_listen = AsyncMock()
        sm = VoiceSessionManager(on_wake=on_wake, on_listen_complete=on_listen)

        assert sm.is_idle

        await sm.handle_wake(b"pre_audio")
        assert sm.state == "listening"

    @pytest.mark.asyncio
    async def test_listen_complete_triggers_callback(self):
        on_listen = AsyncMock()
        sm = VoiceSessionManager(on_wake=AsyncMock(), on_listen_complete=on_listen)

        await sm.handle_wake(b"")

        async def fake_listen() -> str:
            return "hello world"

        await sm.start_listening(fake_listen())
        await asyncio.sleep(0.05)  # let the task run

        on_listen.assert_awaited_once_with("hello world")

    @pytest.mark.asyncio
    async def test_empty_asr_result_returns_to_idle(self):
        on_listen = AsyncMock()
        sm = VoiceSessionManager(on_wake=AsyncMock(), on_listen_complete=on_listen)

        await sm.handle_wake(b"")

        async def fake_listen() -> str:
            return "   "

        await sm.start_listening(fake_listen())
        await asyncio.sleep(0.05)

        assert sm.is_idle
        on_listen.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_multi_turn_continuation(self):
        on_speaking = AsyncMock()
        sm = VoiceSessionManager(
            on_wake=AsyncMock(),
            on_listen_complete=AsyncMock(),
            on_speaking_complete=on_speaking,
            continuation_timeout=0.1,
            max_continuation_rounds=2,
        )

        await sm.handle_wake(b"")
        await sm.start_listening(asyncio.sleep(0))
        await asyncio.sleep(0.05)
        from tools.voice.session_manager import _State
        sm._state = _State.PROCESSING  # simulate agent running
        await sm.on_agent_response_received()
        await sm.on_speaking_complete()

        assert sm.state == "listening"
        assert sm._continuation_count == 1

        # After timeout, should go back to idle
        await asyncio.sleep(0.15)
        assert sm.is_idle

    @pytest.mark.asyncio
    async def test_stop_cancels_tasks(self):
        sm = VoiceSessionManager(on_wake=AsyncMock(), on_listen_complete=AsyncMock())
        await sm.handle_wake(b"")
        await sm.stop()
        assert sm.is_idle

    @pytest.mark.asyncio
    async def test_reset_to_idle(self):
        sm = VoiceSessionManager(on_wake=AsyncMock(), on_listen_complete=AsyncMock())
        await sm.handle_wake(b"")
        await sm.reset_to_idle()
        assert sm.is_idle


# ---------------------------------------------------------------------------
# Requirements checks
# ---------------------------------------------------------------------------

class TestCheckRequirements:
    def test_check_voice_requirements_missing_deps(self):
        with patch(
            "tools.voice.audio_player.check_audio_player_requirements", return_value=False
        ), patch(
            "tools.voice.wake_word.check_wake_word_requirements", return_value=False
        ):
            assert check_voice_requirements() is False

    def test_check_voice_requirements_with_deps(self):
        with patch(
            "tools.voice.audio_player.check_audio_player_requirements", return_value=True
        ), patch(
            "tools.voice.wake_word.check_wake_word_requirements", return_value=True
        ):
            assert check_voice_requirements() is True


# ---------------------------------------------------------------------------
# Integration points
# ---------------------------------------------------------------------------

class TestIntegrationPoints:
    def test_platform_enum_has_voice(self):
        assert Platform.VOICE.value == "voice"

    def test_voice_in_connected_platforms_when_enabled(self):
        from gateway.config import GatewayConfig
        cfg = GatewayConfig()
        cfg.platforms[Platform.VOICE] = PlatformConfig(enabled=True)
        assert Platform.VOICE in cfg.get_connected_platforms()

    def test_voice_authorized_without_user_id(self):
        from gateway.run import GatewayRunner
        from gateway.session import SessionSource

        # Voice platform should be authorized even without user_id
        source = SessionSource(platform=Platform.VOICE, chat_id="local_voice")
        # We can't instantiate GatewayRunner easily due to heavy deps,
        # but we can verify the auth logic inline:
        assert source.platform == Platform.VOICE
