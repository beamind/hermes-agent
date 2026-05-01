from __future__ import annotations

import asyncio
import logging
from enum import Enum, auto
from typing import Callable, Awaitable

logger = logging.getLogger(__name__)


class _State(Enum):
    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()
    SPEAKING = auto()
    SENSORY_ACTIVE = auto()  # sensory feedback playing (music/sound), wake-word only


class VoiceSessionManager:
    """Manages the lifecycle of a local voice interaction session.

    State machine:
        IDLE ──wake──▶ LISTENING ──asr_done──▶ PROCESSING ──response──▶ SPEAKING ──tts_done──▶ (IDLE | LISTENING)
                                                          └─sensory_feedback──▶ SENSORY_ACTIVE ──wake──▶ LISTENING
                                                                                          └─end──▶ IDLE

    Supports multi-turn continuation: after SPEAKING, if the user speaks
    within ``continuation_timeout`` seconds, we skip the wake word and
    go directly back to LISTENING.

    When sensory feedback is active (e.g. music playing), ASR is stopped
    and only wake-word detection runs.  Hitting the wake word pauses
    playback so the user can issue a voice command without background
    audio interfering with ASR.
    """

    def __init__(
        self,
        on_wake: Callable[[bytes], Awaitable[None]],
        on_listen_complete: Callable[[str], Awaitable[None]],
        on_speaking_complete: Callable[[], Awaitable[None]] | None = None,
        on_pause_sensory: Callable[[], None] | None = None,
        on_resume_sensory: Callable[[], None] | None = None,
        continuation_timeout: float = 30.0,
        max_continuation_rounds: int = 3,
    ) -> None:
        self._on_wake = on_wake
        self._on_listen_complete = on_listen_complete
        self._on_speaking_complete = on_speaking_complete
        self._on_pause_sensory = on_pause_sensory
        self._on_resume_sensory = on_resume_sensory
        self._continuation_timeout = continuation_timeout
        self._max_continuation_rounds = max_continuation_rounds

        self._state = _State.IDLE
        self._state_lock = asyncio.Lock()
        self._continuation_count = 0
        self._continuation_timer: asyncio.Task | None = None
        self._listen_task: asyncio.Task | None = None
        self._stopped = False
        self._interrupted_from_sensory = False

    @property
    def state(self) -> str:
        return self._state.name.lower()

    @property
    def is_idle(self) -> bool:
        return self._state == _State.IDLE

    @property
    def is_sensory_active(self) -> bool:
        return self._state == _State.SENSORY_ACTIVE

    async def handle_wake(self, pre_audio: bytes) -> None:
        """Called by WakeWordDetector when wake word is detected."""
        async with self._state_lock:
            if self._stopped:
                return

            if self._state == _State.SENSORY_ACTIVE:
                # Interrupted during playback: pause sensory feedback, start listening
                self._state = _State.LISTENING
                self._continuation_count = 0
                self._interrupted_from_sensory = True
                logger.info("Wake word during playback → pausing music, starting ASR")
                if self._on_pause_sensory:
                    try:
                        self._on_pause_sensory()
                    except Exception:
                        logger.exception("on_pause_sensory failed")

            elif self._state == _State.IDLE:
                self._state = _State.LISTENING
                self._continuation_count = 0
                self._interrupted_from_sensory = False
                logger.info("Wake word triggered → LISTENING")

            else:
                logger.debug("Wake word ignored: state=%s", self._state.name)
                return

        # Cancel any pending continuation timer
        if self._continuation_timer and not self._continuation_timer.done():
            self._continuation_timer.cancel()
            self._continuation_timer = None

        await self._on_wake(pre_audio)

    async def start_listening(self, listen_coro: Awaitable[str]) -> None:
        """Start the ASR listening phase. Called after wake word or continuation."""
        async with self._state_lock:
            if self._stopped:
                return
            if self._state != _State.LISTENING:
                logger.warning("start_listening called in wrong state: %s", self._state.name)
                return

        self._listen_task = asyncio.create_task(self._do_listen(listen_coro))

    async def _do_listen(self, listen_coro: Awaitable[str]) -> None:
        """Run ASR and transition to PROCESSING on completion."""
        try:
            text = await listen_coro
        except asyncio.CancelledError:
            logger.debug("Listening cancelled")
            async with self._state_lock:
                if self._state == _State.LISTENING:
                    if self._interrupted_from_sensory:
                        self._state = _State.SENSORY_ACTIVE
                        self._interrupted_from_sensory = False
                        logger.info("ASR cancelled during interrupt → resuming playback")
                        if self._on_resume_sensory:
                            try:
                                self._on_resume_sensory()
                            except Exception:
                                logger.exception("on_resume_sensory failed")
                    else:
                        self._state = _State.IDLE
            raise
        except Exception:
            logger.exception("ASR failed")
            async with self._state_lock:
                self._state = _State.IDLE
            return

        if not text or not text.strip():
            logger.info("ASR returned empty text → back to IDLE")
            async with self._state_lock:
                self._state = _State.IDLE
                self._interrupted_from_sensory = False
            return

        async with self._state_lock:
            if self._stopped:
                return
            if self._state != _State.LISTENING:
                return
            self._state = _State.PROCESSING
            self._interrupted_from_sensory = False
            logger.info("ASR result: %r → PROCESSING", text)

        try:
            await self._on_listen_complete(text)
        except Exception:
            logger.exception("Process message handler failed")
            async with self._state_lock:
                self._state = _State.IDLE

    async def on_agent_response_received(self) -> None:
        """Called when the agent response has been received and TTS is about to play."""
        async with self._state_lock:
            if self._stopped:
                return
            if self._state != _State.PROCESSING:
                return
            self._state = _State.SPEAKING
            logger.info("TTS playing → SPEAKING")

    async def on_tts_skipped(self) -> None:
        """Called when the response is delivered without TTS playback.

        Sensory-feedback tools (e.g. play_music) produce their own audio,
        so TTS is suppressed.  We enter SENSORY_ACTIVE state where ASR is
        stopped and only wake-word detection runs.  The wake word can pause
        playback and restart ASR for voice commands.
        """
        async with self._state_lock:
            if self._stopped:
                return
            if self._state != _State.PROCESSING:
                return

            self._state = _State.SENSORY_ACTIVE
            self._continuation_count = 0
            logger.info("Sensory feedback active → SENSORY_ACTIVE, ASR stopped")

    async def on_sensory_finished(self) -> None:
        """Called when sensory feedback (music/sound) finishes naturally."""
        async with self._state_lock:
            if self._stopped:
                return
            if self._state != _State.SENSORY_ACTIVE:
                return
            self._state = _State.IDLE
            self._continuation_count = 0
            self._interrupted_from_sensory = False
            logger.info("Sensory finished → IDLE, waiting for wake word")

    async def on_tool_executed(self, action: str) -> None:
        """Called after a tool executes when the agent was in PROCESSING state.

        Handles both the initial sensory entry (play_music from IDLE path)
        and interrupt commands (next/pause/etc. from SENSORY_ACTIVE path).
        """
        async with self._state_lock:
            if self._stopped:
                return
            if self._state != _State.PROCESSING:
                logger.warning("on_tool_executed called in wrong state: %s", self._state.name)
                return

            if action in {"play", "resume", "next", "previous"}:
                self._state = _State.SENSORY_ACTIVE
                self._continuation_count = 0
                logger.info("Tool %s → SENSORY_ACTIVE", action)
            elif action in {"pause", "stop"}:
                self._state = _State.IDLE
                self._continuation_count = 0
                logger.info("Tool %s → IDLE", action)
            else:
                self._state = _State.SENSORY_ACTIVE
                self._continuation_count = 0
                if self._on_resume_sensory:
                    try:
                        self._on_resume_sensory()
                    except Exception:
                        logger.exception("on_resume_sensory failed")
                logger.info("Tool %s → SENSORY_ACTIVE", action)

    async def on_speaking_complete(self) -> None:
        """Called when TTS playback finishes."""
        async with self._state_lock:
            if self._stopped:
                return
            if self._state != _State.SPEAKING:
                return

            if self._continuation_count < self._max_continuation_rounds:
                self._state = _State.LISTENING
                self._continuation_count += 1
                logger.info(
                    "TTS done → LISTENING (continuation %d/%d, timeout=%.1fs)",
                    self._continuation_count,
                    self._max_continuation_rounds,
                    self._continuation_timeout,
                )
            else:
                self._state = _State.IDLE
                self._continuation_count = 0
                logger.info("TTS done → IDLE (max continuations reached)")
                return

        # Start continuation timer
        if self._continuation_timer and not self._continuation_timer.done():
            self._continuation_timer.cancel()

        async def _timeout() -> None:
            await asyncio.sleep(self._continuation_timeout)
            async with self._state_lock:
                if self._state == _State.LISTENING:
                    logger.info("Continuation timeout → IDLE")
                    self._state = _State.IDLE
                    self._continuation_count = 0

        self._continuation_timer = asyncio.create_task(_timeout())

        if self._on_speaking_complete:
            try:
                await self._on_speaking_complete()
            except Exception:
                logger.exception("on_speaking_complete hook failed")

    async def stop(self) -> None:
        """Stop the session manager and cancel any pending tasks."""
        self._stopped = True
        if self._listen_task and not self._listen_task.done():
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
        if self._continuation_timer and not self._continuation_timer.done():
            self._continuation_timer.cancel()
            try:
                await self._continuation_timer
            except asyncio.CancelledError:
                pass
        async with self._state_lock:
            self._state = _State.IDLE
            self._continuation_count = 0
        logger.info("VoiceSessionManager stopped")

    async def reset_to_idle(self) -> None:
        """Force reset to idle state (e.g. on /new or /reset command)."""
        async with self._state_lock:
            old_state = self._state
            self._state = _State.IDLE
            self._continuation_count = 0
            self._interrupted_from_sensory = False
        if old_state == _State.LISTENING and self._listen_task and not self._listen_task.done():
            self._listen_task.cancel()
        if self._continuation_timer and not self._continuation_timer.done():
            self._continuation_timer.cancel()
        logger.info("Session reset to IDLE")
