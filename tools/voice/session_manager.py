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


class VoiceSessionManager:
    """Manages the lifecycle of a local voice interaction session.

    State machine:
        IDLE ──wake──▶ LISTENING ──asr_done──▶ PROCESSING ──response──▶ SPEAKING ──tts_done──▶ (IDLE | LISTENING)

    Supports multi-turn continuation: after SPEAKING, if the user speaks
    within ``continuation_timeout`` seconds, we skip the wake word and
    go directly back to LISTENING.
    """

    def __init__(
        self,
        on_wake: Callable[[bytes], Awaitable[None]],
        on_listen_complete: Callable[[str], Awaitable[None]],
        on_speaking_complete: Callable[[], Awaitable[None]] | None = None,
        continuation_timeout: float = 30.0,
        max_continuation_rounds: int = 3,
    ) -> None:
        self._on_wake = on_wake
        self._on_listen_complete = on_listen_complete
        self._on_speaking_complete = on_speaking_complete
        self._continuation_timeout = continuation_timeout
        self._max_continuation_rounds = max_continuation_rounds

        self._state = _State.IDLE
        self._state_lock = asyncio.Lock()
        self._continuation_count = 0
        self._continuation_timer: asyncio.Task | None = None
        self._listen_task: asyncio.Task | None = None
        self._stopped = False

    @property
    def state(self) -> str:
        return self._state.name.lower()

    @property
    def is_idle(self) -> bool:
        return self._state == _State.IDLE

    async def handle_wake(self, pre_audio: bytes) -> None:
        """Called by WakeWordDetector when wake word is detected."""
        async with self._state_lock:
            if self._stopped:
                return
            if self._state != _State.IDLE:
                logger.debug("Wake word ignored: state=%s", self._state.name)
                return

            self._state = _State.LISTENING
            self._continuation_count = 0
            logger.info("Wake word triggered → LISTENING")

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
            return

        async with self._state_lock:
            if self._stopped:
                return
            if self._state != _State.LISTENING:
                return
            self._state = _State.PROCESSING
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
        if old_state == _State.LISTENING and self._listen_task and not self._listen_task.done():
            self._listen_task.cancel()
        if self._continuation_timer and not self._continuation_timer.done():
            self._continuation_timer.cancel()
        logger.info("Session reset to IDLE")
