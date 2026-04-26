from __future__ import annotations

from abc import ABC, abstractmethod


class BaseASR(ABC):
    """Abstract base for ASR (speech-to-text) engines."""

    @abstractmethod
    async def recognize(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        """Recognize speech from pre-recorded audio data and return text."""

    @abstractmethod
    async def recognize_from_microphone(self, pre_audio: bytes = b"") -> str:
        """Stream audio from the microphone and return the recognized sentence.

        Args:
            pre_audio: optional PCM bytes (int16, mono) captured before the
                       mic stream starts (e.g. from a wake-word ring buffer).

        The ASR engine decides when the user has finished speaking
        (via endpoint detection / sentence end). Returns the final text.
        """

    @abstractmethod
    async def test_connection(self) -> bool:
        """Verify the ASR service is reachable and the model works."""
