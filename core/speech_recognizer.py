"""Provider-neutral speech recognition result and interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RecognitionResult:
    """Recognized text or a stable, provider-neutral error code."""

    success: bool
    text: str = ""
    error: str | None = None


class SpeechRecognizer(Protocol):
    """Interface implemented by the configured local speech provider."""

    def listen(
        self,
        *,
        language: str,
        listen_timeout: float,
        phrase_time_limit: float,
        device: int | None = None,
    ) -> RecognitionResult: ...
