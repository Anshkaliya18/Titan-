"""Opt-in, one-shot voice input that returns plain text."""

from __future__ import annotations

from core.config_manager import get_value
from core.speech_recognizer import (
    RecognitionResult,
    SpeechRecognizer,
)
from core.windows_speech_recognizer import WindowsSpeechRecognizer
from utils.logger import get_logger

logger = get_logger(__name__)

MAX_TIMEOUT_SECONDS = 30


class VoiceInput:
    """Lazy voice source; microphone access occurs only on explicit listen."""

    def __init__(self, recognizer: SpeechRecognizer | None = None) -> None:
        self._recognizer = recognizer

    def listen(self) -> RecognitionResult:
        """Listen for one configured phrase, returning text or a safe error."""
        if get_value("voice", "enabled", False) is not True:
            return RecognitionResult(False, error="DISABLED")

        provider = get_value("voice", "provider", "windows")
        language = get_value("voice", "language", "en-US")
        listen_timeout = _bounded_setting("listen_timeout_seconds", 5)
        phrase_limit = _bounded_setting("phrase_time_limit_seconds", 10)
        device = get_value("voice", "device", None)
        if (
            provider != "windows"
            or not isinstance(language, str) or not language.strip()
            or (
                device is not None
                and (type(device) is not int or not 0 <= device <= 255)
            )
        ):
            return RecognitionResult(False, error="INVALID_CONFIGURATION")

        # The provider is constructed lazily and opens a native SAPI input
        # stream only for this explicit, bounded voice request.
        if self._recognizer is None:
            self._recognizer = WindowsSpeechRecognizer()
        return self._recognizer.listen(
            language=language.strip(),
            listen_timeout=listen_timeout,
            phrase_time_limit=phrase_limit,
            device=device,
        )


def _bounded_setting(name: str, default: int) -> int:
    value = get_value("voice", name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return default
    return min(value, MAX_TIMEOUT_SECONDS)
