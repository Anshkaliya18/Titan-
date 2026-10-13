"""Output-only, local text-to-speech for final TITAN responses."""

import re
from typing import Callable

from core.config_manager import get_value
from utils.logger import get_logger

logger = get_logger(__name__)

_ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SECRET = re.compile(
    r"(?i)\b(?:GEMINI_API_KEY|OPENAI_API_KEY|API_KEY|ACCESS_TOKEN|"
    r"AUTH_TOKEN|PASSWORD|SECRET)\s*[:=]\s*[^\s,;]+|"
    r"\bAIza[0-9A-Za-z_-]{20,}\b|\bsk-[0-9A-Za-z_-]{20,}\b|"
    r"\bgh[pousr]_[0-9A-Za-z]{20,}\b|\bgithub_pat_[0-9A-Za-z_]{20,}\b"
)
_WINDOWS_PATH = re.compile(r"(?i)\b[A-Z]:\\(?:[^\s<>|\"]+\\?)*")
_INTERNAL_MARKER = re.compile(
    r"(?i)\b(?:DEBUG|TRACEBACK|INTERNAL_ERROR|TTS_UNAVAILABLE|"
    r"PROVIDER_ERROR|API_KEY_MISSING)\b[^\n]*"
)


def sanitize_speech_text(text: str | None) -> str:
    """Remove terminal controls and obvious internal/secrets from text."""
    if not isinstance(text, str) or not text.strip():
        return ""
    cleaned = _ANSI.sub(" ", text)
    cleaned = _CONTROL.sub(" ", cleaned)
    # Structured payloads and tracebacks are never useful as speech.
    if cleaned.lstrip().startswith(("{", "[")) or "Traceback (most recent call last):" in cleaned:
        return ""
    cleaned = _SECRET.sub("[redacted]", cleaned)
    cleaned = _INTERNAL_MARKER.sub(" ", cleaned)
    cleaned = _WINDOWS_PATH.sub(" ", cleaned)
    return " ".join(cleaned.split()).strip()


class TTSManager:
    """Speak final response strings through local pyttsx3/SAPI only.

    This class has no parser, executor, shell, or file-system authority.
    The pyttsx3 package and audio engine are loaded only on the first
    enabled, non-empty speech request.
    """

    def __init__(self, engine_factory: Callable | None = None):
        self._engine_factory = engine_factory
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            if self._engine_factory is None:
                import pyttsx3

                self._engine_factory = pyttsx3.init
            self._engine = self._engine_factory()
            self._engine.setProperty("rate", get_value("tts", "rate", 175))
            self._engine.setProperty("volume", get_value("tts", "volume", 1.0))
            voice_name = get_value("tts", "voice")
            if voice_name:
                voices = self._engine.getProperty("voices")
                match = next(
                    (
                        voice for voice in voices
                        if voice_name.casefold() in {
                            str(getattr(voice, "id", "")).casefold(),
                            str(getattr(voice, "name", "")).casefold(),
                        }
                    ),
                    None,
                )
                if match is None:
                    self._engine = None
                    raise LookupError("Configured TTS voice is unavailable")
                self._engine.setProperty("voice", match.id)
            logger.info("Local TTS initialized")
        return self._engine

    def speak(self, text: str | None) -> dict:
        """Speak text if enabled; failures never escape into command flow."""
        if get_value("tts", "enabled", False) is not True:
            return {"success": False, "message": "Speech is disabled.", "error": "DISABLED"}
        safe_text = sanitize_speech_text(text)
        if not safe_text:
            return {"success": False, "message": "No speech to play.", "error": "EMPTY_TEXT"}
        try:
            engine = self._get_engine()
            engine.say(safe_text)
            engine.runAndWait()
            return {"success": True, "message": "Speech completed.", "error": None}
        except Exception as exc:
            # Do not log response text or engine exception strings.
            logger.warning("Local TTS unavailable (%s)", type(exc).__name__)
            self._engine = None
            return {
                "success": False,
                "message": "Unable to speak response.",
                "error": "TTS_UNAVAILABLE",
            }

    def stop(self) -> None:
        """Stop active local speech during orderly application shutdown."""
        engine = self._engine
        if engine is None:
            return
        try:
            engine.stop()
        except Exception as exc:
            logger.warning("Local TTS shutdown failed (%s)", type(exc).__name__)
