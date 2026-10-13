"""Bounded local dictation through the installed Windows SAPI recognizer.

All COM and microphone work is lazy and happens only inside ``listen``. The
provider returns text only; it has no access to Jarvis command execution.
"""

from __future__ import annotations

import importlib
import locale
import math
import re
import sys
import time
from typing import Any

from core.speech_recognizer import RecognitionResult
from utils.logger import get_logger

logger = get_logger(__name__)

_RECOGNIZER_CATEGORY = r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Recognizers"
_SRE_RECOGNITION = 0x10
_SRE_HYPOTHESIS = 0x20
_SPRS_INACTIVE = 0
_SPRS_ACTIVE = 1
_SPAS_CLOSED = 0


class _SpeechEvents:
    """Small SAPI event sink. It stores text only, never audio buffers."""

    def __init__(self) -> None:
        self.text = ""
        self.speech_started_at: float | None = None

    def OnHypothesis(self, _stream_number: int, _position: int, _result: Any) -> None:
        if self.speech_started_at is None:
            self.speech_started_at = time.monotonic()

    def OnRecognition(
        self, _stream_number: int, _position: int, _recognition_type: int,
        result: Any,
    ) -> None:
        try:
            self.text = str(result.PhraseInfo.GetText()).strip()
        except Exception:
            self.text = ""


class WindowsSpeechRecognizer:
    """Use the installed Windows SAPI dictation engine for one short phrase."""

    def __init__(self, backend: Any | None = None) -> None:
        # Injectable solely to keep provider behavior testable without COM or
        # physical audio hardware. Production imports remain lazy.
        self._backend = backend

    def listen(
        self,
        *,
        language: str,
        listen_timeout: float,
        phrase_time_limit: float,
        device: int | None = None,
    ) -> RecognitionResult:
        if (
            not isinstance(language, str)
            or not re.fullmatch(r"[A-Za-z]{2,3}-[A-Za-z]{2,4}", language.strip())
            or not _valid_timeout(listen_timeout)
            or not _valid_timeout(phrase_time_limit)
            or (
                device is not None
                and (type(device) is not int or not 0 <= device <= 255)
            )
        ):
            return RecognitionResult(False, error="INVALID_CONFIGURATION")
        if sys.platform != "win32":
            return RecognitionResult(False, error="WINDOWS_SPEECH_REQUIRES_WINDOWS")

        backend = self._backend
        if backend is None:
            try:
                backend = _load_backend()
            except (ImportError, OSError) as exc:
                logger.info("Windows speech bridge unavailable (%s)", type(exc).__name__)
                return RecognitionResult(False, error="SPEECH_UNAVAILABLE")

        co_initialized = False
        recognizer = audio_input = context = grammar = event_sink = None
        activation_started = False
        result = RecognitionResult(False, error="SPEECH_UNAVAILABLE")
        try:
            backend.pythoncom.CoInitialize()
            co_initialized = True

            recognizer, token, language_error = _create_recognizer(backend, language)
            if language_error:
                return RecognitionResult(False, error=language_error)

            audio_input = backend.client.Dispatch("SAPI.SpMMAudioIn")
            audio_input.DeviceId = -1 if device is None else device
            recognizer.AudioInput = audio_input

            context = recognizer.CreateRecoContext()
            context.EventInterests = _SRE_RECOGNITION | _SRE_HYPOTHESIS
            event_sink = backend.client.WithEvents(context, _SpeechEvents)
            grammar = context.CreateGrammar(1)
            grammar.DictationLoad("", 0)

            started = time.monotonic()
            overall_deadline = started + float(listen_timeout) + float(phrase_time_limit)
            activation_started = True
            grammar.DictationSetState(_SPRS_ACTIVE)

            while time.monotonic() < overall_deadline:
                backend.pythoncom.PumpWaitingMessages()
                text = getattr(event_sink, "text", "").strip()
                if text:
                    result = RecognitionResult(True, text=text)
                    break

                speech_started = getattr(event_sink, "speech_started_at", None)
                if speech_started is not None:
                    if time.monotonic() >= min(
                        overall_deadline,
                        speech_started + float(phrase_time_limit),
                    ):
                        result = RecognitionResult(False, error="TIMEOUT")
                        break
                elif time.monotonic() >= started + float(listen_timeout):
                    result = RecognitionResult(False, error="NO_SPEECH")
                    break
                time.sleep(0.01)
            else:
                result = RecognitionResult(
                    False,
                    error="TIMEOUT" if event_sink.speech_started_at is not None else "NO_SPEECH",
                )
        except KeyboardInterrupt:
            result = RecognitionResult(False, error="INTERRUPTED")
        except PermissionError:
            result = RecognitionResult(False, error="PERMISSION_DENIED")
        except (OSError, AssertionError) as exc:
            logger.info("Windows speech device unavailable (%s)", type(exc).__name__)
            result = RecognitionResult(False, error="MICROPHONE_UNAVAILABLE")
        except Exception as exc:
            error = "SPEECH_UNAVAILABLE"
            if activation_started:
                error = "INVALID_DEVICE" if device is not None else "MICROPHONE_UNAVAILABLE"
            logger.info("Windows speech recognition failed (%s)", type(exc).__name__)
            result = RecognitionResult(False, error=error)
        finally:
            # Stop dictation first, then close the SAPI input device. The COM
            # references are released before returning to Jarvis or TTS.
            if grammar is not None:
                try:
                    grammar.DictationSetState(_SPRS_INACTIVE)
                except Exception:
                    pass
            if context is not None:
                try:
                    context.EventInterests = 0
                except Exception:
                    pass
            if event_sink is not None:
                try:
                    event_sink.close()
                except Exception:
                    pass
            if audio_input is not None:
                if recognizer is not None:
                    try:
                        recognizer.AudioInput = None
                    except Exception:
                        pass
                try:
                    audio_input.SetState(_SPAS_CLOSED)
                except Exception:
                    pass
            recognizer = audio_input = context = grammar = event_sink = None
            if co_initialized:
                try:
                    backend.pythoncom.CoUninitialize()
                except Exception:
                    pass
        if not result.success and result.error not in {"NO_SPEECH", "TIMEOUT", "INTERRUPTED"}:
            logger.debug("Windows speech recognition ended (%s)", result.error)
        elif result.success:
            logger.debug("Windows speech recognition received %d characters", len(result.text))
        return result


def _load_backend() -> Any:
    pythoncom = importlib.import_module("pythoncom")
    client = importlib.import_module("win32com.client")
    return type("SapiBackend", (), {"pythoncom": pythoncom, "client": client})


def _create_recognizer(backend: Any, language: str):
    """Find a locally installed recognizer with the exact requested locale."""
    locale_tag = language.strip().replace("_", "-").casefold()
    language_id = next(
        (
            lang_id for lang_id, tag in locale.windows_locale.items()
            if tag.replace("_", "-").casefold() == locale_tag
        ),
        None,
    )
    if language_id is None:
        return None, None, "UNSUPPORTED_LANGUAGE"

    category = backend.client.Dispatch("SAPI.SpObjectTokenCategory")
    category.SetId(_RECOGNIZER_CATEGORY)
    tokens = category.EnumerateTokens()
    token = next(
        (
            candidate for candidate in tokens
            if _token_supports_language(candidate, language_id)
        ),
        None,
    )
    if token is None:
        return None, None, "UNSUPPORTED_LANGUAGE" if tokens else "SPEECH_UNAVAILABLE"

    recognizer = backend.client.Dispatch("SAPI.SpInProcRecognizer")
    recognizer.Recognizer = token
    return recognizer, token, None


def _token_supports_language(token: Any, language_id: int) -> bool:
    try:
        get_attribute = getattr(token, "GetAttribute", None)
        if callable(get_attribute):
            raw = get_attribute("Language")
        else:
            raw = token.Attributes.Item("Language")
        return any(
            int(part.strip(), 16) == language_id
            for part in str(raw).split(";")
            if part.strip()
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _valid_timeout(value: float) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 < value <= 30
    )
