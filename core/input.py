"""
Input Module

Handles reading user input from the command line
and normalizing it for further processing.
"""

import sys

from core.config_manager import get_value
from core.voice_input import VoiceInput
from core.wake_word import WakeWordDetector
from utils.logger import get_logger

logger = get_logger(__name__)
_voice_input = VoiceInput()
_wake_detector = WakeWordDetector()
_CONTINUE_PROMPT = object()

_VOICE_ERROR_MESSAGES = {
    "DISABLED": "Voice input is disabled. Enable it in config/settings.json.",
    "DEPENDENCY_MISSING": "Windows speech support is unavailable. Install requirements-voice.txt.",
    "SPEECH_UNAVAILABLE": "Windows speech recognition is unavailable on this device.",
    "WINDOWS_SPEECH_REQUIRES_WINDOWS": "Voice input requires Windows.",
    "UNSUPPORTED_LANGUAGE": "The configured speech language is not installed in Windows.",
    "PROVIDER_ERROR": "Windows speech recognition failed. Please try again.",
    "MICROPHONE_UNAVAILABLE": "I couldn't access the microphone.",
    "PERMISSION_DENIED": "Microphone permission was denied.",
    "INVALID_DEVICE": "The microphone device is unavailable.",
    "NO_SPEECH": "No speech detected.",
    "TIMEOUT": "Speech recognition timed out.",
    "PROVIDER_UNAVAILABLE": "Windows speech recognition is unavailable.",
    "INTERRUPTED": "Voice input was interrupted.",
    "INVALID_CONFIGURATION": "Voice input configuration is invalid.",
}


def _assistant_name() -> str:
    """Return the configured assistant name (defaults to 'Jarvis')."""
    name = get_value("general", "assistant_name")
    return name if isinstance(name, str) and name.strip() else "Jarvis"


def get_input() -> str:
    """
    Display the assistant prompt and read user input.

    Only the length of the input is logged — raw input is never
    written to the log because it could contain sensitive content.

    Returns:
        str: The user's input with leading/trailing whitespace removed.

    Exits:
        If EOF (Ctrl+D) or KeyboardInterrupt (Ctrl+C) is encountered,
        the program exits gracefully.
    """
    while True:
        was_wake_mode = _wake_detector.is_running()
        try:
            if was_wake_mode:
                result = _listen_for_wake_command()
                if result is _CONTINUE_PROMPT:
                    continue
                return result

            user_input = input(f"{_assistant_name()} > ")
            folded = user_input.strip().casefold()
            if folded in {"wake", "wake word"}:
                _start_wake_mode()
                continue
            if folded == "wake off":
                print("Wake-word mode: OFF.\n")
                return ""
            if folded == "voice":
                if get_value("voice", "enabled", False) is True:
                    print("Listening...")
                result = _voice_input.listen()
                return _confirm_transcription(result)
            # Length only — never log the raw input itself.
            logger.debug("Received user input (%d characters)", len(user_input))
            return user_input.strip()
        except KeyboardInterrupt:
            if _wake_detector.is_running():
                _wake_detector.stop()
                logger.info("Wake-word mode stopped by user")
                print("\nWake-word mode: OFF. Returning to typed input.\n")
                continue
            logger.info("Input stream ended")
            print("\nGoodbye!")
            sys.exit(0)
        except EOFError:
            if _wake_detector.is_running():
                _wake_detector.stop()
                print("\nWake-word mode: OFF.")
            logger.info("Input stream ended")
            print("\nGoodbye!")
            sys.exit(0)


def _start_wake_mode() -> None:
    if get_value("wake_word", "enabled", False) is not True:
        print("Wake-word mode is disabled in config/settings.json.")
        return
    result = _wake_detector.start()
    if not result.detected:
        messages = {
            "VOICE_DISABLED": "Enable voice input in config/settings.json before wake mode.",
            "MODEL_NOT_INSTALLED": (
                "Wake model missing. Install requirements-voice.txt and run "
                "python scripts/setup_wake_word.py."
            ),
            "INITIALIZATION_FAILED": "Local wake-word detector could not start.",
            "INVALID_CONFIGURATION": "Wake-word configuration is invalid; mode stays off.",
        }
        print(messages.get(result.error, "Wake-word mode could not start."))
        return
    phrase = get_value("wake_word", "phrase", "hey jarvis")
    print("Wake-word mode: ON")
    print(f'Listening for: "{phrase}" (press Ctrl+C to stop)')
    logger.info("Wake-word mode enabled")


def _listen_for_wake_command():
    """Wait for a wake phrase, then run the existing one-shot voice path."""
    while _wake_detector.is_running():
        wake_result = _wake_detector.wait_for_wake()
        if not wake_result.detected:
            if wake_result.error == "STOPPED":
                _wake_detector.stop()
                return _CONTINUE_PROMPT
            messages = {
                "AUDIO_DEVICE_UNAVAILABLE": "Microphone unavailable. Wake-word mode stopped.",
                "DETECTOR_ERROR": "Wake-word detection failed. Wake-word mode stopped.",
                "NOT_RUNNING": "Wake-word mode is off.",
            }
            _wake_detector.stop()
            print(messages.get(wake_result.error, "Wake-word mode stopped safely."))
            print("Wake-word mode: OFF.")
            return _CONTINUE_PROMPT

        print("Wake detected.")
        # wait_for_wake has already closed its microphone before this call,
        # so Step 30 opens the only microphone stream used for transcription.
        speech = _voice_input.listen()
        if not speech.success:
            print(_VOICE_ERROR_MESSAGES.get(speech.error, "Voice input failed. Please try again."))
            if speech.error == "NO_SPEECH":
                continue
            if speech.error == "INTERRUPTED":
                _wake_detector.stop()
                print("Wake-word mode: OFF. Returning to typed input.")
                return _CONTINUE_PROMPT
            _wake_detector.stop()
            return _CONTINUE_PROMPT

        print(f"You said: {speech.text}")
        accepted = input("Run this command? Type y to confirm [y/N]: ")
        if accepted.strip().casefold() not in {"y", "yes"}:
            print("Voice command discarded. Wake-word mode remains on.\n")
            return ""

        normalized = " ".join(speech.text.casefold().split())
        if normalized in {"wake off", "stop listening", "disable wake word"}:
            _wake_detector.stop()
            logger.info("Wake-word mode disabled by user")
            print("Wake-word mode: OFF.\n")
            return ""
        return speech.text.strip()


def _confirm_transcription(result) -> str:
    if result.success:
        print(f"You said: {result.text}")
        accepted = input("Run this command? Type y to confirm [y/N]: ")
        if accepted.strip().casefold() in {"y", "yes"}:
            return result.text.strip()
        print("Voice command discarded.\n")
        return ""
    print(_VOICE_ERROR_MESSAGES.get(result.error, "Voice input failed. Please try again."))
    return ""


def normalize_input(query: str) -> str:
    """
    Normalize the input string before tokenization.

    - Removes leading/trailing whitespace
    - Converts repeated spaces into a single space
    - Converts the command to lowercase
    - Safely handles empty strings

    Args:
        query (str): The raw input string.

    Returns:
        str: The normalized string.
    """
    # Strip leading/trailing whitespace
    query = query.strip()

    # Convert to lowercase
    query = query.lower()

    # Replace multiple spaces with a single space
    query = " ".join(query.split())

    return query
