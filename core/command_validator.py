"""
Command Validator Module

Validates a structured command before it reaches the Executor.

This module does NOT execute commands. It only checks data.

Architecture:
    Command Registry
           ↑
    Command Validator
           ↑
    Command Parser / Executor

The validator performs basic structural validation only:
- the intent is registered
- required targets are present and well-formed
- arguments are a dictionary
- URLs use allowed schemes

It does NOT check whether applications are installed, whether files
exist, or whether paths are safe — those remain the responsibility of
the specialized managers.
"""

from urllib.parse import urlparse

from core.command_registry import is_valid_intent
from utils.logger import get_logger

logger = get_logger(__name__)

# Intents that require a non-empty string target
TARGET_REQUIRED_INTENTS = {
    "OPEN_APPLICATION",
    "CLOSE_APPLICATION",
    "FIND_APPLICATION",
    "REFRESH_APPLICATIONS",
    "CREATE_FOLDER",
    "DELETE_FILE",
    "OPEN_FILE",
    "SEARCH_FILE",
    "OPEN_URL",
    "SEARCH_WEB",
}

# URL schemes allowed for OPEN_URL
ALLOWED_URL_SCHEMES = {"http", "https"}


def validate_command(command) -> dict:
    """
    Validate a structured command before execution.

    Args:
        command: The structured command to validate, normally a dict:
            {
                "intent": str,
                "target": str | None,
                "arguments": dict,
            }

    Returns:
        dict: A structured validation result:
            {
                "valid": bool,
                "message": str,
                "errors": list[str],
            }
    """
    # --- Structure -------------------------------------------------------
    if not isinstance(command, dict):
        logger.warning("Validation failed: command is not a dictionary")
        return _result(
            False,
            "Command must be a dictionary.",
            ["Command is not a dictionary."],
        )

    # --- Intent ----------------------------------------------------------
    intent = command.get("intent")

    if intent is None:
        logger.warning("Validation failed: missing intent")
        return _result(
            False,
            "Command is missing 'intent'.",
            ["Command is missing 'intent'."],
        )

    if not isinstance(intent, str) or not intent.strip():
        logger.warning("Validation failed: invalid intent value")
        return _result(
            False,
            "'intent' must be a non-empty string.",
            ["'intent' must be a non-empty string."],
        )

    intent = intent.strip().upper()

    if not is_valid_intent(intent):
        logger.warning(f"Validation failed: unknown intent '{intent}'")
        return _result(
            False,
            f"Unknown intent: '{intent}'.",
            [f"Unknown intent: '{intent}'."],
        )

    # UNKNOWN is registered but must never be executed
    if intent == "UNKNOWN":
        logger.warning("Validation failed: UNKNOWN is not executable")
        return _result(
            False,
            "Unknown command.",
            ["UNKNOWN is not an executable command."],
        )

    # --- Arguments -------------------------------------------------------
    arguments = command.get("arguments")
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        logger.warning("Validation failed: arguments is not a dictionary")
        return _result(
            False,
            "'arguments' must be a dictionary.",
            ["'arguments' must be a dictionary."],
        )

    # --- Target ----------------------------------------------------------
    target = command.get("target")

    if intent in TARGET_REQUIRED_INTENTS:
        if not _is_valid_target(target):
            logger.warning(f"Validation failed: {intent} requires a target")
            return _result(
                False,
                f"{intent.title().replace('_', ' ')} requires a target.",
                [f"{intent} requires a target."],
            )

    # --- URL -------------------------------------------------------------
    if intent == "OPEN_URL":
        url_error = _validate_url(target)
        if url_error is not None:
            logger.warning(f"Validation failed: {url_error}")
            return _result(False, "Invalid URL.", [url_error])

    logger.debug(f"Command validation passed: {intent}")
    return _result(True, "Command is valid.", [])


def _is_valid_target(target) -> bool:
    """
    Check that a target is a non-empty string.

    None, empty strings, whitespace-only strings, and non-string
    values are all rejected.
    """
    return isinstance(target, str) and bool(target.strip())


def _validate_url(url) -> str | None:
    """
    Validate a URL target for OPEN_URL.

    Only http and https schemes are allowed, and a network location
    (host) is required. Returns an error message, or None if valid.
    """
    if not isinstance(url, str) or not url.strip():
        return "OPEN_URL requires a URL target."

    parsed = urlparse(url.strip())

    if parsed.scheme.lower() not in ALLOWED_URL_SCHEMES:
        return f"URL scheme '{parsed.scheme}' is not allowed."

    if not parsed.netloc:
        return "URL is missing a host."

    return None


def _result(valid: bool, message: str, errors: list[str]) -> dict:
    """
    Build a consistent validation result structure.

    Args:
        valid: Whether the command passed validation.
        message: Short user-facing message.
        errors: List of specific validation error strings.

    Returns:
        dict: {valid, message, errors}
    """
    return {
        "valid": valid,
        "message": message,
        "errors": errors,
    }
