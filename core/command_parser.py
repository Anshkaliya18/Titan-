"""
Command Parser Module

Combines the detected intent with corrected tokens to produce
a structured command that can later be passed to an executor.

This module does NOT execute anything — it only parses.
"""

from core.intent import Intent, extract_url_from_tokens
from core.command_registry import is_valid_intent
from utils.logger import get_logger

logger = get_logger(__name__)


# Intents that operate without a target — the action itself is the
# whole command (system operations, opening the default browser,
# help / command-discovery, and command history management)
TARGETLESS_INTENTS = {
    Intent.LOCK_COMPUTER,
    Intent.SHUTDOWN_COMPUTER,
    Intent.RESTART_COMPUTER,
    Intent.ENABLE_STARTUP,
    Intent.DISABLE_STARTUP,
    Intent.STARTUP_STATUS,
    Intent.OPEN_BROWSER,
    Intent.HELP,
    Intent.LIST_COMMANDS,
    Intent.HISTORY,
    Intent.CLEAR_HISTORY,
    Intent.REFRESH_APPLICATIONS,
}

# Keywords to remove from tokens when extracting the target, per intent
INTENT_KEYWORDS = {
    Intent.OPEN_APPLICATION: {"open", "launch", "start"},
    Intent.CLOSE_APPLICATION: {"close", "quit", "exit"},
    Intent.CREATE_FOLDER: {"create", "make", "folder"},
    Intent.DELETE_FILE: {"delete", "remove", "file"},
    Intent.OPEN_FILE: {"open", "file"},
    Intent.SEARCH_FILE: {"search", "find", "file"},
    Intent.SEARCH_WEB: {"search", "web", "the", "for", "google"},
    Intent.FIND_APPLICATION: {"find", "search", "installed", "app", "apps", "application", "applications", "containing"},
}


def parse_command(tokens: list[str], intent: Intent) -> dict:
    """
    Parse tokens and intent into a structured command.

    Args:
        tokens (list[str]): Corrected tokens from the spell checker.
        intent (Intent): The detected intent.

    Returns:
        dict: A structured command with intent, target, and arguments.
    """
    command = _build_command(tokens, intent)
    # Intent only — the target may echo user-typed content.
    logger.debug(f"Command parsed successfully: {command.get('intent')}")
    return command


def _build_command(tokens: list[str], intent: Intent) -> dict:
    """Build the structured command (see parse_command)."""
    # Reject intents that are not registered in the Command Registry
    if not is_valid_intent(intent.value):
        return {
            "intent": Intent.UNKNOWN.value,
            "target": None,
            "arguments": {},
        }

    if intent == Intent.UNKNOWN:
        return {
            "intent": intent.value,
            "target": None,
            "arguments": {},
        }

    # Targetless commands: system operations and opening the
    # default browser act without a target
    if intent in TARGETLESS_INTENTS:
        return {
            "intent": intent.value,
            "target": None,
            "arguments": {},
        }

    if intent == Intent.FIND_APPLICATION:
        lowered = {word.casefold() for word in tokens}
        if lowered <= {"find", "installed", "apps", "applications"} or lowered == {
            "what", "apps", "do", "i", "have"
        }:
            return {"intent": intent.value, "target": None, "arguments": {}}

    # A URL is reassembled from the tokenizer's fragments
    # (e.g. ['https', ':', '//github.com'] -> 'https://github.com')
    if intent == Intent.OPEN_URL:
        return {
            "intent": intent.value,
            "target": extract_url_from_tokens(tokens),
            "arguments": {},
        }

    # Get the keywords to remove for this intent
    keywords_to_remove = INTENT_KEYWORDS.get(intent, set())

    # Extract the target by removing command keywords
    target_words = [word for word in tokens if word.lower() not in keywords_to_remove]
    target = " ".join(target_words) if target_words else None

    return {
        "intent": intent.value,
        "target": target,
        "arguments": {},
    }
