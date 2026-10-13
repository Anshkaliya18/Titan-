"""
Intent Module

Determines what the user wants to do from corrected tokens.
Uses simple rule-based keyword matching — no ML or LLM.

Priority order:
1. Known application commands (checks application registry first,
   then URL structure, then file extensions)
2. System commands (lock, shutdown, restart)
3. File/folder commands
4. Browser commands (open browser, open URL, web search)
5. General application commands (single keyword)
"""

from enum import Enum

from core.application_manager import is_application
from core.command_registry import is_valid_intent
from utils.logger import get_logger

logger = get_logger(__name__)


class Intent(Enum):
    """Enumeration of all possible user intents."""

    OPEN_APPLICATION = "open_application"
    CLOSE_APPLICATION = "close_application"
    LOCK_COMPUTER = "lock_computer"
    SHUTDOWN_COMPUTER = "shutdown_computer"
    RESTART_COMPUTER = "restart_computer"
    ENABLE_STARTUP = "enable_startup"
    DISABLE_STARTUP = "disable_startup"
    STARTUP_STATUS = "startup_status"
    CREATE_FOLDER = "create_folder"
    DELETE_FILE = "delete_file"
    OPEN_FILE = "open_file"
    SEARCH_FILE = "search_file"
    OPEN_BROWSER = "open_browser"
    OPEN_URL = "open_url"
    SEARCH_WEB = "search_web"
    HELP = "help"
    LIST_COMMANDS = "list_commands"
    HISTORY = "history"
    CLEAR_HISTORY = "clear_history"
    FIND_APPLICATION = "find_application"
    REFRESH_APPLICATIONS = "refresh_applications"
    UNKNOWN = "unknown"


# Common file extensions used to identify file targets
FILE_EXTENSIONS = {
    ".txt", ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".mp3", ".mp4", ".avi",
    ".zip", ".rar", ".7z", ".csv", ".json", ".xml", ".html", ".htm",
    ".py", ".java", ".cpp", ".c", ".js", ".ts", ".md", ".rtf",
    ".exe", ".msi", ".bat", ".ps1",
}

# Action keywords for opening
OPEN_KEYWORDS = {"open", "launch", "start"}

# Action keywords for closing
CLOSE_KEYWORDS = {"close", "quit", "exit"}

# URL schemes recognized in user input (the browser manager
# allows exactly these schemes as well)
URL_SCHEMES = {"http", "https"}


def _has_file_extension(target: str) -> bool:
    """Check if a target has a known file extension."""
    target_lower = target.lower()
    return any(target_lower.endswith(ext) for ext in FILE_EXTENSIONS)


def extract_url_from_tokens(tokens: list[str]) -> str | None:
    """
    Reassemble a URL from tokens.

    The tokenizer splits "https://github.com" into
    ['https', ':', '//github.com'], so a URL is recognized by
    finding a scheme token immediately followed by ':' and
    concatenating the remaining pieces back together.

    Args:
        tokens (list[str]): Tokens to scan (typically the tokens
            after the action keyword).

    Returns:
        The reassembled URL, or None when no URL is present.
    """
    for i, token in enumerate(tokens):
        if (
            token.lower() in URL_SCHEMES
            and i + 1 < len(tokens)
            and tokens[i + 1] == ":"
        ):
            rest = "".join(tokens[i + 2:])
            if rest:
                return f"{token.lower()}:{rest}"
    return None


def detect_intent(tokens: list[str]) -> Intent:
    """
    Detect the user's intent from corrected tokens.

    Uses rule-based keyword matching, then validates the result
    against the Command Registry.

    Args:
        tokens (list[str]): Corrected tokens from the spell checker.

    Returns:
        Intent: The detected intent, or Intent.UNKNOWN if no match
        or if the intent is not registered.
    """
    intent = _detect_intent_rules(tokens)
    if not is_valid_intent(intent.value):
        logger.debug("Detected intent: UNKNOWN")
        return Intent.UNKNOWN
    logger.debug(f"Detected intent: {intent.value}")
    return intent


def _detect_intent_rules(tokens: list[str]) -> Intent:
    """
    Apply deterministic rule-based intent detection.

    Priority order:
    1. Command-history and help meta-commands
    2. Known application commands (application registry first)
    3. System commands
    4. File/folder commands
    5. Browser commands
    6. General application commands

    Args:
        tokens (list[str]): Corrected tokens from the spell checker.

    Returns:
        Intent: The detected intent, or Intent.UNKNOWN if no match.
    """
    if not tokens:
        return Intent.UNKNOWN

    first_token = tokens[0].lower()

    token_set = set(tokens)

    # Read-only installed-app discovery and in-memory refresh controls.
    if {"refresh", "applications"}.issubset(token_set) or {"refresh", "apps"}.issubset(token_set):
        return Intent.REFRESH_APPLICATIONS
    if (token_set <= {"find", "installed", "apps", "applications"}
            and ("find" in token_set or "installed" in token_set)) or (
            {"what", "apps", "do", "i", "have"}.issubset(token_set)
    ):
        return Intent.FIND_APPLICATION
    if first_token in {"find", "search"} and len(tokens) > 1 and any(
        word in token_set for word in {"app", "apps", "application", "applications"}
    ):
        return Intent.FIND_APPLICATION

    # Priority 0: Command-history management (Step 20). Checked before
    # the generic help/list rules so "recent commands" resolves to
    # HISTORY instead of LIST_COMMANDS, and before every other rule.
    # Exact deterministic token-set matching only — no fuzzy matching.
    # "search history" deliberately does NOT match: it falls through
    # like any other search query (bare-search rule below).
    if (
        {"clear", "history"}.issubset(token_set)
        or {"delete", "history"}.issubset(token_set)
        or {"remove", "history"}.issubset(token_set)
    ):
        return Intent.CLEAR_HISTORY
    if (
        token_set == {"history"}
        or {"show", "history"}.issubset(token_set)
        or {"command", "history"}.issubset(token_set)
        or {"recent", "commands"}.issubset(token_set)
    ):
        return Intent.HISTORY

    # Priority 0b: Help / command-discovery commands.
    # These are meta-commands and are checked first so that phrases
    # like "show help" resolve to HELP rather than falling through
    # to application or file interpretation.
    if "help" in token_set:
        return Intent.HELP
    if {"what", "can", "you", "do"}.issubset(token_set):
        return Intent.HELP
    if {"how", "can", "you", "help"}.issubset(token_set):
        return Intent.HELP
    if "commands" in token_set:
        return Intent.LIST_COMMANDS

    # Priority 0c: Windows startup management (Step 23). Deterministic
    # token-set matching only — no fuzzy matching. Checked BEFORE the
    # application rules so "start with windows" can never be read as an
    # application launch ("start" is an open keyword).
    if {"enable", "startup"}.issubset(token_set) or (
        token_set == {"start", "with", "windows"}
    ):
        return Intent.ENABLE_STARTUP
    if (
        {"disable", "startup"}.issubset(token_set)
        or {"stop", "startup"}.issubset(token_set)
    ):
        return Intent.DISABLE_STARTUP
    if (
        token_set == {"startup"}
        or {"startup", "status"}.issubset(token_set)
        or {"check", "startup"}.issubset(token_set)
        or {"is", "startup", "enabled"}.issubset(token_set)
    ):
        return Intent.STARTUP_STATUS

    # Priority 1: Known application commands
    # Check if the target (all tokens after the action) matches
    # a known application alias. This must happen BEFORE generic
    # file interpretation so "open file explorer" is recognized
    # as an application, not a file operation.
    if first_token in OPEN_KEYWORDS:
        rest = tokens[1:]
        target = " ".join(rest)
        # "open url" is a registered OPEN_URL alias — catch it before
        # the application check so it is not misread as an app name.
        if "url" in token_set:
            return Intent.OPEN_URL
        if is_application(target):
            return Intent.OPEN_APPLICATION
        # URL structure beats file-extension checks — a URL path
        # may end in something like .pdf, which must stay OPEN_URL
        if extract_url_from_tokens(rest):
            return Intent.OPEN_URL
        # If target has a file extension, it's a file
        if _has_file_extension(target):
            return Intent.OPEN_FILE

    if first_token in CLOSE_KEYWORDS:
        target = " ".join(tokens[1:])
        if is_application(target):
            return Intent.CLOSE_APPLICATION

    # Priority 2: System commands.
    # Each rule matches a specific action word IN THE CONTEXT of
    # "computer" — never a bare "computer" check on its own.
    if {"lock", "computer"}.issubset(token_set):
        return Intent.LOCK_COMPUTER
    if {"shutdown", "computer"}.issubset(token_set):
        return Intent.SHUTDOWN_COMPUTER
    if {"shut", "down", "computer"}.issubset(token_set):
        return Intent.SHUTDOWN_COMPUTER
    if {"restart", "computer"}.issubset(token_set):
        return Intent.RESTART_COMPUTER
    if {"reboot", "computer"}.issubset(token_set):
        return Intent.RESTART_COMPUTER

    # Bare system action words — registered aliases that are natural
    # to type ("lock", "shutdown", "shut down", "power off", "restart",
    # "reboot"). Dangerous commands still park as confirmation
    # downstream, so recognizing them here is safe.
    if "lock" in token_set:
        return Intent.LOCK_COMPUTER
    if (
        "shutdown" in token_set
        or {"shut", "down"}.issubset(token_set)
        or {"power", "off"}.issubset(token_set)
    ):
        return Intent.SHUTDOWN_COMPUTER
    if "restart" in token_set or "reboot" in token_set:
        return Intent.RESTART_COMPUTER

    # Priority 3: File/folder commands
    if {"create", "folder"}.issubset(token_set):
        return Intent.CREATE_FOLDER
    if {"make", "folder"}.issubset(token_set):
        return Intent.CREATE_FOLDER
    if {"delete", "file"}.issubset(token_set):
        return Intent.DELETE_FILE
    if {"remove", "file"}.issubset(token_set):
        return Intent.DELETE_FILE
    if {"search", "file"}.issubset(token_set):
        return Intent.SEARCH_FILE
    if {"find", "file"}.issubset(token_set):
        return Intent.SEARCH_FILE
    if {"open", "file"}.issubset(token_set):
        return Intent.OPEN_FILE

    # Priority 4: Browser commands
    if {"open", "browser"}.issubset(token_set):
        return Intent.OPEN_BROWSER
    if {"launch", "browser"}.issubset(token_set):
        return Intent.OPEN_BROWSER

    # Web search: "search web X" / "search the web for X"
    if {"search", "web"}.issubset(token_set):
        return Intent.SEARCH_WEB

    # Bare search query without "web": "search python tutorials".
    # A query is required — a lone "search" is not a command.
    if first_token == "search" and len(tokens) > 1:
        return Intent.SEARCH_WEB

    # Web search started with "google": "google python tutorials".
    # "google chrome" stays an application name, and a bare "google"
    # (no query) is not a search.
    if first_token == "google" and len(tokens) > 1:
        rest = " ".join(tokens[1:])
        if not is_application(rest):
            return Intent.SEARCH_WEB

    # Priority 5: General application commands (single keyword)
    if first_token in OPEN_KEYWORDS:
        return Intent.OPEN_APPLICATION
    if first_token in CLOSE_KEYWORDS:
        return Intent.CLOSE_APPLICATION

    return Intent.UNKNOWN
