"""
Vocabulary Module

Builds the protected vocabulary used by the spell checker.

Protected words are never spell-corrected. This includes:
- Command keywords (open, close, create, etc.)
- System keywords (computer, folder, file, browser)
- All application names and their component words

This is a reusable mechanism — new applications added to the
registry are automatically protected.
"""

from core.application_manager import APP_NAMES

# Command keywords that should never be spell-corrected
COMMAND_KEYWORDS = {
    "open", "close", "create", "delete", "search",
    "launch", "start", "quit", "exit", "make", "remove", "find",
    # System command keywords
    "lock", "shutdown", "restart", "reboot",
    # Help/command-discovery keywords (Step 15)
    "help", "commands", "available",
    # Windows startup management keywords (Step 23). Protected so the
    # edit-distance model can never rewrite e.g. "startup" to "start"
    # or "enable" to "enabler" — the intent rules match exact words.
    "enable", "disable", "startup", "stop", "check", "status",
}

# System/context words
SYSTEM_KEYWORDS = {
    "computer", "folder", "file", "browser",
    # Browser/URL context words (Step 11)
    "web", "website", "http", "https", "url",
    # Windows startup context words (Step 23)
    "windows", "enabled",
}

# Deterministic corrections for known system-command typos.
# These are explicit one-word fixes — not ML — so a typo like "lok"
# cannot be mis-corrected to an unrelated word ("loka").
COMMAND_TYPOS = {
    "lok": "lock",
}


def get_protected_words() -> set[str]:
    """
    Build the protected words set.

    Includes:
    - Command keywords
    - System keywords
    - All application names (full aliases)
    - All words that make up application names

    Returns:
        set[str]: The set of protected words.
    """
    protected = COMMAND_KEYWORDS | SYSTEM_KEYWORDS

    # Add all application names and their component words
    for app_name in APP_NAMES:
        protected.add(app_name)
        for word in app_name.split():
            protected.add(word)

    return protected


# Module-level protected words set (built once)
PROTECTED_WORDS = get_protected_words()
