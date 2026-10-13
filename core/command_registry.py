"""
Command Registry
----------------
Central source of truth for supported TITAN commands.

It defines:
- intent names
- command aliases
- descriptions

It does not execute commands.

Architecture:
    Command Registry
           ↓
    describes commands

    Intent Detector ───────────────→ Command Registry
           ↓
    Command Parser ────────────────→ Command Registry
           ↓
    Executor ──────────────────────→ Command Registry
           ↓
    Specialized Manager
"""

from dataclasses import dataclass

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class CommandDefinition:
    """
    Metadata describing a single supported command.

    Attributes:
        intent: Canonical intent name, e.g. "OPEN_APPLICATION".
        aliases: User-facing phrases that map to this intent.
        description: Short human-readable explanation.
        category: Command category used for help grouping.
    """

    intent: str
    aliases: tuple[str, ...]
    description: str
    category: str = "general"


# ---------------------------------------------------------------------------
# Registered commands
#
# Application names (chrome, notepad, ...) are intentionally NOT listed
# here — they are targets handled by the Application Manager, not command
# aliases.
# ---------------------------------------------------------------------------
_COMMANDS: dict[str, CommandDefinition] = {
    "OPEN_APPLICATION": CommandDefinition(
        intent="OPEN_APPLICATION",
        aliases=("open", "launch", "start"),
        description="Open an installed application.",
        category="application",
    ),
    "CLOSE_APPLICATION": CommandDefinition(
        intent="CLOSE_APPLICATION",
        aliases=("close", "quit", "exit"),
        description="Close a running application.",
        category="application",
    ),
    "FIND_APPLICATION": CommandDefinition(
        intent="FIND_APPLICATION",
        aliases=("find installed apps", "search installed applications", "what apps do i have"),
        description="Find installed applications without opening them.",
        category="application",
    ),
    "REFRESH_APPLICATIONS": CommandDefinition(
        intent="REFRESH_APPLICATIONS",
        aliases=("refresh applications", "refresh app list"),
        description="Refresh the installed application index.",
        category="application",
    ),
    "LOCK_COMPUTER": CommandDefinition(
        intent="LOCK_COMPUTER",
        aliases=("lock", "lock computer"),
        description="Lock the computer.",
        category="system",
    ),
    "SHUTDOWN_COMPUTER": CommandDefinition(
        intent="SHUTDOWN_COMPUTER",
        aliases=("shutdown", "shut down", "power off"),
        description="Shut down the computer.",
        category="system",
    ),
    "RESTART_COMPUTER": CommandDefinition(
        intent="RESTART_COMPUTER",
        aliases=("restart", "reboot"),
        description="Restart the computer.",
        category="system",
    ),
    "ENABLE_STARTUP": CommandDefinition(
        intent="ENABLE_STARTUP",
        aliases=("enable startup", "start with windows", "enable windows startup"),
        description="Start TITAN automatically when Windows starts.",
        category="system",
    ),
    "DISABLE_STARTUP": CommandDefinition(
        intent="DISABLE_STARTUP",
        aliases=("disable startup", "stop startup", "disable windows startup"),
        description="Stop TITAN from starting with Windows.",
        category="system",
    ),
    "STARTUP_STATUS": CommandDefinition(
        intent="STARTUP_STATUS",
        aliases=("startup status", "check startup", "is startup enabled"),
        description="Check whether TITAN starts with Windows.",
        category="system",
    ),
    "CREATE_FOLDER": CommandDefinition(
        intent="CREATE_FOLDER",
        aliases=("create folder", "make folder"),
        description="Create a new folder.",
        category="file",
    ),
    "DELETE_FILE": CommandDefinition(
        intent="DELETE_FILE",
        aliases=("delete file", "remove file"),
        description="Delete a file.",
        category="file",
    ),
    "OPEN_FILE": CommandDefinition(
        intent="OPEN_FILE",
        aliases=("open file",),
        description="Open a file.",
        category="file",
    ),
    "SEARCH_FILE": CommandDefinition(
        intent="SEARCH_FILE",
        aliases=("search file", "find file"),
        description="Search for a file.",
        category="file",
    ),
    "OPEN_BROWSER": CommandDefinition(
        intent="OPEN_BROWSER",
        aliases=("open browser", "launch browser"),
        description="Open the default web browser.",
        category="browser",
    ),
    "SEARCH_WEB": CommandDefinition(
        intent="SEARCH_WEB",
        aliases=("search web", "web search"),
        description="Search the web.",
        category="browser",
    ),
    "OPEN_URL": CommandDefinition(
        intent="OPEN_URL",
        aliases=("open url",),
        description="Open a URL in the default browser.",
        category="browser",
    ),
    "HELP": CommandDefinition(
        intent="HELP",
        aliases=("help", "what can you do", "how can you help", "show help"),
        description="Show help information.",
        category="help",
    ),
    "LIST_COMMANDS": CommandDefinition(
        intent="LIST_COMMANDS",
        aliases=("commands", "list commands", "show commands", "available commands"),
        description="List available TITAN commands.",
        category="help",
    ),
    "HISTORY": CommandDefinition(
        intent="HISTORY",
        aliases=("history", "show history", "command history", "recent commands"),
        description="View recent commands.",
        category="help",
    ),
    "CLEAR_HISTORY": CommandDefinition(
        intent="CLEAR_HISTORY",
        aliases=("clear history", "delete history", "remove history"),
        description="Clear the current session's command history.",
        category="help",
    ),
    "UNKNOWN": CommandDefinition(
        intent="UNKNOWN",
        aliases=(),
        description="No supported command was recognized.",
    ),
}


def get_command(intent: str) -> CommandDefinition | None:
    """
    Get a command definition by intent name.

    Lookup is case-insensitive: "open_application" and
    "OPEN_APPLICATION" resolve to the same command.

    Args:
        intent: The intent name to look up.

    Returns:
        The CommandDefinition, or None if the intent is not registered.
    """
    if not isinstance(intent, str):
        return None
    return _COMMANDS.get(intent.upper())


def get_all_commands() -> dict[str, CommandDefinition]:
    """
    Return all registered command definitions.

    Returns:
        A copy of the registry mapping intent names to definitions.
    """
    return dict(_COMMANDS)


def is_valid_intent(intent: str) -> bool:
    """
    Check whether an intent name is registered.

    Args:
        intent: The intent name to validate.

    Returns:
        True if the intent exists in the registry, False otherwise.
    """
    if not isinstance(intent, str):
        return False
    return intent.upper() in _COMMANDS


def get_aliases(intent: str) -> tuple[str, ...]:
    """
    Get the aliases registered for a command.

    Args:
        intent: The intent name to look up.

    Returns:
        The tuple of aliases, or an empty tuple if the intent is unknown.
    """
    command = get_command(intent)
    return command.aliases if command is not None else ()


def get_description(intent: str) -> str | None:
    """
    Get the description registered for a command.

    Args:
        intent: The intent name to look up.

    Returns:
        The description string, or None if the intent is unknown.
    """
    command = get_command(intent)
    return command.description if command is not None else None


def find_intent_by_alias(alias: str) -> str | None:
    """
    Find the intent name for a given alias.

    Matching is exact and case-insensitive — no fuzzy matching.

    Args:
        alias: The alias to look up, e.g. "reboot".

    Returns:
        The intent name, or None if no command has this alias.
    """
    if not isinstance(alias, str):
        return None
    normalized = alias.lower()
    for command in _COMMANDS.values():
        if normalized in command.aliases:
            return command.intent
    return None


def validate_registry(
    commands: dict[str, CommandDefinition] | None = None,
) -> None:
    """
    Validate a command registry for consistency.

    Detects:
    - duplicate intents (dict key does not match the definition's intent)
    - duplicate aliases across different commands
    - empty intent names
    - empty aliases
    - invalid command definitions (wrong type)

    Args:
        commands: The registry to validate. Defaults to the module registry.

    Raises:
        ValueError: If any inconsistency is found.
    """
    cmds = commands if commands is not None else _COMMANDS

    alias_to_intent: dict[str, str] = {}

    for key, command in cmds.items():
        if not isinstance(command, CommandDefinition):
            _fail(
                f"Registry entry '{key}' is not a CommandDefinition."
            )

        # The dict key must match the definition's intent — this catches
        # duplicate/mismatched intents in a dict-based registry.
        if key != command.intent:
            _fail(
                f"Registry key '{key}' does not match intent "
                f"'{command.intent}'."
            )

        if not command.intent or not command.intent.strip():
            _fail("Command definition has an empty intent.")

        if not isinstance(command.aliases, tuple):
            _fail(
                f"Command '{command.intent}' aliases must be a tuple."
            )

        for alias in command.aliases:
            if not alias or not alias.strip():
                _fail(
                    f"Command '{command.intent}' has an empty alias."
                )
            if alias in alias_to_intent:
                other = alias_to_intent[alias]
                _fail(
                    f"Alias '{alias}' is registered for both "
                    f"'{other}' and '{command.intent}'."
                )
            alias_to_intent[alias] = command.intent


def _fail(message: str) -> None:
    """Log a registry validation error, then raise ValueError."""
    logger.error(f"Command registry validation failed: {message}")
    raise ValueError(message)


# Validate the module registry on import so a misconfigured registry
# fails loudly instead of silently misbehaving.
validate_registry()
logger.debug(f"Command registry validated ({len(_COMMANDS)} commands)")
