"""
Executor Module

Receives a structured command from the command parser
and performs the corresponding safe Windows action.

This module does NOT parse, tokenize, or detect intent.
It only executes structured commands.
"""

from typing import Any

from core.application_manager import close_application, launch_application
from core.startup_manager import (  # noqa: E402
    disable_startup,
    enable_startup,
    startup_status,
)
from core.system_manager import lock_computer, shutdown_computer, restart_computer
from core.file_manager import create_folder, delete_file, open_file, search_file
from core.browser_manager import open_browser, open_url, search_web
from core.command_history import get_command_history
from core.command_validator import validate_command
from core.exceptions import JarvisError
from core.help_manager import get_command_list, get_help
from utils.logger import get_logger

logger = get_logger(__name__)


def execute_command(command: dict, confirmed: bool | None = None) -> dict:
    """
    Execute a structured command.

    Args:
        command (dict): Structured command with intent, target, and arguments.
        confirmed: Optional pre-obtained confirmation answer for
            dangerous commands (shutdown/restart/delete). When None —
            the default — the executor asks inline via input(). The
            interactive CLI passes confirmed=True/False here after the
            session-level confirmation round-trip, so the dangerous
            operation is never prompted for twice.

    Returns:
        dict: Execution result with success, message, and data.
    """
    # Defensively validate the command before routing — a command may
    # arrive here from a module other than main.py. Validation runs
    # first so that malformed commands (including non-dict values)
    # never reach a manager.
    validation = validate_command(command)
    if not validation["valid"]:
        result = _result(
            False,
            "Invalid command.",
            {"validation_errors": validation["errors"]},
        )
        logger.warning(f"Invalid command: {validation['message']}")
        return result

    intent = command.get("intent")
    target = command.get("target")

    logger.info(f"Executing command: {intent}")

    # Route to the appropriate handler. Intent is normalized to
    # lowercase to match the canonical Intent enum values, so the
    # executor is robust to case variations.
    handlers = {
        "open_application": _handle_open_application,
        "close_application": _handle_close_application,
        "find_application": _handle_find_application,
        "refresh_applications": _handle_refresh_applications,
        "open_browser": _handle_open_browser,
        "open_url": _handle_open_url,
        "search_web": _handle_search_web,
        "lock_computer": _handle_lock_computer,
        "shutdown_computer": (
            lambda t, a: _handle_shutdown_computer(t, a, confirmed)
        ),
        "restart_computer": (
            lambda t, a: _handle_restart_computer(t, a, confirmed)
        ),
        "enable_startup": (
            lambda t, a: _handle_enable_startup(t, a, confirmed)
        ),
        "disable_startup": (
            lambda t, a: _handle_disable_startup(t, a, confirmed)
        ),
        "startup_status": _handle_startup_status,
        "create_folder": _handle_create_folder,
        "delete_file": (
            lambda t, a: _handle_delete_file(t, a, confirmed)
        ),
        "open_file": _handle_open_file,
        "search_file": _handle_search_file,
        "help": _handle_help,
        "list_commands": _handle_list_commands,
        "history": _handle_history,
        "clear_history": _handle_clear_history,
    }

    handler = handlers.get(intent.lower() if isinstance(intent, str) else intent)

    if handler is None:
        result = _result(False, "Unknown command.", {})
        logger.warning(f"Unknown command: {intent}")
        return result

    # Error boundary: a manager must never be able to crash the
    # assistant. Expected Jarvis errors are logged without a traceback;
    # unexpected exceptions are logged with the full traceback and both
    # produce a safe, generic user-facing result.
    try:
        return handler(target, command.get("arguments", {}))
    except JarvisError as e:
        logger.error("Command failed: %s", e)
        return _result(False, "I couldn't complete that command.", {})
    except Exception:
        logger.exception("Unexpected error while executing: %s", intent)
        return _result(
            False,
            "Something went wrong while processing that command.",
            {},
        )


def _handle_open_application(target: str | None, arguments: dict) -> dict:
    """Handle OPEN_APPLICATION intent."""
    if not target:
        return _result(False, "Which application should I open?", {})
    return launch_application(target)


def _handle_close_application(target: str | None, arguments: dict) -> dict:
    """Handle CLOSE_APPLICATION intent."""
    if not target:
        return _result(False, "Which application should I close?", {})
    return close_application(target)


def _handle_find_application(target: str | None, arguments: dict) -> dict:
    """Read-only, bounded search of the cached installed-app index."""
    from core.application_discovery import get_discovery
    discovery = get_discovery()
    if target:
        resolution = discovery.resolve(target)
        if resolution.status == "MATCH" and resolution.application:
            return _result(True, f"Found {resolution.application.name}.", {"applications": [resolution.application.name]})
        if resolution.status == "AMBIGUOUS":
            names = [a.name for a in resolution.candidates[:discovery.max_results]]
            return _result(True, "Matching applications: " + ", ".join(names), {"applications": names})
        matches = discovery.find_candidates(target)
    else:
        matches = discovery.get_all()[:discovery.max_results]
    names = [a.name for a in matches[:discovery.max_results]]
    if not names:
        return _result(True, "No matching installed applications were found.", {"applications": []})
    return _result(True, "Installed applications: " + ", ".join(names), {"applications": names})


def _handle_refresh_applications(target: str | None, arguments: dict) -> dict:
    """Explicitly rebuild the bounded, read-only app index."""
    from core.application_discovery import refresh_application_index
    apps = refresh_application_index()
    return _result(True, f"Application list refreshed ({len(apps)} applications indexed).", {"count": len(apps)})


def _handle_open_browser(target: str | None, arguments: dict) -> dict:
    """Handle OPEN_BROWSER intent."""
    return open_browser()


def _handle_help(target: str | None, arguments: dict) -> dict:
    """Handle HELP intent. Routes to the Help Manager."""
    return get_help()


def _handle_list_commands(target: str | None, arguments: dict) -> dict:
    """Handle LIST_COMMANDS intent. Routes to the Help Manager."""
    return get_command_list()


def _handle_history(target: str | None, arguments: dict) -> dict:
    """Handle HISTORY intent. Routes to Command History (read-only).

    Reading history never replays, re-executes, or modifies any
    stored command — it only builds the structured result.
    """
    return get_command_history().get_history_result()


def _handle_clear_history(target: str | None, arguments: dict) -> dict:
    """Handle CLEAR_HISTORY intent. Clears the in-memory session
    history only — no files, databases, or disk state are touched.
    """
    get_command_history().clear_history()
    return _result(True, "Command history cleared.", {})


def _handle_open_url(target: str | None, arguments: dict) -> dict:
    """Handle OPEN_URL intent."""
    if not target:
        return _result(False, "Which URL should I open?", {})
    return open_url(target)


def _handle_search_web(target: str | None, arguments: dict) -> dict:
    """Handle SEARCH_WEB intent."""
    if not target:
        return _result(False, "What should I search for?", {})
    return search_web(target)


def _handle_lock_computer(target: str | None, arguments: dict) -> dict:
    """Handle LOCK_COMPUTER intent. Locking needs no confirmation."""
    return lock_computer()


# Intents that require an explicit user confirmation before anything
# executes. Shared by the interactive CLI (main.py) and the
# multi-command executor so every entry point uses the exact same
# safety policy. Keys are canonical lowercase intent values.
#
# enable_startup / disable_startup change the user's Windows
# configuration (Step 23, Phase 17), so they use the same session
# confirmation round-trip as shutdown/restart/delete — there is no
# second confirmation system. startup_status only reads and runs
# immediately.
CONFIRMATION_INTENTS = frozenset({
    "shutdown_computer",
    "restart_computer",
    "delete_file",
    "enable_startup",
    "disable_startup",
})


def confirmation_prompt(command: dict) -> str:
    """Return the confirmation prompt for a dangerous command.

    Single source of truth for confirmation wording — both the inline
    prompt below and the session-level round-trip in main.py use it,
    so the user always sees the same text for the same command.

    Args:
        command: Structured command awaiting confirmation.

    Returns:
        The prompt string to display to the user.
    """
    intent = str(command.get("intent", "")).lower()
    if intent == "shutdown_computer":
        return "Are you sure you want to shut down the computer? (yes/no): "
    if intent == "restart_computer":
        return "Are you sure you want to restart the computer? (yes/no): "
    if intent == "delete_file":
        target = command.get("target")
        return f"Are you sure you want to delete '{target}'? [yes/no]: "
    if intent == "enable_startup":
        return "Are you sure you want Jarvis to start with Windows? (yes/no): "
    if intent == "disable_startup":
        return (
            "Are you sure you want to stop Jarvis from starting with "
            "Windows? (yes/no): "
        )
    return "Are you sure? (yes/no): "


def _handle_shutdown_computer(
    target: str | None, arguments: dict, confirmed: bool | None = None
) -> dict:
    """Handle SHUTDOWN_COMPUTER intent.

    Asks for explicit confirmation and passes the answer to the
    system manager, which performs the single confirmation gate.
    """
    if confirmed is None:
        confirmed = _confirm(confirmation_prompt({"intent": "shutdown_computer"}))
    return shutdown_computer(confirmed=confirmed)


def _handle_restart_computer(
    target: str | None, arguments: dict, confirmed: bool | None = None
) -> dict:
    """Handle RESTART_COMPUTER intent.

    Asks for explicit confirmation and passes the answer to the
    system manager, which performs the single confirmation gate.
    """
    if confirmed is None:
        confirmed = _confirm(confirmation_prompt({"intent": "restart_computer"}))
    return restart_computer(confirmed=confirmed)


def _handle_enable_startup(
    target: str | None, arguments: dict, confirmed: bool | None = None
) -> dict:
    """Handle ENABLE_STARTUP intent.

    Changing the user's Windows configuration needs an explicit
    confirmation (the interactive CLI passes it after the session
    round-trip; a direct call with confirmed=None asks inline and
    fails safe when no answer is available). The target is always
    ignored — the Startup Manager registers only its own install.
    """
    if confirmed is None:
        confirmed = _confirm(confirmation_prompt({"intent": "enable_startup"}))
    if not confirmed:
        return _result(False, "Windows startup change cancelled.", {})
    return enable_startup()


def _handle_disable_startup(
    target: str | None, arguments: dict, confirmed: bool | None = None
) -> dict:
    """Handle DISABLE_STARTUP intent. Same confirmation gate as
    enable_startup; the target is always ignored."""
    if confirmed is None:
        confirmed = _confirm(confirmation_prompt({"intent": "disable_startup"}))
    if not confirmed:
        return _result(False, "Windows startup change cancelled.", {})
    return disable_startup()


def _handle_startup_status(target: str | None, arguments: dict) -> dict:
    """Handle STARTUP_STATUS intent. Read-only — never asks for
    confirmation and never changes the registry."""
    return startup_status()


def _handle_create_folder(target: str | None, arguments: dict) -> dict:
    """Handle CREATE_FOLDER intent."""
    if not target:
        return _result(False, "What folder should I create?", {})
    return create_folder(target)


def _handle_delete_file(
    target: str | None, arguments: dict, confirmed: bool | None = None
) -> dict:
    """Handle DELETE_FILE intent with confirmation."""
    if not target:
        return _result(False, "Which file should I delete?", {})
    if confirmed is None:
        confirmed = _confirm(
            confirmation_prompt({"intent": "delete_file", "target": target})
        )
    if not confirmed:
        return _result(False, "Deletion cancelled.", {})
    return delete_file(target, confirmed=True)


def _handle_open_file(target: str | None, arguments: dict) -> dict:
    """Handle OPEN_FILE intent."""
    if not target:
        return _result(False, "Which file should I open?", {})
    return open_file(target)


def _handle_search_file(target: str | None, arguments: dict) -> dict:
    """Handle SEARCH_FILE intent."""
    if not target:
        return _result(False, "What file should I search for?", {})
    return search_file(target)


def _confirm(prompt: str) -> bool:
    """Ask the user for confirmation."""
    try:
        answer = input(prompt).strip().lower()
        return answer in {"yes", "y", "confirm"}
    except (EOFError, KeyboardInterrupt, OSError):
        # No interactive terminal available (or the user interrupted)
        # — fail safe: never treat an unread answer as approval.
        return False


def _result(success: bool, message: str, data: dict[str, Any]) -> dict:
    """Create a consistent result structure."""
    result = {
        "success": success,
        "message": message,
        "data": data,
    }
    if success:
        logger.info(f"Command executed successfully: {message}")
    else:
        logger.warning(f"Command execution failed: {message}")
    return result
