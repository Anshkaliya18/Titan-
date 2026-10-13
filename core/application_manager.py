"""
Application Manager

Centralized application management responsible for:
- Application registration
- Application aliases
- Application lookup
- Executable/path resolution
- Installed checks
- Safe application launching

This module does NOT depend on natural-language input.
It receives application names/aliases only.

Architecture:
    executor
      -> application_manager
        -> application registry
          -> Windows executable
"""

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Application:
    """Application data model.

    Metadata is kept separate from execution logic.
    """

    name: str                       # canonical name, e.g. "chrome"
    aliases: list[str] = field(default_factory=list)   # e.g. ["chrome", "google chrome"]
    executable_paths: list[str] = field(default_factory=list)  # candidate locations
    executable: str = ""            # fallback executable name for PATH search
    description: str = ""           # human-readable description
    is_browser: bool = False        # True for web browsers (used by the browser manager)
    shortcut_path: str | None = None  # validated Start Menu shortcut, when discovered


def _env(name: str) -> str:
    """Get a Windows environment variable; empty string if unset."""
    return os.environ.get(name, "")


def _join(*parts: str) -> str:
    """Join path parts; returns empty string if any part is missing."""
    if not all(parts):
        return ""
    return os.path.join(*parts)


# Windows environment locations (safe — no hardcoded usernames)
PROGRAM_FILES = _env("PROGRAMFILES")
PROGRAM_FILES_X86 = _env("PROGRAMFILES(X86)")
LOCAL_APP_DATA = _env("LOCALAPPDATA")
WINDOWS_DIR = _env("WINDIR")


# ---------------------------------------------------------------------------
# Central application registry (single source of truth)
# ---------------------------------------------------------------------------

APPLICATIONS: list[Application] = [
    Application(
        name="chrome",
        aliases=["chrome", "google chrome", "google"],
        executable_paths=[
            _join(PROGRAM_FILES, "Google", "Chrome", "Application", "chrome.exe"),
            _join(PROGRAM_FILES_X86, "Google", "Chrome", "Application", "chrome.exe"),
            _join(LOCAL_APP_DATA, "Google", "Chrome", "Application", "chrome.exe"),
        ],
        executable="chrome.exe",
        description="Google Chrome web browser",
        is_browser=True,
    ),
    Application(
        name="firefox",
        aliases=["firefox", "mozilla firefox"],
        executable_paths=[
            _join(PROGRAM_FILES, "Mozilla Firefox", "firefox.exe"),
            _join(PROGRAM_FILES_X86, "Mozilla Firefox", "firefox.exe"),
        ],
        executable="firefox.exe",
        description="Mozilla Firefox web browser",
        is_browser=True,
    ),
    Application(
        name="edge",
        aliases=["edge", "microsoft edge"],
        executable_paths=[
            _join(PROGRAM_FILES_X86, "Microsoft", "Edge", "Application", "msedge.exe"),
            _join(PROGRAM_FILES, "Microsoft", "Edge", "Application", "msedge.exe"),
        ],
        executable="msedge.exe",
        description="Microsoft Edge web browser",
        is_browser=True,
    ),
    Application(
        name="notepad",
        aliases=["notepad"],
        executable_paths=[
            _join(WINDOWS_DIR, "System32", "notepad.exe"),
        ],
        executable="notepad.exe",
        description="Windows Notepad text editor",
    ),
    Application(
        name="calculator",
        aliases=["calculator", "calc"],
        executable_paths=[
            _join(WINDOWS_DIR, "System32", "calc.exe"),
        ],
        executable="calc.exe",
        description="Windows Calculator",
    ),
    Application(
        name="explorer",
        aliases=["explorer", "file explorer"],
        executable_paths=[
            _join(WINDOWS_DIR, "explorer.exe"),
        ],
        executable="explorer.exe",
        description="Windows File Explorer",
    ),
    Application(
        name="cmd",
        aliases=["cmd", "command prompt", "commandprompt"],
        executable_paths=[
            _join(WINDOWS_DIR, "System32", "cmd.exe"),
        ],
        executable="cmd.exe",
        description="Windows Command Prompt",
    ),
]

# Flat alias -> Application mapping (derived from the registry)
ALIAS_TO_APP: dict[str, Application] = {
    alias: app for app in APPLICATIONS for alias in app.aliases
}

# Set of all aliases for fast lookup
APP_NAMES: set[str] = set(ALIAS_TO_APP.keys())

# Successful executable resolutions, keyed by canonical application
# name (Step 22). Only *positive* results are cached: a failed
# resolution is retried on the next command, so an application
# installed while TITAN is running is discovered automatically.
# Values come only from the static allowlisted executable_paths or
# from shutil.which() on the same allowlisted executable name — user
# input never reaches this cache, and the allowlist lookup in
# get_application() happens before any resolution does.
_RESOLVED_PATHS: dict[str, str] = {}


def invalidate_application_cache(name: str | None = None) -> None:
    """
    Forget cached executable resolutions.

    Provided as a safe invalidation hook: normally nothing needs to
    call it (only positive resolutions are cached and they re-check
    cheaply on failure), but tests or long-running sessions where an
    application was moved/uninstalled can clear state explicitly.

    Args:
        name: Application name/alias to invalidate, or None for all.
    """
    if name is None:
        _RESOLVED_PATHS.clear()
        return
    app = ALIAS_TO_APP.get(_normalize(name))
    _RESOLVED_PATHS.pop(app.name if app else _normalize(name), None)


def _normalize(name: str) -> str:
    """Normalize an application name: lowercase, collapse whitespace."""
    return " ".join(name.strip().lower().split())


def is_application(name: str) -> bool:
    """Check if a name is a known application alias."""
    return get_application(name) is not None


def get_application(name: str) -> Application | None:
    """
    Look up an application by alias.

    Lookup is case-insensitive, whitespace-tolerant, and deterministic.
    No fuzzy matching is performed — spell correction runs earlier
    in the pipeline.

    Args:
        name: Application alias (e.g. "chrome", "google chrome").

    Returns:
        The Application, or None if the alias is unknown.
    """
    logger.info("Application lookup requested")
    static = ALIAS_TO_APP.get(_normalize(name))
    if static is not None:
        return static
    # Discovery supplements the stable static registry. A path or an
    # ambiguous name is never converted to an Application object.
    from core.application_discovery import get_discovery
    resolution = get_discovery().resolve(name)
    if resolution.status != "MATCH" or resolution.application is None:
        return None
    discovered = resolution.application
    return Application(
        name=discovered.name,
        aliases=[discovered.name, *discovered.aliases],
        executable_paths=[discovered.executable],
        executable=os.path.basename(discovered.executable),
        description="Discovered installed application",
        shortcut_path=discovered.shortcut_path,
    )


def _resolve_executable(app: Application) -> str | None:
    """Resolve an Application to an executable path (no lookup logging).

    Successful resolutions are memoized per application name (see
    _RESOLVED_PATHS): subsequent "open chrome" commands reuse the path
    instead of re-running isfile()/which() discovery. Failures are not
    cached so newly installed applications are found on retry.
    """
    if app.description == "Discovered installed application":
        from core.application_discovery import validate_executable
        return str(validate_executable(app.executable_paths[0])) if app.executable_paths else None

    cached = _RESOLVED_PATHS.get(app.name)
    if cached is not None:
        return cached

    # Check configured installation paths in order
    for path in app.executable_paths:
        if path and os.path.isfile(path):
            _RESOLVED_PATHS[app.name] = path
            return path

    # Fallback: search the system PATH
    if app.executable:
        found = shutil.which(app.executable)
        if found:
            _RESOLVED_PATHS[app.name] = found
        return found

    return None


def resolve_application(name: str) -> str | None:
    """
    Resolve an application alias to an executable path.

    Checks configured installation paths first, then falls back
    to searching the system PATH. Does not launch anything.

    Args:
        name: Application alias (e.g. "chrome", "google chrome").

    Returns:
        The resolved executable path, or None if not found.
    """
    app = get_application(name)
    if app is None:
        return None

    executable = _resolve_executable(app)
    if executable:
        logger.info(f"Application resolved: {app.name}")
    else:
        logger.warning(f"Executable not found for {app.name}")

    return executable


def is_installed(name: str) -> bool:
    """
    Check whether an application is installed.

    Internally uses application resolution. Does not launch
    the application.

    Args:
        name: Application alias.

    Returns:
        True if the application's executable can be located.
    """
    return resolve_application(name) is not None


def launch_application(name: str) -> dict[str, Any]:
    """
    Safely launch an application from the allowlisted registry.

    Flow:
        name -> get_application -> resolve_application -> safe launch

    Only applications that exist in the allowlisted registry are
    launched. Raw user input is never passed to subprocess.

    Args:
        name: Application alias.

    Returns:
        Structured result with success, message, application, path.
    """
    if sys.platform != "win32":
        return _result(
            False, "This command requires Windows.", None, None
        )

    app = get_application(name)
    if app is None:
        # Give a useful deterministic response for near-name collisions.
        if isinstance(name, str) and name and not os.path.isabs(name):
            from core.application_discovery import get_discovery
            resolution = get_discovery().resolve(name)
            if resolution.status == "AMBIGUOUS":
                labels = ", ".join(candidate.name for candidate in resolution.candidates[:10])
                logger.info("Ambiguous application request")
                return _result(False, f"I found multiple matching applications: {labels}. Which one should I open?", None, None)
        logger.warning(f"Application not found: {name}")
        return _result(
            False,
            f"I couldn't find {str(name)[:100]} installed on this computer (not allowed or configured).",
            None,
            None,
        )

    executable = _resolve_executable(app)
    if executable is None and app.description == "Discovered installed application":
        # Discovery indexes are staleable. Rebuild once if a cached target
        # disappeared or failed validation, then retry name resolution.
        from core.application_discovery import refresh_application_index
        refresh_application_index()
        refreshed_app = get_application(name)
        if refreshed_app is not None:
            app = refreshed_app
            executable = _resolve_executable(app)
    if executable is None:
        logger.warning(f"Executable not found for {name}")
        return _result(
            False,
            f"{app.name.title()} is not installed or could not be located.",
            app.name,
            None,
        )

    logger.info(f"Application resolved: {app.name}")
    logger.info(f"Launching application: {name}")

    try:
        # Prefer the revalidated Start Menu shortcut when available. Windows
        # then applies the app's own installed shortcut settings (not user
        # supplied arguments), which is required for Store-backed and PWA
        # shortcuts. Direct executable fallback remains a structured argv.
        if app.shortcut_path:
            from core.application_discovery import validate_shortcut
            shortcut = validate_shortcut(app.shortcut_path, executable)
            if shortcut is not None:
                os.startfile(str(shortcut))
                logger.info("Launched application shortcut: %s", app.name)
                return _result(True, f"{app.name.title()} opened successfully.", app.name, executable)
        # Launch only the resolved executable — never raw user input.
        subprocess.Popen([executable], shell=False)
        logger.info(f"Launched application: {name}")
        return _result(
            True,
            f"{app.name.title()} opened successfully.",
            app.name,
            executable,
        )
    except Exception:
        # Full traceback goes to the log; the user gets a safe message
        # that never exposes internal exception details.
        logger.exception(f"Failed to launch application: {name}")
        return _result(
            False,
            f"Failed to launch {app.name}.",
            app.name,
            executable,
        )


def close_application(name: str) -> dict[str, Any]:
    """
    Close an application using its registered process image name.

    Args:
        name: Application alias.

    Returns:
        Structured result with success, message, application, path.
    """
    if sys.platform != "win32":
        return _result(
            False, "This command requires Windows.", None, None
        )

    app = get_application(name)
    if app is None:
        logger.warning(f"Application not found: {name}")
        return _result(
            False,
            f"Application '{name}' is not allowed or configured.",
            None,
            None,
        )

    image_name = app.executable
    if not image_name:
        return _result(
            False,
            f"Application '{name}' is not configured for closing.",
            app.name,
            None,
        )

    try:
        completed = subprocess.run(
            ["taskkill", "/f", "/im", image_name], capture_output=True
        )
        if completed.returncode != 0:
            # taskkill failed — most commonly the process is not
            # running. Report failure rather than a false success.
            logger.info(f"No running process to close: {name}")
            return _result(
                False,
                f"{app.name.title()} is not running.",
                app.name,
                image_name,
            )
        logger.info(f"Closed application: {name}")
        return _result(
            True,
            f"{app.name.title()} closed successfully.",
            app.name,
            image_name,
        )
    except Exception:
        logger.exception(f"Failed to close application: {name}")
        return _result(
            False,
            f"Failed to close {app.name}.",
            app.name,
            image_name,
        )


def _result(
    success: bool,
    message: str,
    application: str | None,
    path: str | None,
) -> dict[str, Any]:
    """
    Create a consistent application result structure.

    Follows the project-wide {success, message, data} convention so
    multi-command chaining and history keep the application details
    instead of silently dropping them.
    """
    return {
        "success": success,
        "message": message,
        "data": {"application": application, "path": path},
    }
