"""
Windows Startup Manager (Step 23 — Phases 13–15)

Optional, user-level registration of Jarvis in Windows startup.

Mechanism (Phase 14) — one value in the per-user "Run" registry key:

    Key   : HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Run
    Name  : Jarvis
    Data  : the resolved command that starts this Jarvis installation

Windows runs that value when the user logs on. That is the whole
mechanism: no service, no driver, no scheduled task, no system-wide
change, no administrator rights, no hidden process. The entry is
visible (startup status prints the exact key, value name and command)
and reversible with disable_startup().

Security rules (Phase 15)
-------------------------
- enable_startup() takes NO arguments. The executable path is derived
  from the running installation only (sys.executable plus, in source
  mode, this project's main.py) and is validated before anything is
  written: it must be absolute, an existing file, and an .exe/.py.
  User input can never determine the executable, the registry key,
  the value name, or any argument.
- The command string is built by quoting those fixed paths ourselves.
  No shell is involved anywhere (no os.system, no shell=True), so the
  registry data can never turn into command execution with arguments —
  it is one quoted program path in packaged mode, or a quoted
  interpreter plus a quoted script path in source mode.
- This module never runs programs and never reads user-typed input:
  it only reads or writes that single registry value. All registry
  operations are wrapped, so a failure returns {success: False} and
  never raises into the pipeline.
- Registration is idempotent: enabling twice keeps exactly one value.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from core.version import APP_NAME
from utils.logger import get_logger
from utils.paths import is_frozen

logger = get_logger(__name__)

# winreg exists on Windows only; a missing module is reported as an
# unavailable feature instead of raising (Jarvis must run anywhere).
try:
    import winreg
except Exception:  # pragma: no cover - non-Windows environments
    winreg = None  # type: ignore[assignment]

# The single startup location (documented in README.md and printed by
# the status command — never influenced by user input).
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
VALUE_NAME = APP_NAME
RUN_KEY_DISPLAY = rf"HKEY_CURRENT_USER\{RUN_KEY}"

# Executables this module is willing to register.
_ALLOWED_SUFFIXES = {".exe", ".py"}


def _result(success: bool, message: str, data: dict[str, Any]) -> dict:
    """Create the standard {success, message, data} structure."""
    result = {"success": success, "message": message, "data": data}
    log = logger.info if success else logger.warning
    log(message)
    return result


def _quote(path: Path) -> str:
    """Quote an absolute path (Windows file names cannot contain quotes)."""
    return f'"{path}"'


def _resolve_install_command() -> str:
    """
    Build the validated logon command for THIS installation.

    The path always comes from the running installation — sys.executable
    and, in source mode, main.py next to this project. Nothing here can
    be supplied by user input.

    Returns:
        str: Packaged mode -> '"C:\\...\\Jarvis.exe"'
             Source mode   -> '"C:\\...\\python.exe" "C:\\...\\main.py"'

    Raises:
        ValueError: If the installation cannot be verified (missing
            interpreter, missing entry point, or an unexpected
            program type). The caller reports a safe message.
    """
    raw_exe = Path(sys.executable)
    if not raw_exe.is_absolute():
        # sys.executable is always absolute in normal Python; a
        # relative value cannot be verified, so it is refused.
        raise ValueError("interpreter path is not absolute")
    exe = raw_exe.resolve()

    # Never register a hidden console when a console Python exists
    # alongside it: Jarvis is a CLI and must stay visible (Phase 19).
    if exe.name.lower() == "pythonw.exe":
        console_exe = exe.with_name("python.exe")
        if console_exe.is_file():
            exe = console_exe

    if not exe.is_file():
        raise ValueError("interpreter not found")
    if exe.suffix.lower() not in _ALLOWED_SUFFIXES:
        raise ValueError(f"unexpected program type: {exe.suffix}")

    if is_frozen():
        # Packaged mode: the executable itself, no arguments.
        return _quote(exe)

    # Source mode: the interpreter runs this project's entry point.
    entry_point = Path(__file__).resolve().parent.parent / "main.py"
    if not entry_point.is_file():
        raise ValueError("Jarvis entry point not found")
    return f"{_quote(exe)} {_quote(entry_point)}"


def _target_path(command: str) -> Path | None:
    """
    Extract the program path from a stored command (read-only check).

    Only the first quoted (or first whitespace-delimited) token is
    taken — the same layout this module writes. Used by the status
    command to detect a stale entry; never used to run anything.
    """
    text = (command or "").strip()
    if not text:
        return None
    if text.startswith('"'):
        end = text.find('"', 1)
        if end <= 1:
            return None
        return Path(text[1:end])
    return Path(text.split()[0])


# ---------------------------------------------------------------------------
# Registry leaves — the only functions that touch the registry. Kept small
# and private so tests can replace them without ever writing to the real
# user's startup entry.
# ---------------------------------------------------------------------------

def _registry_read() -> tuple[bool, str | None]:
    """
    Read the Jarvis Run value.

    Returns:
        tuple[bool, str | None]: (registry available, stored command
        or None when the value is absent). A missing Run key reads as
        "not registered"; a genuine read error is logged and treated
        the same way (reporting "disabled" is the safe direction).
    """
    if winreg is None:
        return False, None
    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_READ
        ) as key:
            try:
                value, _reg_type = winreg.QueryValueEx(key, VALUE_NAME)
            except FileNotFoundError:
                return True, None
        return True, value if isinstance(value, str) else str(value)
    except FileNotFoundError:
        return True, None
    except OSError as exc:
        logger.warning("Could not read the Windows startup entry: %s", exc)
        return True, None


def _registry_write(command: str) -> None:
    """Write (create or replace) the single Jarvis Run value."""
    if winreg is None:
        raise OSError("the Windows registry is unavailable")
    with winreg.CreateKeyEx(
        winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE
    ) as key:
        winreg.SetValueEx(key, VALUE_NAME, 0, winreg.REG_SZ, command)


def _registry_delete() -> bool:
    """
    Delete the Jarvis Run value.

    Returns:
        bool: True when a value was removed, False when there was
        nothing to remove (deleting twice is always safe).
    """
    if winreg is None:
        raise OSError("the Windows registry is unavailable")
    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE
        ) as key:
            winreg.DeleteValue(key, VALUE_NAME)
        return True
    except FileNotFoundError:
        return False


# ---------------------------------------------------------------------------
# Public API (Phases 13–15)
# ---------------------------------------------------------------------------

def enable_startup() -> dict:
    """
    Register this Jarvis installation to start when Windows starts.

    Takes no arguments: the command comes from the validated
    installation location only. Writing the same value twice keeps
    exactly one startup entry (Windows stores one value per name).

    Returns:
        dict: {success, message, data} — data carries the registry
        location, value name, and the exact command registered.
    """
    if winreg is None:
        return _result(
            False,
            "Windows startup is only supported on Windows.",
            {},
        )

    try:
        command = _resolve_install_command()
    except ValueError as exc:
        logger.warning("Startup registration refused: %s", exc)
        return _result(
            False,
            "I couldn't verify this Jarvis installation, so startup "
            "was not changed.",
            {},
        )

    available, existing = _registry_read()
    if not available:
        return _result(
            False,
            "Windows startup is only supported on Windows.",
            {},
        )

    try:
        _registry_write(command)
    except OSError as exc:
        logger.error("Could not write the Windows startup entry: %s", exc)
        return _result(
            False,
            "I couldn't update the Windows startup entry.",
            {},
        )

    data = {
        "enabled": True,
        "location": RUN_KEY_DISPLAY,
        "value": VALUE_NAME,
        "command": command,
    }
    if existing == command:
        return _result(True, f"{APP_NAME} already starts with Windows.", data)
    return _result(True, f"{APP_NAME} will start with Windows.", data)


def disable_startup() -> dict:
    """
    Remove the Windows startup registration.

    Only the Jarvis Run value is removed — user configuration, logs,
    and every other startup entry are left untouched. Disabling when
    nothing is registered is a safe no-op.

    Returns:
        dict: {success, message, data}
    """
    if winreg is None:
        return _result(
            False,
            "Windows startup is only supported on Windows.",
            {},
        )

    try:
        removed = _registry_delete()
    except OSError as exc:
        logger.error("Could not remove the Windows startup entry: %s", exc)
        return _result(
            False,
            "I couldn't update the Windows startup entry.",
            {},
        )

    data = {"enabled": False, "location": RUN_KEY_DISPLAY, "value": VALUE_NAME}
    if removed:
        return _result(
            True, f"{APP_NAME} will no longer start with Windows.", data
        )
    return _result(True, f"{APP_NAME} does not start with Windows.", data)


def is_startup_enabled() -> bool:
    """
    Report whether a Jarvis startup entry exists.

    Returns:
        bool: True when the Run value is present (even if the program
        it points at has since been removed), False otherwise or when
        the registry is unavailable. Reading never changes anything.
    """
    try:
        available, command = _registry_read()
    except OSError as exc:
        # A read failure must never raise out of a status query —
        # reporting "disabled" is the safe direction.
        logger.warning("Could not read the Windows startup entry: %s", exc)
        return False
    if not available:
        return False
    # An empty value counts as "not registered" — a malformed entry
    # must never be reported as an active startup entry.
    return bool(command)


def startup_status() -> dict:
    """
    Describe the current startup registration (read-only).

    Returns:
        dict: {success, message, data} with data.enabled, the registry
        location, the stored command, and whether the target program
        still exists (a stale entry is reported, never repaired
        implicitly).
    """
    try:
        available, command = _registry_read()
    except OSError as exc:
        logger.warning("Could not read the Windows startup entry: %s", exc)
        return _result(
            False,
            "Windows startup is only supported on Windows.",
            {},
        )
    if not available:
        return _result(
            False,
            "Windows startup is only supported on Windows.",
            {},
        )

    # Tolerate a wrong-typed value: report it, never crash on it.
    if command is not None and not isinstance(command, str):
        command = str(command)
    enabled = bool(command)
    data: dict[str, Any] = {
        "enabled": enabled,
        "location": RUN_KEY_DISPLAY,
        "value": VALUE_NAME,
    }
    if command:
        data["command"] = command
        target = _target_path(command)
        target_exists = bool(target is not None and target.is_file())
        data["target_exists"] = target_exists
        if target_exists:
            message = f"{APP_NAME} starts with Windows."
        else:
            message = (
                f"{APP_NAME} starts with Windows, but the registered "
                f"program no longer exists."
            )
    else:
        message = f"{APP_NAME} does not start with Windows."
    return _result(True, message, data)
