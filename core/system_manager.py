"""
System Manager Module

Windows operating-system actions: lock, shutdown, restart.

Receives structured calls — never natural-language commands:

    lock_computer()
    shutdown_computer(confirmed=True)
    restart_computer(confirmed=True)

Safe execution rules:
- Windows-only operations are gated by is_windows() (platform.system()).
- Locking uses the Windows user32 API via ctypes.
- Shutdown/restart use shutdown.exe with a fixed argument list.
- Arbitrary shell execution is forbidden — commands are never built
  from user input and no shell interpreter is ever enabled.
- Destructive operations require explicit confirmation (confirmed=True).
"""

import ctypes
import platform
import subprocess
from typing import Any

from core.config_manager import get_value
from utils.logger import get_logger

logger = get_logger(__name__)


def _confirmation_enabled(key: str) -> bool:
    """
    Return whether confirmation is required for a dangerous operation.

    Reads the safety configuration, but confirmation is ALWAYS
    enforced regardless of the configured value. The configuration
    defaults to True and a malformed value is treated as True, so a
    setting can never disable this protection.
    """
    configured = get_value("safety", key)
    if not isinstance(configured, bool):
        return True
    # Never allow configuration to disable confirmation.
    return True


def is_windows() -> bool:
    """
    Check whether the current operating system is Windows.

    Uses platform.system() so the answer is detected at runtime,
    never assumed from the project's target platform.

    Returns:
        bool: True when running on Windows, False otherwise.
    """
    return platform.system() == "Windows"


def lock_computer() -> dict[str, Any]:
    """
    Lock the current Windows workstation.

    Uses the Windows user32 LockWorkStation API via ctypes.
    Locking is not destructive, so no confirmation is required.

    Returns:
        dict: {success, message, data} result.
    """
    logger.info("System operation requested: lock_computer")

    if not is_windows():
        logger.warning("Lock blocked: current OS is not Windows")
        return _result(False, "This operation is only supported on Windows.")

    try:
        locked = ctypes.windll.user32.LockWorkStation()
    except Exception:
        logger.exception("Failed to lock computer")
        return _result(False, "Unable to lock the computer.")

    if not locked:
        logger.error("Failed to lock computer: LockWorkStation reported failure")
        return _result(False, "Unable to lock the computer.")

    logger.info("Computer lock initiated")
    return _result(True, "Computer locked successfully.")


def shutdown_computer(confirmed: bool = False) -> dict[str, Any]:
    """
    Shut down the computer.

    Shutdown is destructive, so it requires explicit confirmation.
    Without confirmed=True the operation is refused.

    Uses shutdown.exe with a fixed argument list — no shell, no
    arbitrary commands, no user input.

    Args:
        confirmed: Must be True before any shutdown occurs.

    Returns:
        dict: {success, message, data} result.
    """
    logger.info("System operation requested: shutdown_computer")

    if not confirmed:
        logger.warning("Shutdown requested without confirmation")
        return _result(
            False,
            "Confirmation required before shutting down the computer.",
        )

    if not is_windows():
        logger.warning("Shutdown blocked: current OS is not Windows")
        return _result(False, "This operation is only supported on Windows.")

    try:
        subprocess.run(
            ["shutdown", "/s", "/t", "0"],
            capture_output=True,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        logger.exception("Failed to shut down computer")
        return _result(False, "Unable to shut down the computer.")

    logger.info("Shutdown initiated")
    return _result(True, "Computer shutting down.")


def restart_computer(confirmed: bool = False) -> dict[str, Any]:
    """
    Restart the computer.

    Restart is disruptive, so it requires explicit confirmation.
    Without confirmed=True the operation is refused.

    Uses shutdown.exe with a fixed argument list — no shell, no
    arbitrary commands, no user input.

    Args:
        confirmed: Must be True before any restart occurs.

    Returns:
        dict: {success, message, data} result.
    """
    logger.info("System operation requested: restart_computer")

    if not confirmed:
        logger.warning("Restart requested without confirmation")
        return _result(
            False,
            "Confirmation required before restarting the computer.",
        )

    if not is_windows():
        logger.warning("Restart blocked: current OS is not Windows")
        return _result(False, "This operation is only supported on Windows.")

    try:
        subprocess.run(
            ["shutdown", "/r", "/t", "0"],
            capture_output=True,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        logger.exception("Failed to restart computer")
        return _result(False, "Unable to restart the computer.")

    logger.info("Restart initiated")
    return _result(True, "Computer restarting.")


def _result(success: bool, message: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Build the standard result structure used by every manager.

    Args:
        success: Whether the operation succeeded.
        message: Human-readable explanation.
        data: Optional extra information (defaults to an empty dict).

    Returns:
        dict: {success, message, data}
    """
    return {
        "success": success,
        "message": message,
        "data": data if data is not None else {},
    }
