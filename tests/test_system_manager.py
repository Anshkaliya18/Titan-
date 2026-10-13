"""
Tests for the Windows system manager.

Every Windows API and system command is mocked — these tests never
lock, shut down, or restart the machine.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add parent directory to path so we can import core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import core.system_manager as system_manager
from core.system_manager import (
    is_windows,
    lock_computer,
    restart_computer,
    shutdown_computer,
)


class TestWindowsDetection(unittest.TestCase):
    """Windows detection via platform.system()."""

    @patch("core.system_manager.platform.system", return_value="Windows")
    def test_is_windows_true(self, mock_system):
        """platform.system() == 'Windows' should report True."""
        self.assertTrue(is_windows())
        mock_system.assert_called_once()

    @patch("core.system_manager.platform.system", return_value="Linux")
    def test_is_windows_false(self, mock_system):
        """platform.system() == 'Linux' should report False."""
        self.assertFalse(is_windows())
        mock_system.assert_called_once()


class TestLockComputer(unittest.TestCase):
    """Lock operations — the Windows API is always mocked."""

    @patch("core.system_manager.platform.system", return_value="Windows")
    @patch("ctypes.windll.user32.LockWorkStation", return_value=1)
    def test_lock_calls_windows_api(self, mock_lock, mock_platform):
        """lock_computer() should call the user32 LockWorkStation API."""
        result = lock_computer()
        self.assertTrue(result["success"])
        self.assertEqual(result["message"], "Computer locked successfully.")
        self.assertEqual(result["data"], {})
        mock_lock.assert_called_once()

    @patch("core.system_manager.platform.system", return_value="Windows")
    @patch("ctypes.windll.user32.LockWorkStation", return_value=0)
    def test_lock_api_failure_returns_failure(self, mock_lock, mock_platform):
        """A FALSE return from the API should be reported as failure."""
        result = lock_computer()
        self.assertFalse(result["success"])
        self.assertEqual(result["message"], "Unable to lock the computer.")

    @patch("core.system_manager.platform.system", return_value="Windows")
    @patch("ctypes.windll.user32.LockWorkStation", side_effect=OSError("boom"))
    def test_lock_api_exception_returns_failure(self, mock_lock, mock_platform):
        """An exception from the API should be a clean failure, not a crash."""
        result = lock_computer()
        self.assertFalse(result["success"])
        self.assertEqual(result["message"], "Unable to lock the computer.")

    @patch("core.system_manager.platform.system", return_value="Linux")
    @patch("ctypes.windll.user32.LockWorkStation")
    def test_lock_non_windows_fails_cleanly(self, mock_lock, mock_platform):
        """On a non-Windows OS, locking must fail cleanly without calling the API."""
        result = lock_computer()
        self.assertFalse(result["success"])
        self.assertEqual(
            result["message"], "This operation is only supported on Windows."
        )
        self.assertEqual(result["data"], {})
        mock_lock.assert_not_called()


class TestShutdownComputer(unittest.TestCase):
    """Shutdown operations — shutdown.exe is always mocked."""

    @patch("core.system_manager.platform.system", return_value="Windows")
    @patch("core.system_manager.subprocess.run")
    def test_shutdown_requires_confirmation(self, mock_run, mock_platform):
        """confirmed=False must refuse and never run shutdown.exe."""
        result = shutdown_computer(confirmed=False)
        self.assertFalse(result["success"])
        self.assertEqual(
            result["message"],
            "Confirmation required before shutting down the computer.",
        )
        self.assertEqual(result["data"], {})
        mock_run.assert_not_called()

    def test_shutdown_default_is_unconfirmed(self):
        """Calling with no arguments must also refuse."""
        result = shutdown_computer()
        self.assertFalse(result["success"])
        self.assertIn("confirmation required", result["message"].lower())

    @patch("core.system_manager.platform.system", return_value="Windows")
    @patch("core.system_manager.subprocess.run")
    def test_shutdown_confirmed_calls_shutdown_command(
        self, mock_run, mock_platform
    ):
        """confirmed=True should run shutdown.exe with the fixed argument list."""
        mock_run.return_value = MagicMock(returncode=0)
        result = shutdown_computer(confirmed=True)
        self.assertTrue(result["success"])
        self.assertEqual(result["message"], "Computer shutting down.")
        mock_run.assert_called_once_with(
            ["shutdown", "/s", "/t", "0"], capture_output=True, check=True
        )

    @patch("core.system_manager.platform.system", return_value="Linux")
    @patch("core.system_manager.subprocess.run")
    def test_shutdown_non_windows_fails_cleanly(self, mock_run, mock_platform):
        """On a non-Windows OS, shutdown must fail cleanly without running anything."""
        result = shutdown_computer(confirmed=True)
        self.assertFalse(result["success"])
        self.assertEqual(
            result["message"], "This operation is only supported on Windows."
        )
        mock_run.assert_not_called()

    @patch("core.system_manager.platform.system", return_value="Windows")
    @patch("core.system_manager.subprocess.run", side_effect=OSError("missing"))
    def test_shutdown_failure_returns_failure(self, mock_run, mock_platform):
        """A failed shutdown command should be a clean failure, not a crash."""
        result = shutdown_computer(confirmed=True)
        self.assertFalse(result["success"])
        self.assertIn("unable", result["message"].lower())

    @patch("core.system_manager.platform.system", return_value="Linux")
    @patch("core.system_manager.subprocess.run")
    def test_unconfirmed_never_runs_on_any_platform(
        self, mock_run, mock_platform
    ):
        """The confirmation gate applies even on non-Windows systems."""
        result = shutdown_computer(confirmed=False)
        self.assertFalse(result["success"])
        self.assertIn("confirmation required", result["message"].lower())
        mock_run.assert_not_called()


class TestRestartComputer(unittest.TestCase):
    """Restart operations — shutdown.exe is always mocked."""

    @patch("core.system_manager.platform.system", return_value="Windows")
    @patch("core.system_manager.subprocess.run")
    def test_restart_requires_confirmation(self, mock_run, mock_platform):
        """confirmed=False must refuse and never run shutdown.exe."""
        result = restart_computer(confirmed=False)
        self.assertFalse(result["success"])
        self.assertEqual(
            result["message"],
            "Confirmation required before restarting the computer.",
        )
        self.assertEqual(result["data"], {})
        mock_run.assert_not_called()

    def test_restart_default_is_unconfirmed(self):
        """Calling with no arguments must also refuse."""
        result = restart_computer()
        self.assertFalse(result["success"])
        self.assertIn("confirmation required", result["message"].lower())

    @patch("core.system_manager.platform.system", return_value="Windows")
    @patch("core.system_manager.subprocess.run")
    def test_restart_confirmed_calls_restart_command(
        self, mock_run, mock_platform
    ):
        """confirmed=True should run shutdown.exe with the restart flag."""
        mock_run.return_value = MagicMock(returncode=0)
        result = restart_computer(confirmed=True)
        self.assertTrue(result["success"])
        self.assertEqual(result["message"], "Computer restarting.")
        mock_run.assert_called_once_with(
            ["shutdown", "/r", "/t", "0"], capture_output=True, check=True
        )

    @patch("core.system_manager.platform.system", return_value="Linux")
    @patch("core.system_manager.subprocess.run")
    def test_restart_non_windows_fails_cleanly(self, mock_run, mock_platform):
        """On a non-Windows OS, restart must fail cleanly without running anything."""
        result = restart_computer(confirmed=True)
        self.assertFalse(result["success"])
        self.assertEqual(
            result["message"], "This operation is only supported on Windows."
        )
        mock_run.assert_not_called()

    @patch("core.system_manager.platform.system", return_value="Windows")
    @patch("core.system_manager.subprocess.run", side_effect=OSError("missing"))
    def test_restart_failure_returns_failure(self, mock_run, mock_platform):
        """A failed restart command should be a clean failure, not a crash."""
        result = restart_computer(confirmed=True)
        self.assertFalse(result["success"])
        self.assertIn("unable", result["message"].lower())


class TestResultStructure(unittest.TestCase):
    """Every operation returns the standard {success, message, data} shape."""

    def test_success_result_shape(self):
        with patch("core.system_manager.platform.system", return_value="Windows"):
            with patch(
                "ctypes.windll.user32.LockWorkStation", return_value=1
            ):
                result = lock_computer()
        self.assertEqual(set(result.keys()), {"success", "message", "data"})
        self.assertIsInstance(result["success"], bool)
        self.assertIsInstance(result["message"], str)
        self.assertIsInstance(result["data"], dict)


class TestSafeExecution(unittest.TestCase):
    """Static safety checks on the system manager source."""

    def test_no_arbitrary_shell_execution(self):
        """The module must never use os.system() or shell=True."""
        with open(system_manager.__file__, encoding="utf-8") as f:
            source = f.read()
        self.assertNotIn("os.system", source)
        self.assertNotIn("shell=True", source)
        self.assertNotIn("shell = True", source)


if __name__ == "__main__":
    unittest.main()
