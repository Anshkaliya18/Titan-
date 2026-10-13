"""
Tests for the executor module.

These tests use mocks to avoid performing dangerous real operations.

The executor routes application commands to the application manager,
so application tests mock at the executor level. Full-flow tests
mock the application manager internals.
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add parent directory to path so we can import core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.executor import execute_command
from core.application_manager import Application


class TestExecutorApplicationRouting(unittest.TestCase):
    """Test that the executor routes application commands to the manager."""

    @patch("core.executor.launch_application")
    def test_open_application_routes_to_manager(self, mock_launch):
        """OPEN_APPLICATION should call application_manager.launch_application."""
        mock_launch.return_value = {
            "success": True,
            "message": "Chrome opened successfully.",
            "application": "chrome",
            "path": r"C:\fake\chrome.exe",
        }
        result = execute_command({
            "intent": "open_application",
            "target": "chrome",
            "arguments": {},
        })
        self.assertTrue(result["success"])
        mock_launch.assert_called_once_with("chrome")

    @patch("core.executor.launch_application")
    def test_open_application_passes_target(self, mock_launch):
        """The requested target should be passed to the application manager."""
        mock_launch.return_value = {
            "success": True,
            "message": "",
            "application": None,
            "path": None,
        }
        execute_command({
            "intent": "open_application",
            "target": "google chrome",
            "arguments": {},
        })
        mock_launch.assert_called_once_with("google chrome")

    @patch("core.executor.launch_application")
    def test_open_application_missing_target(self, mock_launch):
        """Missing target should not call the application manager."""
        result = execute_command({
            "intent": "open_application",
            "target": None,
            "arguments": {},
        })
        self.assertFalse(result["success"])
        self.assertIn("requires a target", result["data"]["validation_errors"][0])
        mock_launch.assert_not_called()

    @patch("core.executor.close_application")
    def test_close_application_routes_to_manager(self, mock_close):
        """CLOSE_APPLICATION should call application_manager.close_application."""
        mock_close.return_value = {
            "success": True,
            "message": "Chrome closed successfully.",
            "application": "chrome",
            "path": "chrome.exe",
        }
        result = execute_command({
            "intent": "close_application",
            "target": "chrome",
            "arguments": {},
        })
        self.assertTrue(result["success"])
        mock_close.assert_called_once_with("chrome")

    @patch("core.executor.close_application")
    def test_close_application_missing_target(self, mock_close):
        """Missing target should not call the application manager."""
        result = execute_command({
            "intent": "close_application",
            "target": None,
            "arguments": {},
        })
        self.assertFalse(result["success"])
        mock_close.assert_not_called()

    @patch("core.application_manager.subprocess.Popen")
    @patch("core.application_manager._resolve_executable")
    @patch("core.application_manager.get_application")
    def test_open_application_full_flow(
        self, mock_get, mock_resolve, mock_popen
    ):
        """Full flow: executor -> application manager -> safe launch."""
        app = Application(
            name="chrome",
            aliases=["chrome", "google chrome"],
            executable_paths=[],
            executable="chrome.exe",
        )
        mock_get.return_value = app
        mock_resolve.return_value = r"C:\fake\chrome.exe"
        mock_popen.return_value = MagicMock()

        result = execute_command({
            "intent": "open_application",
            "target": "chrome",
            "arguments": {},
        })

        self.assertTrue(result["success"])
        self.assertEqual(result["data"]["application"], "chrome")
        self.assertEqual(result["data"]["path"], r"C:\fake\chrome.exe")
        mock_popen.assert_called_once_with(
            [r"C:\fake\chrome.exe"], shell=False
        )


class TestExecutor(unittest.TestCase):
    """Test cases for non-application executor commands."""

    def test_unknown_intent(self):
        """Test unknown intent returns failure."""
        result = execute_command({
            "intent": "unknown",
            "target": None,
            "arguments": {},
        })
        self.assertFalse(result["success"])
        self.assertIn("unknown", result["data"]["validation_errors"][0].lower())

    @patch("core.system_manager.platform.system", return_value="Windows")
    @patch("ctypes.windll.user32.LockWorkStation", return_value=1)
    def test_lock_computer(self, mock_lock, mock_platform):
        """Test locking the computer through the full flow (mocked API)."""
        result = execute_command({
            "intent": "lock_computer",
            "target": None,
            "arguments": {},
        })
        self.assertTrue(result["success"])
        mock_lock.assert_called_once()

    @patch("core.executor.shutdown_computer")
    def test_shutdown_declined_passes_confirmed_false(self, mock_shutdown):
        """Declining confirmation must pass confirmed=False to the manager."""
        mock_shutdown.return_value = {
            "success": False,
            "message": "Confirmation required before shutting down the computer.",
            "data": {},
        }
        with patch("builtins.input", return_value="no"):
            result = execute_command({
                "intent": "shutdown_computer",
                "target": None,
                "arguments": {},
            })
        self.assertFalse(result["success"])
        mock_shutdown.assert_called_once_with(confirmed=False)

    @patch("core.executor.shutdown_computer")
    def test_shutdown_confirmed_passes_confirmed_true(self, mock_shutdown):
        """Confirming must pass confirmed=True to the manager."""
        mock_shutdown.return_value = {
            "success": True,
            "message": "Computer shutting down.",
            "data": {},
        }
        with patch("builtins.input", return_value="yes"):
            result = execute_command({
                "intent": "shutdown_computer",
                "target": None,
                "arguments": {},
            })
        self.assertTrue(result["success"])
        mock_shutdown.assert_called_once_with(confirmed=True)

    @patch("core.executor.shutdown_computer")
    def test_shutdown_vague_answers_do_not_confirm(self, mock_shutdown):
        """Only yes/y/confirm confirm — vague answers pass confirmed=False."""
        mock_shutdown.return_value = {
            "success": False,
            "message": "Confirmation required before shutting down the computer.",
            "data": {},
        }
        for answer in ("sure", "okay", "do it", "go ahead", "maybe"):
            with patch("builtins.input", return_value=answer):
                execute_command({
                    "intent": "shutdown_computer",
                    "target": None,
                    "arguments": {},
                })
        self.assertEqual(mock_shutdown.call_count, 5)
        for call in mock_shutdown.call_args_list:
            self.assertEqual(call.kwargs, {"confirmed": False})

    @patch("builtins.input", return_value="yes")
    @patch("core.system_manager.subprocess.run")
    @patch("core.system_manager.platform.system", return_value="Windows")
    def test_shutdown_confirmed_executes_shutdown_command(
        self, mock_platform, mock_run, mock_input
    ):
        """Full flow: executor -> system manager -> shutdown.exe (mocked)."""
        mock_run.return_value = MagicMock(returncode=0)
        result = execute_command({
            "intent": "shutdown_computer",
            "target": None,
            "arguments": {},
        })
        self.assertTrue(result["success"])
        mock_run.assert_called_once_with(
            ["shutdown", "/s", "/t", "0"], capture_output=True, check=True
        )

    @patch("core.executor.restart_computer")
    def test_restart_declined_passes_confirmed_false(self, mock_restart):
        """Declining confirmation must pass confirmed=False to the manager."""
        mock_restart.return_value = {
            "success": False,
            "message": "Confirmation required before restarting the computer.",
            "data": {},
        }
        with patch("builtins.input", return_value="no"):
            result = execute_command({
                "intent": "restart_computer",
                "target": None,
                "arguments": {},
            })
        self.assertFalse(result["success"])
        mock_restart.assert_called_once_with(confirmed=False)

    @patch("core.executor.restart_computer")
    def test_restart_confirmed_passes_confirmed_true(self, mock_restart):
        """Confirming must pass confirmed=True to the manager."""
        mock_restart.return_value = {
            "success": True,
            "message": "Computer restarting.",
            "data": {},
        }
        with patch("builtins.input", return_value="yes"):
            result = execute_command({
                "intent": "restart_computer",
                "target": None,
                "arguments": {},
            })
        self.assertTrue(result["success"])
        mock_restart.assert_called_once_with(confirmed=True)

    @patch("builtins.input", return_value="no")
    @patch("core.system_manager.subprocess.run")
    @patch("core.system_manager.platform.system", return_value="Windows")
    def test_restart_declined_never_runs_shutdown_exe(
        self, mock_platform, mock_run, mock_input
    ):
        """Declining restart must never run shutdown.exe (mocked)."""
        result = execute_command({
            "intent": "restart_computer",
            "target": None,
            "arguments": {},
        })
        self.assertFalse(result["success"])
        self.assertIn("confirmation required", result["message"].lower())
        mock_run.assert_not_called()

    @patch("core.executor.create_folder")
    def test_create_folder_routes_to_manager(self, mock_create):
        """CREATE_FOLDER should call file_manager.create_folder."""
        mock_create.return_value = {
            "success": True,
            "message": "Folder 'Projects' created successfully.",
            "data": {"path": r"C:\fake\Desktop\Projects"},
        }
        result = execute_command({
            "intent": "create_folder",
            "target": "Projects",
            "arguments": {},
        })
        self.assertTrue(result["success"])
        mock_create.assert_called_once_with("Projects")

    @patch("core.executor.create_folder")
    def test_create_folder_missing_target(self, mock_create):
        """Missing target should not call the file manager."""
        result = execute_command({
            "intent": "create_folder",
            "target": None,
            "arguments": {},
        })
        self.assertFalse(result["success"])
        self.assertIn("requires a target", result["data"]["validation_errors"][0])
        mock_create.assert_not_called()

    @patch("core.executor.delete_file")
    def test_delete_file_requires_confirmation(self, mock_delete):
        """Refusing confirmation should not call the file manager."""
        with patch("builtins.input", return_value="no"):
            result = execute_command({
                "intent": "delete_file",
                "target": "test.txt",
                "arguments": {},
            })
        self.assertFalse(result["success"])
        self.assertIn("cancelled", result["message"].lower())
        mock_delete.assert_not_called()

    @patch("core.executor.delete_file")
    def test_delete_file_confirmed_routes_with_confirmed(self, mock_delete):
        """Confirming should call delete_file with confirmed=True."""
        mock_delete.return_value = {
            "success": True,
            "message": "File deleted successfully.",
            "data": {"path": r"C:\fake\Desktop\test.txt"},
        }
        with patch("builtins.input", return_value="yes"):
            result = execute_command({
                "intent": "delete_file",
                "target": "test.txt",
                "arguments": {},
            })
        self.assertTrue(result["success"])
        mock_delete.assert_called_once_with("test.txt", confirmed=True)

    @patch("core.executor.delete_file")
    def test_delete_file_arbitrary_text_is_not_confirmation(self, mock_delete):
        """Only yes/y/confirm count — arbitrary text cancels."""
        for answer in ("maybe", "delete", "sure", "yep", "1", ""):
            with patch("builtins.input", return_value=answer):
                result = execute_command({
                    "intent": "delete_file",
                    "target": "test.txt",
                    "arguments": {},
                })
            self.assertFalse(
                result["success"],
                f"input {answer!r} must not confirm deletion",
            )
        mock_delete.assert_not_called()

    @patch("core.executor.open_file")
    def test_open_file_routes_to_manager(self, mock_open):
        """OPEN_FILE should call file_manager.open_file."""
        mock_open.return_value = {
            "success": True,
            "message": "File opened: notes.txt",
            "data": {"path": r"C:\fake\Desktop\notes.txt"},
        }
        result = execute_command({
            "intent": "open_file",
            "target": "notes.txt",
            "arguments": {},
        })
        self.assertTrue(result["success"])
        mock_open.assert_called_once_with("notes.txt")

    @patch("core.executor.search_file")
    def test_search_file_routes_to_manager(self, mock_search):
        """SEARCH_FILE should call file_manager.search_file."""
        mock_search.return_value = {
            "success": True,
            "message": "Found 1 matching file.",
            "data": {"files": [r"C:\fake\Desktop\report.pdf"]},
        }
        result = execute_command({
            "intent": "search_file",
            "target": "report.pdf",
            "arguments": {},
        })
        self.assertTrue(result["success"])
        self.assertIn("report.pdf", str(result["data"]))
        mock_search.assert_called_once_with("report.pdf")

    @patch("core.executor.open_browser")
    def test_open_browser_routes_to_manager(self, mock_open_browser):
        """OPEN_BROWSER should call browser_manager.open_browser."""
        mock_open_browser.return_value = {
            "success": True,
            "message": "Browser opened successfully.",
            "data": {},
        }
        result = execute_command({
            "intent": "open_browser",
            "target": None,
            "arguments": {},
        })
        self.assertTrue(result["success"])
        mock_open_browser.assert_called_once_with()

    @patch("core.browser_manager.subprocess.Popen")
    @patch(
        "core.browser_manager._resolve_default_browser",
        return_value=r"C:\fake\msedge.exe",
    )
    def test_open_browser_full_flow(self, mock_resolve, mock_popen):
        """Full flow: executor -> browser manager -> default browser (mocked)."""
        result = execute_command({
            "intent": "open_browser",
            "target": None,
            "arguments": {},
        })
        self.assertTrue(result["success"])
        mock_popen.assert_called_once_with(
            [r"C:\fake\msedge.exe"], shell=False
        )

    @patch("core.executor.open_url")
    def test_open_url_routes_to_manager(self, mock_open_url):
        """OPEN_URL should call browser_manager.open_url with the target."""
        mock_open_url.return_value = {
            "success": True,
            "message": "Opened URL successfully.",
            "data": {"url": "https://github.com"},
        }
        result = execute_command({
            "intent": "open_url",
            "target": "https://github.com",
            "arguments": {},
        })
        self.assertTrue(result["success"])
        mock_open_url.assert_called_once_with("https://github.com")

    @patch("core.executor.open_url")
    def test_open_url_missing_target(self, mock_open_url):
        """Missing target should not call the browser manager."""
        result = execute_command({
            "intent": "open_url",
            "target": None,
            "arguments": {},
        })
        self.assertFalse(result["success"])
        self.assertIn("requires a target", result["data"]["validation_errors"][0].lower())
        mock_open_url.assert_not_called()

    @patch("core.browser_manager.webbrowser.open", return_value=True)
    def test_open_url_full_flow(self, mock_web_open):
        """Full flow: executor -> browser manager -> webbrowser (mocked)."""
        result = execute_command({
            "intent": "open_url",
            "target": "https://github.com",
            "arguments": {},
        })
        self.assertTrue(result["success"])
        mock_web_open.assert_called_once_with("https://github.com")

    @patch("core.executor.search_web")
    def test_search_web_routes_to_manager(self, mock_search_web):
        """SEARCH_WEB should call browser_manager.search_web with the query."""
        mock_search_web.return_value = {
            "success": True,
            "message": "Opened web search.",
            "data": {"query": "python tutorials", "url": "..."},
        }
        result = execute_command({
            "intent": "search_web",
            "target": "python tutorials",
            "arguments": {},
        })
        self.assertTrue(result["success"])
        mock_search_web.assert_called_once_with("python tutorials")

    @patch("core.executor.search_web")
    def test_search_web_missing_target(self, mock_search_web):
        """Missing target should not call the browser manager."""
        result = execute_command({
            "intent": "search_web",
            "target": None,
            "arguments": {},
        })
        self.assertFalse(result["success"])
        mock_search_web.assert_not_called()

    @patch("core.executor.launch_application")
    def test_result_structure(self, mock_launch):
        """Test that application results have the correct structure."""
        mock_launch.return_value = {
            "success": True,
            "message": "Notepad opened successfully.",
            "data": {
                "application": "notepad",
                "path": r"C:\Windows\System32\notepad.exe",
            },
        }
        result = execute_command({
            "intent": "open_application",
            "target": "notepad",
            "arguments": {},
        })
        self.assertIn("success", result)
        self.assertIn("message", result)
        self.assertIn("data", result)
        self.assertIsInstance(result["success"], bool)
        self.assertIsInstance(result["message"], str)
        self.assertIsInstance(result["data"], dict)
        self.assertIsInstance(result["data"]["application"], str)
        self.assertIsInstance(result["data"]["path"], str)


class TestHistoryRouting(unittest.TestCase):
    """HISTORY / CLEAR_HISTORY route to Command History only (Step 20)."""

    def setUp(self):
        from core.command_history import CommandHistory, set_command_history

        self.history = CommandHistory(max_entries=20)
        set_command_history(self.history)

    def tearDown(self):
        from core.command_history import set_command_history

        set_command_history(None)

    @staticmethod
    def _command(intent):
        return {"intent": intent, "target": None, "arguments": {}}

    @patch("core.executor.launch_application")
    @patch("core.executor.open_browser")
    @patch("core.executor.lock_computer")
    def test_history_routes_to_history_not_managers(
        self, mock_lock, mock_browser, mock_launch
    ):
        self.history.add_entry(
            input_text="open chrome", command_text="open chrome",
            intent="open_application", success=True, message="ok",
        )
        result = execute_command(self._command("history"))
        self.assertTrue(result["success"])
        self.assertEqual(result["data"]["kind"], "history")
        self.assertEqual(len(result["data"]["entries"]), 1)
        mock_lock.assert_not_called()
        mock_browser.assert_not_called()
        mock_launch.assert_not_called()

    def test_clear_history_routes_to_history(self):
        self.history.add_entry(input_text="a", command_text="a")
        result = execute_command(self._command("clear_history"))
        self.assertTrue(result["success"])
        self.assertEqual(result["message"], "Command history cleared.")
        self.assertEqual(self.history.count(), 0)

    @patch("os.system")
    @patch("subprocess.run")
    @patch("subprocess.Popen")
    def test_history_commands_never_touch_a_shell(
        self, mock_popen, mock_run, mock_system
    ):
        execute_command(self._command("history"))
        execute_command(self._command("clear_history"))
        self.assertEqual(
            mock_system.call_count + mock_run.call_count
            + mock_popen.call_count,
            0,
        )

    def test_history_is_read_only(self):
        """Executing HISTORY must not clear or reorder anything."""
        self.history.add_entry(input_text="a", command_text="a")
        self.history.add_entry(input_text="b", command_text="b")
        execute_command(self._command("history"))
        self.assertEqual(self.history.count(), 2)
        self.assertEqual(
            [e.command_text for e in self.history.get_history()],
            ["a", "b"],
        )


if __name__ == "__main__":
    unittest.main()
