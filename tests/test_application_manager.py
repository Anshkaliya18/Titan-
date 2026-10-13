"""
Tests for the application manager module.

These tests mock filesystem checks and subprocess launching
so no real applications are launched.
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add parent directory to path so we can import core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.application_manager import (
    ALIAS_TO_APP,
    APP_NAMES,
    APPLICATIONS,
    Application,
    close_application,
    get_application,
    is_application,
    is_installed,
    launch_application,
    resolve_application,
)


class TestApplicationLookup(unittest.TestCase):
    """Test application lookup by alias."""

    def test_get_application_chrome(self):
        """chrome should resolve to the chrome application."""
        app = get_application("chrome")
        self.assertIsNotNone(app)
        self.assertEqual(app.name, "chrome")

    def test_get_application_google_chrome(self):
        """google chrome should resolve to the chrome application."""
        app = get_application("google chrome")
        self.assertIsNotNone(app)
        self.assertEqual(app.name, "chrome")

    def test_get_application_explorer(self):
        """explorer should resolve to the explorer application."""
        app = get_application("explorer")
        self.assertIsNotNone(app)
        self.assertEqual(app.name, "explorer")

    def test_get_application_file_explorer(self):
        """file explorer should resolve to the explorer application."""
        app = get_application("file explorer")
        self.assertIsNotNone(app)
        self.assertEqual(app.name, "explorer")

    def test_get_application_calc(self):
        """calc should resolve to the calculator application."""
        app = get_application("calc")
        self.assertIsNotNone(app)
        self.assertEqual(app.name, "calculator")

    def test_get_application_calculator(self):
        """calculator should resolve to the calculator application."""
        app = get_application("calculator")
        self.assertIsNotNone(app)
        self.assertEqual(app.name, "calculator")

    def test_get_application_notepad(self):
        """notepad should resolve to the notepad application."""
        app = get_application("notepad")
        self.assertIsNotNone(app)
        self.assertEqual(app.name, "notepad")

    def test_get_application_edge(self):
        """edge should resolve to the edge application."""
        app = get_application("edge")
        self.assertIsNotNone(app)
        self.assertEqual(app.name, "edge")

    def test_get_application_firefox(self):
        """firefox should resolve to the firefox application."""
        app = get_application("firefox")
        self.assertIsNotNone(app)
        self.assertEqual(app.name, "firefox")


class TestApplicationCaseHandling(unittest.TestCase):
    """Test case-insensitive and whitespace-tolerant lookup."""

    def test_case_insensitive(self):
        """Lookup should be case-insensitive."""
        self.assertIs(get_application("Chrome"), get_application("chrome"))
        self.assertIs(
            get_application("GOOGLE CHROME"),
            get_application("google chrome"),
        )

    def test_whitespace_tolerant(self):
        """Lookup should tolerate leading/trailing/extra whitespace."""
        self.assertIs(
            get_application(" GOOGLE CHROME "),
            get_application("google chrome"),
        )
        self.assertIs(
            get_application("  chrome  "),
            get_application("chrome"),
        )

    def test_multiword_normalized(self):
        """Extra internal whitespace should be collapsed."""
        self.assertIs(
            get_application("google   chrome"),
            get_application("google chrome"),
        )


class TestApplicationUnknown(unittest.TestCase):
    """Test unknown application handling."""

    def test_get_application_unknown(self):
        """Unknown applications should return None."""
        self.assertIsNone(get_application("unknownapp"))
        self.assertIsNone(get_application(""))

    def test_is_application_unknown(self):
        """is_application should return False for unknown apps."""
        self.assertFalse(is_application("unknownapp"))
        self.assertFalse(is_application("report.pdf"))

    def test_is_application_known(self):
        """is_application should return True for known apps."""
        self.assertTrue(is_application("chrome"))
        self.assertTrue(is_application("calc"))
        self.assertTrue(is_application("file explorer"))


class TestApplicationAliases(unittest.TestCase):
    """Test alias equivalence."""

    def test_chrome_aliases_same_application(self):
        """chrome and google chrome should be the same Application."""
        self.assertIs(get_application("chrome"), get_application("google chrome"))

    def test_explorer_aliases_same_application(self):
        """explorer and file explorer should be the same Application."""
        self.assertIs(
            get_application("explorer"),
            get_application("file explorer"),
        )

    def test_calculator_aliases_same_application(self):
        """calc and calculator should be the same Application."""
        self.assertIs(
            get_application("calc"),
            get_application("calculator"),
        )

    def test_edge_aliases_same_application(self):
        """edge and microsoft edge should be the same Application."""
        self.assertIs(get_application("edge"), get_application("microsoft edge"))

    def test_firefox_aliases_same_application(self):
        """firefox and mozilla firefox should be the same Application."""
        self.assertIs(
            get_application("firefox"),
            get_application("mozilla firefox"),
        )

    def test_alias_to_app_mapping(self):
        """ALIAS_TO_APP should map every alias to the correct app."""
        self.assertEqual(ALIAS_TO_APP["chrome"].name, "chrome")
        self.assertEqual(ALIAS_TO_APP["google chrome"].name, "chrome")
        self.assertEqual(ALIAS_TO_APP["calc"].name, "calculator")
        self.assertEqual(ALIAS_TO_APP["file explorer"].name, "explorer")

    def test_app_names_contains_all_aliases(self):
        """APP_NAMES should contain every alias."""
        expected = {
            "chrome", "google chrome",
            "firefox", "mozilla firefox",
            "edge", "microsoft edge",
            "notepad",
            "calculator", "calc",
            "explorer", "file explorer",
        }
        for alias in expected:
            self.assertIn(alias, APP_NAMES)


class TestApplicationResolution(unittest.TestCase):
    """Test executable path resolution."""

    @patch("core.application_manager.shutil.which", return_value=None)
    @patch("core.application_manager.os.path.isfile")
    def test_resolve_returns_first_existing_path(self, mock_isfile, mock_which):
        """Resolution should return the first existing configured path."""
        app = get_application("chrome")
        target_path = app.executable_paths[1]
        mock_isfile.side_effect = lambda p: p == target_path
        result = resolve_application("chrome")
        self.assertEqual(result, target_path)

    @patch("core.application_manager.shutil.which", return_value=None)
    @patch("core.application_manager.os.path.isfile", return_value=False)
    def test_resolve_skips_missing_paths(self, mock_isfile, mock_which):
        """Resolution should skip non-existent configured paths."""
        result = resolve_application("chrome")
        self.assertIsNone(result)

    @patch("core.application_manager.shutil.which")
    @patch("core.application_manager.os.path.isfile", return_value=False)
    def test_resolve_falls_back_to_path(self, mock_isfile, mock_which):
        """Resolution should fall back to PATH search."""
        fake_path = r"C:\fake\chrome.exe"
        mock_which.return_value = fake_path
        result = resolve_application("chrome")
        self.assertEqual(result, fake_path)
        mock_which.assert_called_once_with("chrome.exe")

    @patch("core.application_manager.shutil.which", return_value=None)
    @patch("core.application_manager.os.path.isfile", return_value=False)
    def test_resolve_returns_none_when_not_found(self, mock_isfile, mock_which):
        """Resolution should return None when executable cannot be found."""
        result = resolve_application("chrome")
        self.assertIsNone(result)

    @patch("core.application_manager.shutil.which", return_value=None)
    @patch("core.application_manager.os.path.isfile", return_value=True)
    def test_resolve_chrome_and_google_chrome_equivalent(
        self, mock_isfile, mock_which
    ):
        """chrome and google chrome should resolve to the same path."""
        self.assertEqual(
            resolve_application("chrome"),
            resolve_application("google chrome"),
        )

    @patch("core.application_manager.shutil.which", return_value=None)
    @patch("core.application_manager.os.path.isfile", return_value=True)
    def test_resolve_explorer_and_file_explorer_equivalent(
        self, mock_isfile, mock_which
    ):
        """explorer and file explorer should resolve to the same path."""
        self.assertEqual(
            resolve_application("explorer"),
            resolve_application("file explorer"),
        )

    @patch("core.application_manager.shutil.which", return_value=None)
    @patch("core.application_manager.os.path.isfile", return_value=True)
    def test_resolve_calc_and_calculator_equivalent(
        self, mock_isfile, mock_which
    ):
        """calc and calculator should resolve to the same path."""
        self.assertEqual(
            resolve_application("calc"),
            resolve_application("calculator"),
        )

    @patch("core.application_manager.shutil.which", return_value=None)
    @patch("core.application_manager.os.path.isfile", return_value=True)
    def test_resolve_notepad(self, mock_isfile, mock_which):
        """notepad should resolve to a notepad.exe path."""
        result = resolve_application("notepad")
        self.assertIsNotNone(result)
        self.assertTrue(result.endswith("notepad.exe"))

    @patch("core.application_manager.shutil.which", return_value=None)
    @patch("core.application_manager.os.path.isfile", return_value=True)
    def test_resolve_explorer(self, mock_isfile, mock_which):
        """explorer should resolve to an explorer.exe path."""
        result = resolve_application("explorer")
        self.assertIsNotNone(result)
        self.assertTrue(result.endswith("explorer.exe"))

    @patch("core.application_manager.shutil.which", return_value=None)
    @patch("core.application_manager.os.path.isfile", return_value=True)
    def test_resolve_calc(self, mock_isfile, mock_which):
        """calc should resolve to a calc.exe path."""
        result = resolve_application("calc")
        self.assertIsNotNone(result)
        self.assertTrue(result.endswith("calc.exe"))

    def test_resolve_unknown_application(self):
        """Unknown applications should return None (not configured)."""
        self.assertIsNone(resolve_application("unknownapp"))


class TestApplicationInstalled(unittest.TestCase):
    """Test installed checks."""

    @patch("core.application_manager.resolve_application", return_value=r"C:\fake\chrome.exe")
    def test_is_installed_true(self, mock_resolve):
        """is_installed should return True when the app resolves."""
        self.assertTrue(is_installed("chrome"))
        mock_resolve.assert_called_once_with("chrome")

    @patch("core.application_manager.resolve_application", return_value=None)
    def test_is_installed_false(self, mock_resolve):
        """is_installed should return False when the app cannot be resolved."""
        self.assertFalse(is_installed("chrome"))
        mock_resolve.assert_called_once_with("chrome")

    @patch("core.application_manager.resolve_application", return_value=None)
    def test_is_installed_unknown(self, mock_resolve):
        """is_installed should return False for unknown apps."""
        self.assertFalse(is_installed("spotify"))


class TestApplicationLaunch(unittest.TestCase):
    """Test safe application launching."""

    @patch("core.application_manager.subprocess.Popen")
    @patch("core.application_manager._resolve_executable")
    @patch("core.application_manager.get_application")
    def test_launch_uses_resolved_path_with_shell_false(
        self, mock_get, mock_resolve, mock_popen
    ):
        """Launch should use the resolved path and shell=False."""
        app = Application(
            name="chrome",
            aliases=["chrome", "google chrome"],
            executable_paths=[],
            executable="chrome.exe",
        )
        mock_get.return_value = app
        mock_resolve.return_value = r"C:\fake\chrome.exe"
        mock_popen.return_value = MagicMock()

        result = launch_application("chrome")

        self.assertTrue(result["success"])
        self.assertEqual(result["data"]["application"], "chrome")
        self.assertEqual(result["data"]["path"], r"C:\fake\chrome.exe")
        mock_popen.assert_called_once_with(
            [r"C:\fake\chrome.exe"], shell=False
        )

    @patch("core.application_manager.subprocess.Popen")
    @patch("core.application_manager._resolve_executable")
    @patch("core.application_manager.get_application")
    def test_launch_does_not_execute_raw_user_input(
        self, mock_get, mock_resolve, mock_popen
    ):
        """Launch should never pass raw user input to subprocess."""
        app = Application(
            name="notepad",
            aliases=["notepad"],
            executable_paths=[],
            executable="notepad.exe",
        )
        mock_get.return_value = app
        mock_resolve.return_value = r"C:\Windows\System32\notepad.exe"
        mock_popen.return_value = MagicMock()

        launch_application("notepad")

        args, kwargs = mock_popen.call_args
        self.assertEqual(args[0], [r"C:\Windows\System32\notepad.exe"])
        self.assertEqual(kwargs.get("shell"), False)

    @patch("core.application_manager._resolve_executable", return_value=None)
    @patch("core.application_manager.get_application")
    def test_launch_known_but_not_installed(self, mock_get, mock_resolve):
        """Known app that cannot be located should return a clean error."""
        app = Application(
            name="chrome",
            aliases=["chrome", "google chrome"],
            executable_paths=[],
            executable="chrome.exe",
        )
        mock_get.return_value = app
        result = launch_application("chrome")
        self.assertFalse(result["success"])
        self.assertIn("not installed", result["message"].lower())
        self.assertEqual(result["data"]["application"], "chrome")
        self.assertIsNone(result["data"]["path"])

    @patch("core.application_manager.get_application", return_value=None)
    def test_launch_unknown_application(self, mock_get):
        """Unknown applications should return a clean error."""
        result = launch_application("spotify")
        self.assertFalse(result["success"])
        self.assertIn("not allowed", result["message"].lower())
        self.assertIsNone(result["data"]["application"])
        self.assertIsNone(result["data"]["path"])

    @patch("core.application_manager.subprocess.Popen", side_effect=Exception("boom"))
    @patch("core.application_manager._resolve_executable")
    @patch("core.application_manager.get_application")
    def test_launch_exception_returns_error(self, mock_get, mock_resolve, mock_popen):
        """A launch exception should return a clean error."""
        app = Application(
            name="notepad",
            aliases=["notepad"],
            executable_paths=[],
            executable="notepad.exe",
        )
        mock_get.return_value = app
        mock_resolve.return_value = r"C:\Windows\System32\notepad.exe"
        result = launch_application("notepad")
        self.assertFalse(result["success"])
        self.assertIn("failed to launch", result["message"].lower())

    @patch("core.application_manager.subprocess.Popen")
    @patch("core.application_manager._resolve_executable")
    @patch("core.application_manager.get_application")
    def test_launch_chrome_and_google_chrome_equivalent(
        self, mock_get, mock_resolve, mock_popen
    ):
        """chrome and google chrome should launch the same executable."""
        app = Application(
            name="chrome",
            aliases=["chrome", "google chrome"],
            executable_paths=[],
            executable="chrome.exe",
        )
        mock_get.return_value = app
        mock_resolve.return_value = r"C:\fake\chrome.exe"
        mock_popen.return_value = MagicMock()

        result_chrome = launch_application("chrome")
        result_google = launch_application("google chrome")

        self.assertTrue(result_chrome["success"])
        self.assertTrue(result_google["success"])
        self.assertEqual(
            result_chrome["data"]["path"], result_google["data"]["path"]
        )


class TestApplicationClose(unittest.TestCase):
    """Test application closing."""

    @patch("core.application_manager.subprocess.run")
    @patch("core.application_manager.get_application")
    def test_close_uses_image_name(self, mock_get, mock_run):
        """Close should use the registered process image name."""
        app = Application(
            name="chrome",
            aliases=["chrome", "google chrome"],
            executable_paths=[],
            executable="chrome.exe",
        )
        mock_get.return_value = app
        mock_run.return_value = MagicMock(returncode=0)

        result = close_application("chrome")

        self.assertTrue(result["success"])
        mock_run.assert_called_once_with(
            ["taskkill", "/f", "/im", "chrome.exe"],
            capture_output=True,
        )

    @patch("core.application_manager.subprocess.run")
    @patch("core.application_manager.get_application")
    def test_close_reports_failure_when_taskkill_fails(
        self, mock_get, mock_run
    ):
        """A non-zero taskkill exit code must not report success.

        Regression: close_application used to ignore the exit code and
        reported success even when the process was not running.
        """
        app = Application(
            name="chrome",
            aliases=["chrome", "google chrome"],
            executable_paths=[],
            executable="chrome.exe",
        )
        mock_get.return_value = app
        mock_run.return_value = MagicMock(returncode=1)

        result = close_application("chrome")

        self.assertFalse(result["success"])
        self.assertIn("not running", result["message"].lower())
        mock_run.assert_called_once_with(
            ["taskkill", "/f", "/im", "chrome.exe"],
            capture_output=True,
        )

    @patch("core.application_manager.get_application", return_value=None)
    def test_close_unknown_application(self, mock_get):
        """Unknown applications should return a clean error."""
        result = close_application("spotify")
        self.assertFalse(result["success"])
        self.assertIn("not allowed", result["message"].lower())


class TestApplicationRegistry(unittest.TestCase):
    """Test the central application registry structure."""

    def test_registry_contains_required_applications(self):
        """The registry should contain all required applications."""
        names = {app.name for app in APPLICATIONS}
        for required in ["chrome", "firefox", "edge", "notepad", "calculator", "explorer"]:
            self.assertIn(required, names)

    def test_application_dataclass_fields(self):
        """Each Application should have the required fields."""
        app = get_application("chrome")
        self.assertEqual(app.name, "chrome")
        self.assertIn("chrome", app.aliases)
        self.assertIn("google chrome", app.aliases)
        self.assertTrue(len(app.executable_paths) > 0)
        self.assertEqual(app.executable, "chrome.exe")
        self.assertTrue(len(app.description) > 0)

    def test_all_aliases_are_unique(self):
        """No alias should be registered to two different applications."""
        seen = {}
        for app in APPLICATIONS:
            for alias in app.aliases:
                self.assertNotIn(alias, seen, f"Alias '{alias}' registered twice")
                seen[alias] = app.name


class TestBrowserFlag(unittest.TestCase):
    """Browsers are marked in the shared registry (Step 11)."""

    def test_browsers_are_flagged(self):
        """Chrome, Firefox, and Edge must be flagged as browsers."""
        for alias in ["chrome", "google chrome", "firefox", "edge", "microsoft edge"]:
            app = get_application(alias)
            self.assertTrue(app.is_browser, f"{alias} should be a browser")

    def test_non_browsers_not_flagged(self):
        """Notepad, calc, and explorer must not be flagged as browsers."""
        for alias in ["notepad", "calc", "explorer"]:
            app = get_application(alias)
            self.assertFalse(app.is_browser, f"{alias} should not be a browser")

    def test_chrome_aliases_share_one_definition(self):
        """'chrome' and 'google chrome' must resolve to the same Application."""
        self.assertIs(get_application("chrome"), get_application("google chrome"))


if __name__ == "__main__":
    unittest.main()
