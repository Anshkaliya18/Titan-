"""
Tests for the browser manager.

Real browsers are never opened — webbrowser, subprocess, and the
Windows registry access are all mocked.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add parent directory to path so we can import core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import core.browser_manager as browser_manager
from core.application_manager import Application
from core.browser_manager import (
    DEFAULT_SEARCH_ENGINE,
    _resolve_default_browser,
    is_valid_url,
    open_browser,
    open_url,
    open_url_in_browser,
    search_web,
)


class TestUrlValidation(unittest.TestCase):
    """URL validation with http/https only."""

    def test_valid_https_google(self):
        self.assertTrue(is_valid_url("https://google.com"))

    def test_valid_https_github(self):
        self.assertTrue(is_valid_url("https://github.com"))

    def test_valid_http_example(self):
        self.assertTrue(is_valid_url("http://example.com"))

    def test_valid_www_and_path(self):
        self.assertTrue(is_valid_url("https://www.google.com/search"))

    def test_valid_port_and_query(self):
        self.assertTrue(is_valid_url("https://example.com:8080/path?q=1"))

    def test_valid_with_surrounding_whitespace(self):
        self.assertTrue(is_valid_url("  https://github.com  "))

    def test_rejects_empty(self):
        self.assertFalse(is_valid_url(""))

    def test_rejects_whitespace_only(self):
        self.assertFalse(is_valid_url("   "))

    def test_rejects_javascript(self):
        self.assertFalse(is_valid_url("javascript:alert(1)"))

    def test_rejects_file_scheme(self):
        self.assertFalse(is_valid_url("file:///C:/Windows/System32"))

    def test_rejects_cmd_scheme(self):
        self.assertFalse(is_valid_url("cmd://del%20C:"))

    def test_rejects_powershell_scheme(self):
        self.assertFalse(is_valid_url("powershell://calc"))

    def test_rejects_data_scheme(self):
        self.assertFalse(is_valid_url("data:text/html,<h1>hi"))

    def test_rejects_vbscript_scheme(self):
        self.assertFalse(is_valid_url("vbscript:msgbox(1)"))

    def test_rejects_missing_scheme(self):
        self.assertFalse(is_valid_url("//github.com"))

    def test_rejects_missing_host(self):
        self.assertFalse(is_valid_url("https://"))

    def test_rejects_plain_text(self):
        self.assertFalse(is_valid_url("not a url at all"))


class TestResolveDefaultBrowser(unittest.TestCase):
    """Default browser resolution from the Windows registry."""

    @patch("core.browser_manager.winreg")
    def test_resolves_quoted_command(self, mock_winreg):
        """A quoted command path should be extracted cleanly."""
        mock_winreg.QueryValueEx.side_effect = [
            ("MSEdgeHTM", 1),
            (r'"C:\Program Files\Edge\msedge.exe" --single-argument %1', 1),
        ]
        self.assertEqual(
            _resolve_default_browser(), r"C:\Program Files\Edge\msedge.exe"
        )

    @patch("core.browser_manager.winreg")
    def test_resolves_unquoted_command(self, mock_winreg):
        """An unquoted command should fall back to the first word."""
        mock_winreg.QueryValueEx.side_effect = [
            ("FirefoxURL", 1),
            (r"C:\Fire\firefox.exe -url %1", 1),
        ]
        self.assertEqual(_resolve_default_browser(), r"C:\Fire\firefox.exe")

    @patch("core.browser_manager.winreg")
    def test_registry_error_returns_none(self, mock_winreg):
        """A registry failure should return None, not crash."""
        mock_winreg.OpenKey.side_effect = OSError("no key")
        self.assertIsNone(_resolve_default_browser())

    @patch("core.browser_manager.winreg")
    def test_command_lookup_error_returns_none(self, mock_winreg):
        """A failed command lookup should return None cleanly."""
        mock_winreg.QueryValueEx.side_effect = [
            ("GhostBrowser", 1),
            OSError("no command key"),
        ]
        self.assertIsNone(_resolve_default_browser())


class TestOpenBrowser(unittest.TestCase):
    """Opening the default browser."""

    @patch("core.browser_manager.subprocess.Popen")
    @patch(
        "core.browser_manager._resolve_default_browser",
        return_value=r"C:\fake\msedge.exe",
    )
    def test_open_browser_launches_default(self, mock_resolve, mock_popen):
        """open_browser() should launch the resolved default browser."""
        result = open_browser()
        self.assertTrue(result["success"])
        self.assertEqual(result["message"], "Browser opened successfully.")
        mock_popen.assert_called_once_with(
            [r"C:\fake\msedge.exe"], shell=False
        )

    @patch(
        "core.browser_manager._resolve_default_browser", return_value=None
    )
    def test_open_browser_unresolved_fails(self, mock_resolve):
        """When the default cannot be resolved, fail cleanly."""
        result = open_browser()
        self.assertFalse(result["success"])
        self.assertIn("default browser", result["message"].lower())

    @patch("core.browser_manager.subprocess.Popen", side_effect=OSError("x"))
    @patch(
        "core.browser_manager._resolve_default_browser",
        return_value=r"C:\fake\msedge.exe",
    )
    def test_open_browser_launch_failure(self, mock_resolve, mock_popen):
        """A failed launch should be a clean failure, not a crash."""
        result = open_browser()
        self.assertFalse(result["success"])


class TestOpenUrl(unittest.TestCase):
    """Opening URLs in the default browser."""

    @patch("core.browser_manager.webbrowser.open", return_value=True)
    def test_open_valid_url(self, mock_open):
        """A valid URL should be handed to webbrowser.open."""
        result = open_url("https://github.com")
        self.assertTrue(result["success"])
        self.assertEqual(result["message"], "Opened URL successfully.")
        self.assertEqual(result["data"]["url"], "https://github.com")
        mock_open.assert_called_once_with("https://github.com")

    @patch("core.browser_manager.webbrowser.open", return_value=True)
    def test_url_is_stripped(self, mock_open):
        """Surrounding whitespace should not break the URL."""
        result = open_url("  https://github.com  ")
        self.assertTrue(result["success"])
        mock_open.assert_called_once_with("https://github.com")

    @patch("core.browser_manager.webbrowser.open", return_value=True)
    def test_javascript_rejected(self, mock_open):
        """javascript: URLs must be rejected without opening anything."""
        result = open_url("javascript:alert(1)")
        self.assertFalse(result["success"])
        self.assertEqual(result["message"], "Invalid URL.")
        mock_open.assert_not_called()

    @patch("core.browser_manager.webbrowser.open", return_value=True)
    def test_file_scheme_rejected(self, mock_open):
        """file: URLs must be rejected without opening anything."""
        result = open_url("file:///C:/Windows/System32/calc.exe")
        self.assertFalse(result["success"])
        mock_open.assert_not_called()

    @patch("core.browser_manager.webbrowser.open", return_value=True)
    def test_cmd_scheme_rejected(self, mock_open):
        """cmd: URLs must be rejected without opening anything."""
        result = open_url("cmd://del%20C:")
        self.assertFalse(result["success"])
        mock_open.assert_not_called()

    @patch("core.browser_manager.webbrowser.open", return_value=True)
    def test_empty_url_rejected(self, mock_open):
        """An empty URL should be a clean failure."""
        result = open_url("")
        self.assertFalse(result["success"])
        self.assertIn("no url", result["message"].lower())
        mock_open.assert_not_called()

    @patch("core.browser_manager.webbrowser.open", return_value=True)
    def test_none_url_rejected(self, mock_open):
        """A missing URL should be a clean failure."""
        result = open_url(None)
        self.assertFalse(result["success"])
        mock_open.assert_not_called()

    @patch("core.browser_manager.webbrowser.open", return_value=False)
    def test_browser_refusal_returns_failure(self, mock_open):
        """When webbrowser.open reports failure, report failure."""
        result = open_url("https://github.com")
        self.assertFalse(result["success"])

    @patch("core.browser_manager.webbrowser.open", side_effect=OSError("x"))
    def test_open_exception_returns_failure(self, mock_open):
        """An exception from webbrowser should be a clean failure."""
        result = open_url("https://github.com")
        self.assertFalse(result["success"])


class TestSearchWeb(unittest.TestCase):
    """Web search URL construction."""

    @patch("core.browser_manager.webbrowser.open", return_value=True)
    def test_search_encodes_spaces(self, mock_open):
        """Spaces must become '+' — never raw spaces in the URL."""
        result = search_web("python tutorials")
        self.assertTrue(result["success"])
        expected = "https://www.google.com/search?q=python+tutorials"
        mock_open.assert_called_once_with(expected)
        self.assertEqual(result["data"]["query"], "python tutorials")
        self.assertEqual(result["data"]["url"], expected)

    @patch("core.browser_manager.webbrowser.open", return_value=True)
    def test_search_encodes_special_characters(self, mock_open):
        """'&' and '+' must be percent-encoded."""
        search_web("c++ & python")
        url = mock_open.call_args[0][0]
        self.assertNotIn(" ", url)
        self.assertIn("%26", url)      # encoded '&'
        self.assertIn("c%2B%2B", url)  # encoded '+'
        self.assertNotIn("c++", url)

    @patch("core.browser_manager.webbrowser.open", return_value=True)
    def test_search_strips_whitespace(self, mock_open):
        """Leading/trailing whitespace should not reach the URL."""
        search_web("  python tutorials  ")
        url = mock_open.call_args[0][0]
        self.assertEqual(url, "https://www.google.com/search?q=python+tutorials")

    @patch("core.browser_manager.webbrowser.open", return_value=True)
    def test_empty_query_fails(self, mock_open):
        """An empty query must fail cleanly, not open an empty search."""
        result = search_web("")
        self.assertFalse(result["success"])
        self.assertIn("no search query", result["message"].lower())
        mock_open.assert_not_called()

    @patch("core.browser_manager.webbrowser.open", return_value=True)
    def test_whitespace_query_fails(self, mock_open):
        """A whitespace-only query must also fail cleanly."""
        result = search_web("   ")
        self.assertFalse(result["success"])
        mock_open.assert_not_called()

    def test_search_engine_has_placeholder(self):
        """The engine template must contain the {query} placeholder."""
        self.assertIn("{query}", DEFAULT_SEARCH_ENGINE)


class TestOpenUrlInBrowser(unittest.TestCase):
    """Opening a URL in a specific named browser."""

    @patch("core.browser_manager.subprocess.Popen")
    @patch(
        "core.browser_manager.resolve_application",
        return_value=r"C:\fake\chrome.exe",
    )
    @patch("core.browser_manager.get_application")
    def test_opens_url_in_named_browser(
        self, mock_get, mock_resolve, mock_popen
    ):
        """The resolved browser executable + URL launch without a shell."""
        chrome = Application(
            name="chrome",
            aliases=["chrome", "google chrome"],
            is_browser=True,
        )
        mock_get.return_value = chrome
        result = open_url_in_browser("google chrome", "https://github.com")
        self.assertTrue(result["success"])
        self.assertEqual(result["data"]["browser"], "chrome")
        mock_popen.assert_called_once_with(
            [r"C:\fake\chrome.exe", "https://github.com"], shell=False
        )

    @patch("core.browser_manager.subprocess.Popen")
    @patch("core.browser_manager.get_application")
    def test_invalid_url_never_launches(self, mock_get, mock_popen):
        """An invalid URL must be rejected before any lookup or launch."""
        result = open_url_in_browser("chrome", "javascript:alert(1)")
        self.assertFalse(result["success"])
        self.assertEqual(result["message"], "Invalid URL.")
        mock_get.assert_not_called()
        mock_popen.assert_not_called()

    @patch("core.browser_manager.subprocess.Popen")
    @patch("core.browser_manager.get_application", return_value=None)
    def test_unknown_browser_fails(self, mock_get, mock_popen):
        """An unknown browser name must fail cleanly."""
        result = open_url_in_browser("netscape", "https://github.com")
        self.assertFalse(result["success"])
        mock_popen.assert_not_called()

    @patch("core.browser_manager.subprocess.Popen")
    @patch("core.browser_manager.get_application")
    def test_non_browser_application_fails(self, mock_get, mock_popen):
        """A non-browser application must be rejected."""
        notepad = Application(name="notepad", aliases=["notepad"])
        mock_get.return_value = notepad
        result = open_url_in_browser("notepad", "https://github.com")
        self.assertFalse(result["success"])
        self.assertIn("not a browser", result["message"].lower())
        mock_popen.assert_not_called()

    @patch("core.browser_manager.subprocess.Popen")
    @patch("core.browser_manager.resolve_application", return_value=None)
    @patch("core.browser_manager.get_application")
    def test_missing_executable_fails(self, mock_get, mock_resolve, mock_popen):
        """A browser that cannot be located must fail cleanly."""
        firefox = Application(
            name="firefox", aliases=["firefox"], is_browser=True
        )
        mock_get.return_value = firefox
        result = open_url_in_browser("firefox", "https://github.com")
        self.assertFalse(result["success"])
        mock_popen.assert_not_called()


class TestResultStructure(unittest.TestCase):
    """Every operation returns the standard {success, message, data} shape."""

    @patch("core.browser_manager.webbrowser.open", return_value=True)
    def test_success_result_shape(self, mock_open):
        result = open_url("https://github.com")
        self.assertEqual(set(result.keys()), {"success", "message", "data"})
        self.assertIsInstance(result["success"], bool)
        self.assertIsInstance(result["message"], str)
        self.assertIsInstance(result["data"], dict)


class TestSafeExecution(unittest.TestCase):
    """Static safety checks on the browser manager source."""

    def test_no_arbitrary_shell_execution(self):
        """The module must never use os.system() or shell=True."""
        with open(browser_manager.__file__, encoding="utf-8") as f:
            source = f.read()
        self.assertNotIn("os.system", source)
        self.assertNotIn("shell=True", source)
        self.assertNotIn("shell = True", source)


if __name__ == "__main__":
    unittest.main()
