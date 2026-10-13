"""
Tests for the response manager.

Covers success, failure, malformed results, confirmation results,
Unicode handling, configuration, and full-pipeline integration
(with mocked managers — nothing real executes).
"""

import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

# Add parent directory to path so we can import core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.response_manager import FALLBACK_MESSAGE, format_response


class TestFormatResponseSuccess(unittest.TestCase):
    """Successful results return their message verbatim."""

    def test_successful_result_returns_message(self):
        result = {
            "success": True,
            "message": "Chrome opened successfully.",
            "data": {},
        }
        self.assertEqual(format_response(result), "Chrome opened successfully.")

    def test_success_result_with_data(self):
        result = {
            "success": True,
            "message": "Browser opened successfully.",
            "data": {"browser": r"C:\Edge\msedge.exe"},
        }
        self.assertEqual(format_response(result), "Browser opened successfully.")

    def test_success_does_not_add_prefix(self):
        result = {"success": True, "message": "Done.", "data": {}}
        response = format_response(result)
        self.assertNotIn("SUCCESS", response.upper())
        self.assertEqual(response, "Done.")


class TestFormatResponseFailure(unittest.TestCase):
    """Failed results return their message verbatim."""

    def test_failed_result_returns_message(self):
        result = {
            "success": False,
            "message": "Chrome is not installed.",
            "data": {},
        }
        self.assertEqual(format_response(result), "Chrome is not installed.")

    def test_failure_does_not_add_error_prefix(self):
        result = {"success": False, "message": "File not found.", "data": {}}
        response = format_response(result)
        self.assertNotIn("ERROR", response.upper())
        self.assertEqual(response, "File not found.")


class TestFormatResponseMalformed(unittest.TestCase):
    """Malformed results produce a safe fallback, never a crash."""

    def test_none_result(self):
        self.assertEqual(format_response(None), FALLBACK_MESSAGE)

    def test_empty_dict(self):
        self.assertEqual(format_response({}), FALLBACK_MESSAGE)

    def test_missing_message(self):
        result = {"success": True, "data": {}}
        self.assertEqual(format_response(result), FALLBACK_MESSAGE)

    def test_non_string_message(self):
        result = {"success": True, "message": 42, "data": {}}
        self.assertEqual(format_response(result), FALLBACK_MESSAGE)

    def test_whitespace_only_message(self):
        result = {"success": True, "message": "   ", "data": {}}
        self.assertEqual(format_response(result), FALLBACK_MESSAGE)

    def test_string_input(self):
        self.assertEqual(format_response("not a dict"), FALLBACK_MESSAGE)

    def test_list_input(self):
        self.assertEqual(format_response([1, 2, 3]), FALLBACK_MESSAGE)

    def test_message_is_stripped(self):
        result = {"success": True, "message": "  Done.  ", "data": {}}
        self.assertEqual(format_response(result), "Done.")

    def test_exception_inside_formatting_returns_fallback(self):
        class ExplodingDict(dict):
            def get(self, *args, **kwargs):
                raise RuntimeError("boom")

        self.assertEqual(format_response(ExplodingDict()), FALLBACK_MESSAGE)


class TestConfirmationResults(unittest.TestCase):
    """Confirmation results are displayed verbatim, never approved."""

    def test_confirmation_message_preserved(self):
        result = {
            "success": False,
            "message": "Shutdown requires confirmation.",
            "data": {"requires_confirmation": True},
        }
        self.assertEqual(
            format_response(result), "Shutdown requires confirmation."
        )

    def test_confirmation_not_approved(self):
        result = {
            "success": False,
            "message": "Confirmation required before shutting down the computer.",
            "data": {"requires_confirmation": True},
        }
        response = format_response(result)
        self.assertEqual(
            response,
            "Confirmation required before shutting down the computer.",
        )
        self.assertNotIn("approved", response.lower())

    def test_cancelled_deletion_message_preserved(self):
        result = {
            "success": False,
            "message": "Deletion cancelled.",
            "data": {},
        }
        self.assertEqual(format_response(result), "Deletion cancelled.")


class TestMultiResultFormatting(unittest.TestCase):
    """Multi-command results join one message per command, in order."""

    @staticmethod
    def _entry(message, success=True, command=""):
        return {
            "command": command,
            "success": success,
            "message": message,
            "data": {},
        }

    def test_multiple_results_join_in_order(self):
        result = {
            "success": True,
            "results": [
                self._entry("Chrome opened successfully."),
                self._entry("Calculator opened successfully."),
            ],
            "data": {},
        }
        self.assertEqual(
            format_response(result),
            "Chrome opened successfully.\n"
            "Calculator opened successfully.",
        )

    def test_failure_entry_included_in_place(self):
        result = {
            "success": False,
            "results": [
                self._entry("Chrome opened successfully."),
                self._entry("I couldn't open Firefox.", success=False),
            ],
            "data": {},
        }
        self.assertEqual(
            format_response(result),
            "Chrome opened successfully.\nI couldn't open Firefox.",
        )

    def test_confirmation_prompt_included_as_last_line(self):
        result = {
            "success": False,
            "results": [
                self._entry("Chrome opened successfully."),
                self._entry(
                    "Are you sure you want to shut down the computer? "
                    "(yes/no): ",
                    success=False,
                ),
            ],
            "data": {},
        }
        lines = format_response(result).splitlines()
        self.assertEqual(lines[0], "Chrome opened successfully.")
        self.assertIn("Are you sure", lines[1])

    def test_malformed_entries_are_skipped(self):
        result = {
            "success": True,
            "results": [
                self._entry("First."),
                "not a dict",
                {"success": True},
                self._entry("Second."),
            ],
            "data": {},
        }
        self.assertEqual(format_response(result), "First.\nSecond.")

    def test_all_malformed_entries_fall_back(self):
        result = {"success": True, "results": [42, None], "data": {}}
        self.assertEqual(format_response(result), FALLBACK_MESSAGE)

    def test_empty_results_fall_back_to_message_logic(self):
        result = {"success": True, "results": [], "data": {}}
        self.assertEqual(format_response(result), FALLBACK_MESSAGE)

    def test_messages_are_stripped(self):
        result = {
            "success": True,
            "results": [self._entry("  Done.  ")],
            "data": {},
        }
        self.assertEqual(format_response(result), "Done.")

    def test_single_result_dict_still_uses_message(self):
        """Legacy single-command behavior is unchanged."""
        result = {
            "success": True,
            "message": "Chrome opened successfully.",
            "data": {},
        }
        self.assertEqual(format_response(result), "Chrome opened successfully.")

    def test_formatting_never_executes_anything(self):
        """The multi-result branch is pure formatting."""
        with patch("os.system") as mock_system, \
                patch("subprocess.run") as mock_run:
            result = {
                "success": True,
                "results": [self._entry("Safe.")],
                "data": {},
            }
            self.assertEqual(format_response(result), "Safe.")
        mock_system.assert_not_called()
        mock_run.assert_not_called()


class TestUnicode(unittest.TestCase):
    """Unicode messages pass through correctly."""

    def test_unicode_message(self):
        result = {
            "success": True,
            "message": "Fichier ouvert avec succès — 完成 ✓",
            "data": {},
        }
        self.assertEqual(
            format_response(result), "Fichier ouvert avec succès — 完成 ✓"
        )

    def test_emoji_message(self):
        result = {"success": True, "message": "Done ✓", "data": {}}
        self.assertEqual(format_response(result), "Done ✓")


class TestConfiguration(unittest.TestCase):
    """Debug configuration comes from the Configuration Manager."""

    def test_debug_is_boolean(self):
        from core.config_manager import get_value

        self.assertIsInstance(get_value("general", "debug"), bool)

    def test_debug_defaults_to_off(self):
        from core.config_manager import DEFAULT_CONFIG

        self.assertFalse(DEFAULT_CONFIG["general"]["debug"])


class TestPipelineIntegration(unittest.TestCase):
    """Full pipeline with mocked managers — nothing real executes."""

    def _run_pipeline(self, raw):
        from core.input import normalize_input
        from core.tokenizer import tokenize
        from core.spell_checker import correct_tokens
        from core.intent import detect_intent
        from core.command_parser import parse_command
        from core.executor import execute_command

        normalized = normalize_input(raw)
        tokens = tokenize(normalized)
        corrected = correct_tokens(tokens)
        intent = detect_intent(corrected)
        command = parse_command(corrected, intent)
        result = execute_command(command)
        return format_response(result)

    @patch("core.executor.launch_application")
    def test_open_chrome(self, mock_launch):
        mock_launch.return_value = {
            "success": True,
            "message": "Chrome opened successfully.",
            "application": "chrome",
            "path": r"C:\fake\chrome.exe",
        }
        self.assertEqual(
            self._run_pipeline("open chrome"), "Chrome opened successfully."
        )

    @patch("core.executor.launch_application")
    def test_open_file_explorer(self, mock_launch):
        mock_launch.return_value = {
            "success": True,
            "message": "Explorer opened successfully.",
            "application": "explorer",
            "path": r"C:\Windows\explorer.exe",
        }
        self.assertEqual(
            self._run_pipeline("open file explorer"),
            "Explorer opened successfully.",
        )

    @patch("core.executor.open_browser")
    def test_open_browser(self, mock_browser):
        mock_browser.return_value = {
            "success": True,
            "message": "Browser opened successfully.",
            "data": {"browser": r"C:\fake\msedge.exe"},
        }
        self.assertEqual(
            self._run_pipeline("open browser"), "Browser opened successfully."
        )

    @patch("core.executor.search_web")
    def test_search_web(self, mock_search):
        mock_search.return_value = {
            "success": True,
            "message": "Opened web search.",
            "data": {
                "query": "python tutorial",
                "url": "https://www.google.com/search?q=python+tutorial",
            },
        }
        self.assertEqual(
            self._run_pipeline("search web python tutorials"),
            "Opened web search.",
        )

    @patch("core.executor.lock_computer")
    def test_lock_computer(self, mock_lock):
        mock_lock.return_value = {
            "success": True,
            "message": "Computer locked successfully.",
            "data": {},
        }
        self.assertEqual(
            self._run_pipeline("lock computer"),
            "Computer locked successfully.",
        )

    @patch("core.executor.launch_application")
    def test_unknown_app_failure(self, mock_launch):
        mock_launch.return_value = {
            "success": False,
            "message": "Application 'unknownapp' is not allowed or configured.",
            "application": None,
            "path": None,
        }
        self.assertEqual(
            self._run_pipeline("opn unknownapp"),
            "Application 'unknownapp' is not allowed or configured.",
        )

    @patch("core.executor.launch_application")
    def test_dangerous_input_surfaces_cleanly(self, mock_launch):
        """A dangerous input produces a clean message, never a crash."""
        mock_launch.return_value = {
            "success": False,
            "message": "Application 'javascript : alert ( a )' is not allowed or configured.",
            "application": None,
            "path": None,
        }
        self.assertEqual(
            self._run_pipeline("open javascript:alert(1)"),
            "Application 'javascript : alert ( a )' is not allowed or configured.",
        )

    @patch("core.executor.shutdown_computer")
    @patch("builtins.input", return_value="no")
    def test_shutdown_declined_shows_confirmation_message(
        self, mock_input, mock_shutdown
    ):
        mock_shutdown.return_value = {
            "success": False,
            "message": "Confirmation required before shutting down the computer.",
            "data": {},
        }
        self.assertEqual(
            self._run_pipeline("shutdown computer"),
            "Confirmation required before shutting down the computer.",
        )


class TestMainDisplay(unittest.TestCase):
    """main.py prints only the formatted response when DEBUG is off."""

    def _run_main_once(self, command):
        import main

        buffer = io.StringIO()
        with patch("main.get_input", side_effect=[command, KeyboardInterrupt()]):
            with redirect_stdout(buffer):
                # KeyboardInterrupt at the loop level is now handled
                # gracefully (goodbye + return) instead of propagating.
                main.main()
        return buffer.getvalue()

    @patch("main.DEBUG", False)
    @patch("main.execute_command")
    def test_main_prints_only_response_when_debug_off(self, mock_execute):
        mock_execute.return_value = {
            "success": True,
            "message": "Chrome opened successfully.",
            "data": {},
        }
        output = self._run_main_once("open chrome")
        self.assertIn("Chrome opened successfully.", output)
        self.assertNotIn("Tokens:", output)
        self.assertNotIn("Intent:", output)

    @patch("main.DEBUG", True)
    @patch("main.execute_command")
    def test_main_prints_debug_info_when_debug_on(self, mock_execute):
        mock_execute.return_value = {
            "success": True,
            "message": "Chrome opened successfully.",
            "data": {},
        }
        output = self._run_main_once("open chrome")
        self.assertIn("Chrome opened successfully.", output)
        self.assertIn("Tokens:", output)
        self.assertIn("Intent:", output)
        self.assertIn("Command:", output)
        self.assertIn("Result:", output)


class TestHistoryFormatting(unittest.TestCase):
    """History results render as a simple numbered list (Step 20)."""

    @staticmethod
    def _result(entries, message="Command history retrieved."):
        return {
            "success": True,
            "message": message,
            "data": {"kind": "history", "entries": entries},
        }

    @staticmethod
    def _entry(display_id, command, success=True):
        return {
            "id": display_id,
            "timestamp": "2026-01-01T00:00:00+00:00",
            "input": "the full original input",
            "command": command,
            "intent": "open_application",
            "success": success,
            "message": "whatever",
        }

    def test_numbered_list_in_order(self):
        result = self._result([
            self._entry(1, "open chrome"),
            self._entry(2, "open calculator"),
            self._entry(3, "search web python tutorials"),
        ])
        self.assertEqual(
            format_response(result),
            "1. open chrome\n"
            "2. open calculator\n"
            "3. search web python tutorials",
        )

    def test_empty_history_message(self):
        result = self._result([], message="There is no command history yet.")
        self.assertEqual(
            format_response(result), "There is no command history yet."
        )
        self.assertNotIn("[", format_response(result))

    def test_empty_history_with_unusable_message(self):
        result = self._result([], message="")
        self.assertEqual(
            format_response(result), "There is no command history yet."
        )

    def test_awaiting_confirmation_suffix(self):
        """A pending command is never displayed as completed."""
        result = self._result([
            self._entry(1, "open chrome", success=True),
            self._entry(2, "shutdown computer", success=None),
        ])
        response = format_response(result)
        self.assertIn("1. open chrome", response)
        self.assertIn("2. shutdown computer — awaiting confirmation", response)

    def test_failed_command_displayed_plainly(self):
        result = self._result([self._entry(1, "blah blah", success=False)])
        self.assertEqual(format_response(result), "1. blah blah")

    def test_no_internal_fields_displayed(self):
        result = self._result([self._entry(1, "open chrome")])
        response = format_response(result)
        self.assertNotIn("timestamp", response)
        self.assertNotIn("HistoryEntry", response)
        self.assertNotIn("input", response)
        self.assertNotIn("open_application", response)
        self.assertNotIn("session", response)

    def test_malformed_entries_skipped(self):
        result = self._result([
            "not a dict",
            {"command": "no id"},
            {"id": 1, "command": ""},
            self._entry(2, "open chrome"),
        ])
        self.assertEqual(format_response(result), "2. open chrome")

    def test_all_malformed_falls_back_to_message(self):
        result = self._result([{"id": "x", "command": None}])
        self.assertEqual(format_response(result), "Command history retrieved.")

    def test_history_inside_chain_renders_numbered(self):
        combined = {
            "success": True,
            "results": [
                {
                    "command": "open chrome",
                    "success": True,
                    "message": "Chrome opened successfully.",
                    "data": {},
                },
                {
                    "command": "history",
                    "success": True,
                    "message": "Command history retrieved.",
                    "data": {
                        "kind": "history",
                        "entries": [self._entry(1, "open chrome")],
                    },
                },
            ],
            "data": {"command_count": 2, "executed_count": 2,
                     "stop_reason": None},
        }
        self.assertEqual(
            format_response(combined),
            "Chrome opened successfully.\n1. open chrome",
        )

    def test_empty_history_inside_chain_uses_message(self):
        combined = {
            "success": True,
            "results": [
                {
                    "command": "history",
                    "success": True,
                    "message": "There is no command history yet.",
                    "data": {"kind": "history", "entries": []},
                },
            ],
            "data": {},
        }
        self.assertEqual(
            format_response(combined), "There is no command history yet."
        )


if __name__ == "__main__":
    unittest.main()
