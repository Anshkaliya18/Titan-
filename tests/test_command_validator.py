"""
Tests for the command validator.

Covers structural validation, target requirements, URL scheme
enforcement, and security tests confirming that invalid commands
never reach a manager.
"""

import os
import sys
import unittest
from unittest.mock import patch

# Add parent directory to path so we can import core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.command_validator import validate_command


def _valid(**overrides):
    """Build a valid command dict, applying overrides.

    Intent is normalized to lowercase to match the real command
    format produced by the parser (Intent enum values).
    """
    command = {"intent": "OPEN_APPLICATION", "target": "chrome", "arguments": {}}
    command.update(overrides)
    if isinstance(command["intent"], str):
        command["intent"] = command["intent"].lower()
    return command


class TestValidCommands(unittest.TestCase):
    """Structurally valid commands pass validation."""

    def test_valid_application_command(self):
        result = validate_command(_valid())
        self.assertTrue(result["valid"])
        self.assertEqual(result["message"], "Command is valid.")
        self.assertEqual(result["errors"], [])

    def test_valid_close_application(self):
        result = validate_command(_valid(intent="CLOSE_APPLICATION", target="notepad"))
        self.assertTrue(result["valid"])

    def test_valid_create_folder(self):
        result = validate_command(_valid(intent="CREATE_FOLDER", target="Projects"))
        self.assertTrue(result["valid"])

    def test_valid_delete_file(self):
        result = validate_command(_valid(intent="DELETE_FILE", target="test.txt"))
        self.assertTrue(result["valid"])

    def test_valid_open_file(self):
        result = validate_command(_valid(intent="OPEN_FILE", target="report.pdf"))
        self.assertTrue(result["valid"])

    def test_valid_search_file(self):
        result = validate_command(_valid(intent="SEARCH_FILE", target="report"))
        self.assertTrue(result["valid"])

    def test_valid_browser_command_no_target(self):
        result = validate_command(_valid(intent="OPEN_BROWSER", target=None))
        self.assertTrue(result["valid"])

    def test_valid_lock_computer(self):
        result = validate_command(_valid(intent="LOCK_COMPUTER", target=None))
        self.assertTrue(result["valid"])

    def test_valid_shutdown_computer(self):
        result = validate_command(_valid(intent="SHUTDOWN_COMPUTER", target=None))
        self.assertTrue(result["valid"])

    def test_valid_restart_computer(self):
        result = validate_command(_valid(intent="RESTART_COMPUTER", target=None))
        self.assertTrue(result["valid"])

    def test_valid_web_search(self):
        result = validate_command(_valid(intent="SEARCH_WEB", target="python tutorials"))
        self.assertTrue(result["valid"])

    def test_valid_url(self):
        result = validate_command(_valid(intent="OPEN_URL", target="https://github.com"))
        self.assertTrue(result["valid"])

    def test_valid_http_url(self):
        result = validate_command(_valid(intent="OPEN_URL", target="http://example.com"))
        self.assertTrue(result["valid"])

    def test_missing_arguments_treated_as_empty(self):
        command = {"intent": "OPEN_BROWSER", "target": None}
        result = validate_command(command)
        self.assertTrue(result["valid"])

    def test_none_arguments_treated_as_empty(self):
        result = validate_command(_valid(intent="OPEN_BROWSER", target=None, arguments=None))
        self.assertTrue(result["valid"])


class TestHistoryCommands(unittest.TestCase):
    """HISTORY and CLEAR_HISTORY are valid without a target (Step 20)."""

    def test_history_valid_without_target(self):
        result = validate_command(
            {"intent": "history", "target": None, "arguments": {}}
        )
        self.assertTrue(result["valid"])

    def test_clear_history_valid_without_target(self):
        result = validate_command(
            {"intent": "clear_history", "target": None, "arguments": {}}
        )
        self.assertTrue(result["valid"])

    def test_history_uppercase_intent_valid(self):
        result = validate_command(
            {"intent": "HISTORY", "target": None, "arguments": {}}
        )
        self.assertTrue(result["valid"])

    def test_history_ignores_stray_target(self):
        """A target is not required — and not rejected either."""
        result = validate_command(
            {"intent": "history", "target": "anything", "arguments": {}}
        )
        self.assertTrue(result["valid"])

    def test_history_not_in_target_required_intents(self):
        from core.command_validator import TARGET_REQUIRED_INTENTS

        self.assertNotIn("HISTORY", TARGET_REQUIRED_INTENTS)
        self.assertNotIn("CLEAR_HISTORY", TARGET_REQUIRED_INTENTS)


class TestInvalidIntent(unittest.TestCase):
    """Unregistered or malformed intents are rejected."""

    def test_unknown_intent(self):
        result = validate_command(_valid(intent="RUN_RANDOM_COMMAND"))
        self.assertFalse(result["valid"])
        self.assertIn("Unknown intent", result["message"])

    def test_unknown_keyword_intent(self):
        result = validate_command(_valid(intent="UNKNOWN"))
        self.assertFalse(result["valid"])

    def test_missing_intent(self):
        result = validate_command({"target": "chrome", "arguments": {}})
        self.assertFalse(result["valid"])

    def test_none_intent(self):
        result = validate_command({"intent": None, "target": "chrome", "arguments": {}})
        self.assertFalse(result["valid"])

    def test_empty_intent(self):
        result = validate_command(_valid(intent=""))
        self.assertFalse(result["valid"])

    def test_non_string_intent(self):
        result = validate_command(_valid(intent=123))
        self.assertFalse(result["valid"])

    def test_malformed_intent_not_corrected(self):
        """No fuzzy matching — OPEN_APPLCATON must be rejected."""
        result = validate_command(_valid(intent="OPEN_APPLCATON"))
        self.assertFalse(result["valid"])


class TestMissingTargets(unittest.TestCase):
    """Target-required commands reject missing targets."""

    def test_missing_application_target_none(self):
        result = validate_command(_valid(target=None))
        self.assertFalse(result["valid"])

    def test_empty_target(self):
        result = validate_command(_valid(target=""))
        self.assertFalse(result["valid"])

    def test_whitespace_target(self):
        result = validate_command(_valid(target="   "))
        self.assertFalse(result["valid"])

    def test_missing_search_web_query(self):
        result = validate_command(_valid(intent="SEARCH_WEB", target=None))
        self.assertFalse(result["valid"])

    def test_empty_search_query(self):
        result = validate_command(_valid(intent="SEARCH_WEB", target="  "))
        self.assertFalse(result["valid"])

    def test_missing_url_target(self):
        result = validate_command(_valid(intent="OPEN_URL", target=None))
        self.assertFalse(result["valid"])

    def test_missing_file_target(self):
        result = validate_command(_valid(intent="OPEN_FILE", target=None))
        self.assertFalse(result["valid"])

    def test_missing_folder_target(self):
        result = validate_command(_valid(intent="CREATE_FOLDER", target=""))
        self.assertFalse(result["valid"])


class TestTargetTypes(unittest.TestCase):
    """Non-string targets are rejected."""

    def test_list_target(self):
        result = validate_command(_valid(target=[]))
        self.assertFalse(result["valid"])

    def test_dict_target(self):
        result = validate_command(_valid(target={}))
        self.assertFalse(result["valid"])

    def test_integer_target(self):
        result = validate_command(_valid(target=12345))
        self.assertFalse(result["valid"])

    def test_boolean_target(self):
        result = validate_command(_valid(target=True))
        self.assertFalse(result["valid"])


class TestArguments(unittest.TestCase):
    """Arguments must be a dictionary when present."""

    def test_list_arguments(self):
        result = validate_command(_valid(arguments=[]))
        self.assertFalse(result["valid"])

    def test_string_arguments(self):
        result = validate_command(_valid(arguments="something"))
        self.assertFalse(result["valid"])

    def test_integer_arguments(self):
        result = validate_command(_valid(arguments=42))
        self.assertFalse(result["valid"])

    def test_valid_arguments_dict(self):
        result = validate_command(_valid(arguments={"key": "value"}))
        self.assertTrue(result["valid"])


class TestURLValidation(unittest.TestCase):
    """Only http/https URLs are allowed."""

    def test_javascript_scheme_rejected(self):
        result = validate_command(_valid(intent="OPEN_URL", target="javascript:alert(1)"))
        self.assertFalse(result["valid"])

    def test_data_scheme_rejected(self):
        result = validate_command(_valid(intent="OPEN_URL", target="data:text/html,<h1>hi"))
        self.assertFalse(result["valid"])

    def test_file_scheme_rejected(self):
        result = validate_command(_valid(intent="OPEN_URL", target="file:///C:/Windows/System32"))
        self.assertFalse(result["valid"])

    def test_powershell_scheme_rejected(self):
        result = validate_command(_valid(intent="OPEN_URL", target="powershell://calc"))
        self.assertFalse(result["valid"])

    def test_cmd_scheme_rejected(self):
        result = validate_command(_valid(intent="OPEN_URL", target="cmd://del%20C:"))
        self.assertFalse(result["valid"])

    def test_shell_scheme_rejected(self):
        result = validate_command(_valid(intent="OPEN_URL", target="shell://bash"))
        self.assertFalse(result["valid"])

    def test_vbscript_scheme_rejected(self):
        result = validate_command(_valid(intent="OPEN_URL", target="vbscript:msgbox(1)"))
        self.assertFalse(result["valid"])

    def test_plain_text_rejected(self):
        result = validate_command(_valid(intent="OPEN_URL", target="not-a-url"))
        self.assertFalse(result["valid"])

    def test_missing_host_rejected(self):
        result = validate_command(_valid(intent="OPEN_URL", target="https://"))
        self.assertFalse(result["valid"])

    def test_valid_url_with_path_and_query(self):
        result = validate_command(
            _valid(intent="OPEN_URL", target="https://github.com/user/repo?tab=1")
        )
        self.assertTrue(result["valid"])


class TestMalformedCommands(unittest.TestCase):
    """The validator never crashes on malformed input."""

    def test_none_command(self):
        result = validate_command(None)
        self.assertFalse(result["valid"])

    def test_string_command(self):
        result = validate_command("open chrome")
        self.assertFalse(result["valid"])

    def test_list_command(self):
        result = validate_command(["open", "chrome"])
        self.assertFalse(result["valid"])

    def test_integer_command(self):
        result = validate_command(42)
        self.assertFalse(result["valid"])

    def test_result_structure_on_failure(self):
        result = validate_command(None)
        self.assertIn("valid", result)
        self.assertIn("message", result)
        self.assertIn("errors", result)
        self.assertIsInstance(result["errors"], list)
        self.assertTrue(len(result["errors"]) > 0)


class TestSecurityManagersNotReached(unittest.TestCase):
    """Invalid commands must never reach a manager for execution."""

    def test_manager_not_called_on_invalid_intent(self):
        from core.executor import execute_command

        with patch("core.executor.launch_application") as mock_launch:
            result = execute_command(_valid(intent="RUN_RANDOM_COMMAND"))
            self.assertFalse(result["success"])
            mock_launch.assert_not_called()

    def test_manager_not_called_on_missing_target(self):
        from core.executor import execute_command

        with patch("core.executor.launch_application") as mock_launch:
            result = execute_command(_valid(target=None))
            self.assertFalse(result["success"])
            mock_launch.assert_not_called()

    def test_manager_not_called_on_dangerous_url(self):
        from core.executor import execute_command

        with patch("core.executor.open_url") as mock_url:
            result = execute_command(
                _valid(intent="OPEN_URL", target="javascript:alert(1)")
            )
            self.assertFalse(result["success"])
            mock_url.assert_not_called()

    def test_manager_not_called_on_malformed_command(self):
        from core.executor import execute_command

        with patch("core.executor.launch_application") as mock_launch:
            result = execute_command(None)
            self.assertFalse(result["success"])
            mock_launch.assert_not_called()

    def test_valid_command_reaches_manager(self):
        from core.executor import execute_command

        with patch("core.executor.launch_application") as mock_launch:
            mock_launch.return_value = {
                "success": True,
                "message": "Chrome opened successfully.",
                "application": "chrome",
                "path": r"C:\fake\chrome.exe",
            }
            result = execute_command(_valid())
            self.assertTrue(result["success"])
            mock_launch.assert_called_once_with("chrome")


class TestPipelineIntegration(unittest.TestCase):
    """End-to-end: parse → validate → execute (mocked managers)."""

    def _run(self, raw):
        from core.input import normalize_input
        from core.tokenizer import tokenize
        from core.spell_checker import correct_tokens
        from core.intent import detect_intent
        from core.command_parser import parse_command
        from core.command_validator import validate_command

        normalized = normalize_input(raw)
        tokens = tokenize(normalized)
        corrected = correct_tokens(tokens)
        intent = detect_intent(corrected)
        command = parse_command(corrected, intent)
        return command, validate_command(command)

    def test_open_chrome_valid(self):
        command, validation = self._run("open chrome")
        self.assertEqual(command["intent"], "open_application")
        self.assertEqual(command["target"], "chrome")
        self.assertTrue(validation["valid"])

    def test_open_browser_valid(self):
        command, validation = self._run("open browser")
        self.assertEqual(command["intent"], "open_browser")
        self.assertTrue(validation["valid"])

    def test_search_web_valid(self):
        command, validation = self._run("search web python tutorials")
        self.assertEqual(command["intent"], "search_web")
        # Search query content is preserved (Step 26)
        self.assertEqual(command["target"], "python tutorials")
        self.assertTrue(validation["valid"])

    def test_open_url_valid(self):
        command, validation = self._run("open https://github.com")
        self.assertEqual(command["intent"], "open_url")
        self.assertTrue(validation["valid"])

    def test_open_file_explorer_valid(self):
        command, validation = self._run("open file explorer")
        self.assertEqual(command["intent"], "open_application")
        self.assertEqual(command["target"], "file explorer")
        self.assertTrue(validation["valid"])

    def test_open_alone_invalid(self):
        command, validation = self._run("open")
        self.assertEqual(command["intent"], "open_application")
        self.assertIsNone(command["target"])
        self.assertFalse(validation["valid"])

    def test_search_web_alone_invalid(self):
        command, validation = self._run("search web")
        self.assertEqual(command["intent"], "search_web")
        self.assertFalse(validation["valid"])

    def test_open_unknown_app_still_validates(self):
        """Structural validity is not the same as installation check."""
        command, validation = self._run("open something-invalid")
        self.assertEqual(command["intent"], "open_application")
        self.assertTrue(validation["valid"])


if __name__ == "__main__":
    unittest.main()
