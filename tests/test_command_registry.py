"""
Tests for the command registry.

Covers lookup, validation, alias resolution, collision detection,
and full-pipeline integration (with mocked managers).
"""

import os
import sys
import unittest
from unittest.mock import patch

# Add parent directory to path so we can import core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.command_registry import (
    CommandDefinition,
    find_intent_by_alias,
    get_all_commands,
    get_aliases,
    get_command,
    get_description,
    is_valid_intent,
    validate_registry,
)


class TestGetCommand(unittest.TestCase):
    """get_command returns definitions and is case-insensitive."""

    def test_get_command_returns_definition(self):
        command = get_command("OPEN_APPLICATION")
        self.assertIsNotNone(command)
        self.assertEqual(command.intent, "OPEN_APPLICATION")
        self.assertIsInstance(command, CommandDefinition)

    def test_get_command_case_insensitive(self):
        upper = get_command("OPEN_APPLICATION")
        lower = get_command("open_application")
        mixed = get_command("Open_Application")
        self.assertIsNotNone(upper)
        self.assertEqual(upper, lower)
        self.assertEqual(upper, mixed)

    def test_get_command_unknown_returns_none(self):
        self.assertIsNone(get_command("DO_SOMETHING_RANDOM"))

    def test_get_command_non_string_returns_none(self):
        self.assertIsNone(get_command(None))
        self.assertIsNone(get_command(123))


class TestIsValidIntent(unittest.TestCase):
    """is_valid_intent validates intent names."""

    def test_valid_intent(self):
        self.assertTrue(is_valid_intent("OPEN_BROWSER"))

    def test_valid_intent_lowercase(self):
        self.assertTrue(is_valid_intent("open_browser"))

    def test_invalid_intent(self):
        self.assertFalse(is_valid_intent("DO_SOMETHING_RANDOM"))

    def test_unknown_is_valid(self):
        self.assertTrue(is_valid_intent("UNKNOWN"))

    def test_non_string_intent(self):
        self.assertFalse(is_valid_intent(None))
        self.assertFalse(is_valid_intent(123))


class TestFindIntentByAlias(unittest.TestCase):
    """find_intent_by_alias resolves aliases to intent names."""

    def test_reboot_returns_restart(self):
        self.assertEqual(find_intent_by_alias("reboot"), "RESTART_COMPUTER")

    def test_shutdown_alias(self):
        self.assertEqual(find_intent_by_alias("shutdown"), "SHUTDOWN_COMPUTER")

    def test_multi_word_alias(self):
        self.assertEqual(find_intent_by_alias("lock computer"), "LOCK_COMPUTER")

    def test_case_insensitive_alias(self):
        self.assertEqual(find_intent_by_alias("REBOOT"), "RESTART_COMPUTER")

    def test_unknown_alias_returns_none(self):
        self.assertIsNone(find_intent_by_alias("randomword"))

    def test_non_string_alias_returns_none(self):
        self.assertIsNone(find_intent_by_alias(None))
        self.assertIsNone(find_intent_by_alias(123))


class TestHistoryAliases(unittest.TestCase):
    """The history commands are registered in the single registry."""

    def test_history_aliases(self):
        aliases = get_aliases("HISTORY")
        self.assertIn("history", aliases)
        self.assertIn("show history", aliases)
        self.assertIn("command history", aliases)
        self.assertIn("recent commands", aliases)

    def test_clear_history_aliases(self):
        aliases = get_aliases("CLEAR_HISTORY")
        self.assertIn("clear history", aliases)
        self.assertIn("delete history", aliases)
        self.assertIn("remove history", aliases)

    def test_history_alias_resolves_to_intent(self):
        self.assertEqual(find_intent_by_alias("history"), "HISTORY")
        self.assertEqual(
            find_intent_by_alias("recent commands"), "HISTORY"
        )
        self.assertEqual(
            find_intent_by_alias("clear history"), "CLEAR_HISTORY"
        )
        self.assertEqual(
            find_intent_by_alias("delete history"), "CLEAR_HISTORY"
        )

    def test_history_commands_have_descriptions(self):
        self.assertEqual(
            get_description("HISTORY"), "View recent commands."
        )
        self.assertEqual(
            get_description("CLEAR_HISTORY"),
            "Clear the current session's command history.",
        )

    def test_history_commands_are_not_dangerous(self):
        """History must never be treated as a confirmation intent."""
        from core.executor import CONFIRMATION_INTENTS

        self.assertNotIn("history", CONFIRMATION_INTENTS)
        self.assertNotIn("clear_history", CONFIRMATION_INTENTS)


class TestAliasesAndDescription(unittest.TestCase):
    """get_aliases and get_description return registered metadata."""

    def test_get_aliases_open_application(self):
        aliases = get_aliases("OPEN_APPLICATION")
        self.assertIn("open", aliases)
        self.assertIn("launch", aliases)
        self.assertIn("start", aliases)

    def test_get_aliases_case_insensitive(self):
        self.assertEqual(
            get_aliases("open_application"), get_aliases("OPEN_APPLICATION")
        )

    def test_get_aliases_unknown_returns_empty(self):
        self.assertEqual(get_aliases("DO_SOMETHING_RANDOM"), ())

    def test_get_description_open_browser(self):
        description = get_description("OPEN_BROWSER")
        self.assertIsNotNone(description)
        self.assertIsInstance(description, str)
        self.assertTrue(len(description) > 0)

    def test_get_description_unknown_returns_none(self):
        self.assertIsNone(get_description("DO_SOMETHING_RANDOM"))


class TestGetAllCommands(unittest.TestCase):
    """get_all_commands returns every registered command."""

    def test_all_commands_returned(self):
        commands = get_all_commands()
        self.assertEqual(len(commands), 22)

    def test_all_expected_intents_registered(self):
        commands = get_all_commands()
        expected = {
            "OPEN_APPLICATION",
            "CLOSE_APPLICATION",
            "LOCK_COMPUTER",
            "SHUTDOWN_COMPUTER",
            "RESTART_COMPUTER",
            "ENABLE_STARTUP",
            "DISABLE_STARTUP",
            "STARTUP_STATUS",
            "CREATE_FOLDER",
            "DELETE_FILE",
            "OPEN_FILE",
            "SEARCH_FILE",
            "OPEN_BROWSER",
            "SEARCH_WEB",
            "OPEN_URL",
            "HELP",
            "LIST_COMMANDS",
            "HISTORY",
            "CLEAR_HISTORY",
            "FIND_APPLICATION",
            "REFRESH_APPLICATIONS",
            "UNKNOWN",
        }
        self.assertEqual(set(commands.keys()), expected)

    def test_unknown_command_registered(self):
        commands = get_all_commands()
        self.assertIn("UNKNOWN", commands)
        self.assertEqual(commands["UNKNOWN"].intent, "UNKNOWN")

    def test_application_names_not_in_aliases(self):
        """Application names must not appear as command aliases."""
        commands = get_all_commands()
        all_aliases: set[str] = set()
        for command in commands.values():
            all_aliases.update(command.aliases)
        for app_name in ("chrome", "notepad", "calculator", "explorer", "firefox", "edge"):
            self.assertNotIn(app_name, all_aliases)


class TestValidateRegistry(unittest.TestCase):
    """validate_registry detects collisions and invalid definitions."""

    def test_valid_registry_passes(self):
        # The module registry must be valid — this should not raise.
        validate_registry()

    def test_duplicate_alias_detected(self):
        bad = {
            "OPEN_APPLICATION": CommandDefinition(
                "OPEN_APPLICATION", ("open",), "desc"
            ),
            "OPEN_FILE": CommandDefinition(
                "OPEN_FILE", ("open",), "desc"
            ),
        }
        with self.assertRaises(ValueError) as ctx:
            validate_registry(bad)
        self.assertIn("open", str(ctx.exception))

    def test_mismatched_intent_key_detected(self):
        bad = {
            "WRONG_KEY": CommandDefinition(
                "OPEN_APPLICATION", ("open",), "desc"
            ),
        }
        with self.assertRaises(ValueError):
            validate_registry(bad)

    def test_empty_alias_detected(self):
        bad = {
            "OPEN_APPLICATION": CommandDefinition(
                "OPEN_APPLICATION", ("",), "desc"
            ),
        }
        with self.assertRaises(ValueError):
            validate_registry(bad)

    def test_invalid_definition_type_detected(self):
        bad = {"OPEN_APPLICATION": "not a definition"}
        with self.assertRaises(ValueError):
            validate_registry(bad)


class TestPipelineIntegration(unittest.TestCase):
    """The existing pipeline still works with the registry integrated."""

    def _run_pipeline(self, raw):
        from core.input import normalize_input
        from core.tokenizer import tokenize
        from core.spell_checker import correct_tokens
        from core.intent import detect_intent
        from core.command_parser import parse_command

        normalized = normalize_input(raw)
        tokens = tokenize(normalized)
        corrected = correct_tokens(tokens)
        intent = detect_intent(corrected)
        return parse_command(corrected, intent)

    def test_open_chrome(self):
        command = self._run_pipeline("open chrome")
        self.assertEqual(command["intent"], "open_application")
        self.assertEqual(command["target"], "chrome")

    def test_open_google_chrome(self):
        command = self._run_pipeline("open google chrome")
        self.assertEqual(command["intent"], "open_application")
        self.assertEqual(command["target"], "google chrome")

    def test_open_browser(self):
        command = self._run_pipeline("open browser")
        self.assertEqual(command["intent"], "open_browser")

    def test_search_web(self):
        command = self._run_pipeline("search web python tutorials")
        self.assertEqual(command["intent"], "search_web")

    def test_lock_computer(self):
        command = self._run_pipeline("lock computer")
        self.assertEqual(command["intent"], "lock_computer")

    def test_shutdown_computer(self):
        command = self._run_pipeline("shutdown computer")
        self.assertEqual(command["intent"], "shutdown_computer")

    def test_restart_computer(self):
        command = self._run_pipeline("restart computer")
        self.assertEqual(command["intent"], "restart_computer")

    def test_open_file_explorer(self):
        command = self._run_pipeline("open file explorer")
        self.assertEqual(command["intent"], "open_application")
        self.assertEqual(command["target"], "file explorer")

    def test_unknown_input(self):
        command = self._run_pipeline("xyzrandom")
        self.assertEqual(command["intent"], "unknown")


class TestExecutorValidation(unittest.TestCase):
    """The executor rejects unregistered intents via the registry."""

    def test_executor_rejects_invalid_intent(self):
        from core.executor import execute_command

        result = execute_command({
            "intent": "DO_SOMETHING_RANDOM",
            "target": None,
            "arguments": {},
        })
        self.assertFalse(result["success"])
        self.assertEqual(result["message"], "Invalid command.")
        self.assertIn("validation_errors", result["data"])
        self.assertTrue(len(result["data"]["validation_errors"]) > 0)

    def test_executor_routes_valid_intent(self):
        from core.executor import execute_command

        with patch("core.executor.lock_computer") as mock_lock:
            mock_lock.return_value = {
                "success": True,
                "message": "Computer locked successfully.",
                "data": {},
            }
            result = execute_command({
                "intent": "lock_computer",
                "target": None,
                "arguments": {},
            })
            self.assertTrue(result["success"])


if __name__ == "__main__":
    unittest.main()
