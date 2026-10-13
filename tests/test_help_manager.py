"""
Tests for the help manager and the help/discovery pipeline.

Covers help generation, command discovery, registry synchronization,
intent detection, parsing, validation, and executor routing.
"""

import os
import sys
import unittest
from unittest.mock import patch

# Add parent directory to path so we can import core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.help_manager import get_command_help, get_command_list, get_help
from core.intent import Intent
from core.command_validator import validate_command


class TestGetHelp(unittest.TestCase):
    """General help output."""

    def test_get_help_returns_success(self):
        result = get_help()
        self.assertTrue(result["success"])
        self.assertEqual(result["data"]["type"], "help")

    def test_get_help_message_is_string(self):
        result = get_help()
        self.assertIsInstance(result["message"], str)
        self.assertTrue(len(result["message"]) > 0)

    def test_get_help_mentions_categories(self):
        message = get_help()["message"]
        self.assertIn("Applications", message)
        self.assertIn("Files", message)
        self.assertIn("System", message)
        self.assertIn("Browser", message)

    def test_get_help_lists_examples(self):
        message = get_help()["message"]
        self.assertIn("open chrome", message)
        self.assertIn("search web python tutorials", message)
        self.assertIn("lock computer", message)

    def test_get_help_includes_help_commands(self):
        message = get_help()["message"]
        self.assertIn("help", message)
        self.assertIn("commands", message)


class TestGetCommandList(unittest.TestCase):
    """Command list output comes from the registry."""

    def test_get_command_list_returns_success(self):
        result = get_command_list()
        self.assertTrue(result["success"])
        self.assertEqual(result["data"]["type"], "command_list")

    def test_get_command_list_contains_descriptions(self):
        result = get_command_list()
        self.assertIn("Open an installed application.", result["message"])
        self.assertIn("Lock the computer.", result["message"])

    def test_get_command_list_excludes_unknown(self):
        result = get_command_list()
        self.assertNotIn("No supported command was recognized.", result["message"])

    def test_get_command_list_matches_registry(self):
        """Every registered command (except UNKNOWN) appears in the list."""
        from core.command_registry import get_all_commands

        result = get_command_list()
        commands = get_all_commands()
        for intent, cmd in commands.items():
            if intent == "UNKNOWN":
                continue
            self.assertIn(cmd.description, result["message"])


class TestGetCommandHelp(unittest.TestCase):
    """Command-specific help."""

    def test_known_command_help(self):
        result = get_command_help("OPEN_APPLICATION")
        self.assertTrue(result["success"])
        self.assertIn("Open", result["message"])
        self.assertIn("Open an installed application.", result["message"])

    def test_command_help_includes_aliases(self):
        result = get_command_help("OPEN_APPLICATION")
        self.assertIn("open", result["message"])
        self.assertIn("launch", result["message"])

    def test_command_help_case_insensitive(self):
        lower = get_command_help("open_application")
        upper = get_command_help("OPEN_APPLICATION")
        self.assertTrue(lower["success"])
        self.assertEqual(lower["message"], upper["message"])

    def test_unknown_command_returns_safe_fallback(self):
        result = get_command_help("DOES_NOT_EXIST")
        self.assertFalse(result["success"])
        self.assertIn("Unknown command", result["message"])
        self.assertIn("commands", result["message"])

    def test_unknown_command_no_crash(self):
        result = get_command_help("something-random")
        self.assertFalse(result["success"])
        self.assertIsInstance(result["message"], str)


class TestHistoryHelpIntegration(unittest.TestCase):
    """History commands surface through the registry (Step 20 PART 13)."""

    def test_help_lists_history_examples(self):
        message = get_help()["message"]
        self.assertIn("history", message)
        self.assertIn("clear history", message)

    def test_command_list_shows_history_descriptions(self):
        result = get_command_list()
        self.assertIn("View recent commands.", result["message"])
        self.assertIn(
            "Clear the current session's command history.",
            result["message"],
        )

    def test_command_help_lists_all_aliases(self):
        """Aliases live in the registry — one source of truth."""
        history_help = get_command_help("HISTORY")["message"]
        for alias in ("history", "show history", "command history",
                      "recent commands"):
            self.assertIn(alias, history_help)
        clear_help = get_command_help("CLEAR_HISTORY")["message"]
        for alias in ("clear history", "delete history", "remove history"):
            self.assertIn(alias, clear_help)

    def test_command_help_for_history(self):
        result = get_command_help("HISTORY")
        self.assertTrue(result["success"])
        self.assertIn("View recent commands.", result["message"])
        self.assertIn("show history", result["message"])

    def test_command_help_for_clear_history(self):
        result = get_command_help("clear_history")
        self.assertTrue(result["success"])
        self.assertIn(
            "Clear the current session's command history.",
            result["message"],
        )


class TestRegistrySynchronization(unittest.TestCase):
    """Help discovers commands from the registry, not a separate list."""

    def test_help_discovers_new_commands(self):
        from core.command_registry import CommandDefinition, get_all_commands

        # Temporarily add a test command to the registry
        original = get_all_commands()
        test_cmd = CommandDefinition(
            intent="TEST_COMMAND",
            aliases=("test command",),
            description="A test command.",
            category="general",
        )
        # get_all_commands returns a copy, so we patch the module dict
        import core.command_registry as registry

        registry._COMMANDS["TEST_COMMAND"] = test_cmd
        try:
            result = get_command_list()
            self.assertIn("A test command.", result["message"])
            help_result = get_command_help("TEST_COMMAND")
            self.assertTrue(help_result["success"])
            self.assertIn("A test command.", help_result["message"])
        finally:
            del registry._COMMANDS["TEST_COMMAND"]

        # Confirm it was removed
        self.assertNotIn("TEST_COMMAND", get_all_commands())

    def test_help_count_matches_registry(self):
        from core.command_registry import get_all_commands

        result = get_command_list()
        commands = get_all_commands()
        # All non-UNKNOWN commands should be listed
        expected = sum(1 for k in commands if k != "UNKNOWN")
        listed = result["message"].count("\n- ")
        self.assertEqual(listed, expected)


class TestHelpIntentDetection(unittest.TestCase):
    """Intent detection for help phrases."""

    def _run(self, tokens):
        from core.intent import detect_intent

        return detect_intent(tokens)

    def test_help(self):
        self.assertEqual(self._run(["help"]), Intent.HELP)

    def test_what_can_you_do(self):
        self.assertEqual(
            self._run(["what", "can", "you", "do"]), Intent.HELP
        )

    def test_how_can_you_help(self):
        self.assertEqual(
            self._run(["how", "can", "you", "help"]), Intent.HELP
        )

    def test_show_help(self):
        self.assertEqual(self._run(["show", "help"]), Intent.HELP)

    def test_commands(self):
        self.assertEqual(self._run(["commands"]), Intent.LIST_COMMANDS)

    def test_list_commands(self):
        self.assertEqual(
            self._run(["list", "commands"]), Intent.LIST_COMMANDS
        )

    def test_show_commands(self):
        self.assertEqual(
            self._run(["show", "commands"]), Intent.LIST_COMMANDS
        )

    def test_available_commands(self):
        self.assertEqual(
            self._run(["available", "commands"]), Intent.LIST_COMMANDS
        )


class TestHelpParser(unittest.TestCase):
    """Parser produces targetless commands for help intents."""

    def _parse(self, tokens, intent):
        from core.command_parser import parse_command

        return parse_command(tokens, intent)

    def test_help_parse(self):
        result = self._parse(["help"], Intent.HELP)
        self.assertEqual(result["intent"], "help")
        self.assertIsNone(result["target"])
        self.assertEqual(result["arguments"], {})

    def test_commands_parse(self):
        result = self._parse(["commands"], Intent.LIST_COMMANDS)
        self.assertEqual(result["intent"], "list_commands")
        self.assertIsNone(result["target"])
        self.assertEqual(result["arguments"], {})

    def test_what_can_you_do_parse(self):
        result = self._parse(["what", "can", "you", "do"], Intent.HELP)
        self.assertEqual(result["intent"], "help")
        self.assertIsNone(result["target"])


class TestHelpValidator(unittest.TestCase):
    """HELP and LIST_COMMANDS pass validation without a target."""

    def test_help_valid(self):
        result = validate_command({
            "intent": "help",
            "target": None,
            "arguments": {},
        })
        self.assertTrue(result["valid"])

    def test_list_commands_valid(self):
        result = validate_command({
            "intent": "list_commands",
            "target": None,
            "arguments": {},
        })
        self.assertTrue(result["valid"])

    def test_help_uppercase_valid(self):
        result = validate_command({
            "intent": "HELP",
            "target": None,
            "arguments": {},
        })
        self.assertTrue(result["valid"])


class TestHelpExecutor(unittest.TestCase):
    """Executor routes help intents to the Help Manager."""

    def test_help_routes_to_manager(self):
        from core.executor import execute_command

        with patch("core.executor.get_help") as mock_help:
            mock_help.return_value = {
                "success": True,
                "message": "test help",
                "data": {"type": "help"},
            }
            result = execute_command({
                "intent": "help",
                "target": None,
                "arguments": {},
            })
            self.assertTrue(result["success"])
            mock_help.assert_called_once()

    def test_list_commands_routes_to_manager(self):
        from core.executor import execute_command

        with patch("core.executor.get_command_list") as mock_list:
            mock_list.return_value = {
                "success": True,
                "message": "test list",
                "data": {"type": "command_list"},
            }
            result = execute_command({
                "intent": "list_commands",
                "target": None,
                "arguments": {},
            })
            self.assertTrue(result["success"])
            mock_list.assert_called_once()

    def test_help_does_not_launch_anything(self):
        """Help commands must not launch applications or browsers."""
        from core.executor import execute_command

        with patch("core.executor.open_browser") as mock_browser, \
             patch("core.executor.launch_application") as mock_launch:
            execute_command({
                "intent": "help",
                "target": None,
                "arguments": {},
            })
            mock_browser.assert_not_called()
            mock_launch.assert_not_called()


class TestFullPipeline(unittest.TestCase):
    """End-to-end: input → intent → parser → validator → executor."""

    def _run(self, raw):
        from core.input import normalize_input
        from core.tokenizer import tokenize
        from core.spell_checker import correct_tokens
        from core.intent import detect_intent
        from core.command_parser import parse_command
        from core.command_validator import validate_command
        from core.executor import execute_command
        from core.response_manager import format_response

        normalized = normalize_input(raw)
        tokens = tokenize(normalized)
        corrected = correct_tokens(tokens)
        intent = detect_intent(corrected)
        command = parse_command(corrected, intent)
        validation = validate_command(command)
        if not validation["valid"]:
            return format_response({
                "success": False,
                "message": validation["message"],
                "data": {"validation_errors": validation["errors"]},
            })
        result = execute_command(command)
        return format_response(result)

    def test_help_pipeline(self):
        response = self._run("help")
        self.assertIn("TITAN can help you with", response)

    def test_commands_pipeline(self):
        response = self._run("commands")
        self.assertIn("Available commands", response)

    def test_what_can_you_do_pipeline(self):
        response = self._run("what can you do")
        self.assertIn("TITAN can help you with", response)

    def test_how_can_you_help_pipeline(self):
        response = self._run("how can you help")
        self.assertIn("TITAN can help you with", response)

    def test_help_does_not_launch(self):
        with patch("core.executor.open_browser") as mock_browser, \
             patch("core.executor.launch_application") as mock_launch:
            self._run("help")
            mock_browser.assert_not_called()
            mock_launch.assert_not_called()


class TestSpellProtection(unittest.TestCase):
    """Help phrases survive spell correction unchanged."""

    def test_help_unchanged(self):
        from core.spell_checker import correct_token

        self.assertEqual(correct_token("help"), "help")
        self.assertEqual(correct_token("commands"), "commands")

    def test_help_tokens_unchanged(self):
        from core.spell_checker import correct_tokens

        tokens = ["what", "can", "you", "do"]
        self.assertEqual(correct_tokens(tokens), tokens)


if __name__ == "__main__":
    unittest.main()
