"""
Tests for the Natural Language Command Layer (Step 27).

Covers:
- Filler phrase removal
- Synonym mapping
- Intent determination
- Entity extraction
- Confidence scoring
- Security (no shell execution)
- Integration with existing pipeline
- Multi-command compatibility
- Error handling
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.natural_language import (
    NaturalLanguageInterpreter,
    EntityType,
    Entity,
    interpret_natural_language,
    normalize_natural_command,
    correct_tokens,
    correct_token,
)
from core.command_registry import is_valid_intent


class TestFillerRemoval(unittest.TestCase):
    """Filler phrase removal tests."""

    @classmethod
    def setUpClass(cls):
        cls.interpreter = NaturalLanguageInterpreter()

    def test_please_prefix(self):
        result = self.interpreter.interpret("please open chrome", ["please", "open", "chrome"])
        self.assertEqual(result.intent, "OPEN_APPLICATION")

    def test_can_you_prefix(self):
        result = self.interpreter.interpret("can you open chrome", ["can", "you", "open", "chrome"])
        self.assertEqual(result.intent, "OPEN_APPLICATION")

    def test_could_you_prefix(self):
        result = self.interpreter.interpret("could you open chrome", ["could", "you", "open", "chrome"])
        self.assertEqual(result.intent, "OPEN_APPLICATION")

    def test_would_you_prefix(self):
        result = self.interpreter.interpret("would you open chrome", ["would", "you", "open", "chrome"])
        self.assertEqual(result.intent, "OPEN_APPLICATION")

    def test_for_me_suffix(self):
        result = self.interpreter.interpret("open chrome for me", ["open", "chrome", "for", "me"])
        self.assertEqual(result.intent, "OPEN_APPLICATION")

    def test_multiple_fillers(self):
        result = self.interpreter.interpret("could you please open chrome for me", ["could", "you", "please", "open", "chrome", "for", "me"])
        self.assertEqual(result.intent, "OPEN_APPLICATION")


class TestSynonymMapping(unittest.TestCase):
    """Synonym mapping tests."""

    @classmethod
    def setUpClass(cls):
        cls.interpreter = NaturalLanguageInterpreter()

    def test_launch_maps_to_open(self):
        result = self.interpreter.interpret("launch chrome", ["launch", "chrome"])
        self.assertEqual(result.intent, "OPEN_APPLICATION")

    def test_start_maps_to_open(self):
        result = self.interpreter.interpret("start chrome", ["start", "chrome"])
        self.assertEqual(result.intent, "OPEN_APPLICATION")

    def test_run_maps_to_open(self):
        result = self.interpreter.interpret("run chrome", ["run", "chrome"])
        self.assertEqual(result.intent, "OPEN_APPLICATION")

    def test_bring_up_maps_to_open(self):
        result = self.interpreter.interpret("bring up chrome", ["bring", "up", "chrome"])
        self.assertEqual(result.intent, "OPEN_APPLICATION")

    def test_quit_maps_to_close(self):
        result = self.interpreter.interpret("quit chrome", ["quit", "chrome"])
        self.assertEqual(result.intent, "CLOSE_APPLICATION")

    def test_exit_maps_to_close(self):
        result = self.interpreter.interpret("exit chrome", ["exit", "chrome"])
        self.assertEqual(result.intent, "CLOSE_APPLICATION")

    def test_look_up_maps_to_search(self):
        result = self.interpreter.interpret("look up python tutorials", ["look", "up", "python", "tutorials"])
        self.assertEqual(result.intent, "SEARCH_WEB")

    def test_find_online_maps_to_search(self):
        result = self.interpreter.interpret("find online python tutorials", ["find", "online", "python", "tutorials"])
        self.assertEqual(result.intent, "SEARCH_WEB")

    def test_make_maps_to_create(self):
        result = self.interpreter.interpret("make a folder named projects", ["make", "a", "folder", "named", "projects"])
        self.assertEqual(result.intent, "CREATE_FOLDER")

    def test_turn_off_maps_to_shutdown(self):
        result = self.interpreter.interpret("turn off computer", ["turn", "off", "computer"])
        self.assertEqual(result.intent, "SHUTDOWN_COMPUTER")

    def test_power_off_maps_to_shutdown(self):
        result = self.interpreter.interpret("power off computer", ["power", "off", "computer"])
        self.assertEqual(result.intent, "SHUTDOWN_COMPUTER")

    def test_reboot_maps_to_restart(self):
        result = self.interpreter.interpret("reboot computer", ["reboot", "computer"])
        self.assertEqual(result.intent, "RESTART_COMPUTER")


class TestIntentDetermination(unittest.TestCase):
    """Intent determination tests."""

    @classmethod
    def setUpClass(cls):
        cls.interpreter = NaturalLanguageInterpreter()

    def test_open_application(self):
        result = self.interpreter.interpret("open chrome", ["open", "chrome"])
        self.assertEqual(result.intent, "OPEN_APPLICATION")
        self.assertEqual(result.target, "chrome")

    def test_close_application(self):
        result = self.interpreter.interpret("close chrome", ["close", "chrome"])
        self.assertEqual(result.intent, "CLOSE_APPLICATION")
        self.assertEqual(result.target, "chrome")

    def test_open_browser(self):
        result = self.interpreter.interpret("open browser", ["open", "browser"])
        self.assertEqual(result.intent, "OPEN_BROWSER")

    def test_search_web(self):
        result = self.interpreter.interpret("search web python tutorials", ["search", "web", "python", "tutorials"])
        self.assertEqual(result.intent, "SEARCH_WEB")

    def test_open_url(self):
        result = self.interpreter.interpret("open https://github.com", ["open", "https://github.com"])
        self.assertEqual(result.intent, "OPEN_URL")

    def test_lock_computer(self):
        result = self.interpreter.interpret("lock computer", ["lock", "computer"])
        self.assertEqual(result.intent, "LOCK_COMPUTER")

    def test_shutdown_computer(self):
        result = self.interpreter.interpret("shutdown computer", ["shutdown", "computer"])
        self.assertEqual(result.intent, "SHUTDOWN_COMPUTER")

    def test_restart_computer(self):
        result = self.interpreter.interpret("restart computer", ["restart", "computer"])
        self.assertEqual(result.intent, "RESTART_COMPUTER")

    def test_create_folder(self):
        result = self.interpreter.interpret("create a folder named projects", ["create", "a", "folder", "named", "projects"])
        self.assertEqual(result.intent, "CREATE_FOLDER")

    def test_delete_file(self):
        result = self.interpreter.interpret("delete report.txt", ["delete", "report.txt"])
        self.assertEqual(result.intent, "DELETE_FILE")

    def test_open_file(self):
        result = self.interpreter.interpret("open report.txt", ["open", "report.txt"])
        self.assertEqual(result.intent, "OPEN_FILE")

    def test_search_file(self):
        result = self.interpreter.interpret("find report.txt", ["find", "report.txt"])
        self.assertEqual(result.intent, "SEARCH_FILE")

    def test_help(self):
        result = self.interpreter.interpret("what can you do", ["what", "can", "you", "do"])
        self.assertEqual(result.intent, "HELP")

    def test_list_commands(self):
        result = self.interpreter.interpret("show commands", ["show", "commands"])
        self.assertEqual(result.intent, "LIST_COMMANDS")

    def test_history(self):
        result = self.interpreter.interpret("show my recent commands", ["show", "my", "recent", "commands"])
        self.assertEqual(result.intent, "HISTORY")

    def test_clear_history(self):
        result = self.interpreter.interpret("clear my history", ["clear", "my", "history"])
        self.assertEqual(result.intent, "CLEAR_HISTORY")

    def test_unknown_intent(self):
        result = self.interpreter.interpret("maybe open something", ["maybe", "open", "something"])
        self.assertEqual(result.intent, "UNKNOWN")


class TestEntityExtraction(unittest.TestCase):
    """Entity extraction tests."""

    @classmethod
    def setUpClass(cls):
        cls.interpreter = NaturalLanguageInterpreter()

    def test_extract_application(self):
        result = self.interpreter.interpret("open chrome", ["open", "chrome"])
        app_entities = [e for e in result.entities if e.entity_type == EntityType.APPLICATION]
        self.assertEqual(len(app_entities), 1)
        self.assertEqual(app_entities[0].text, "chrome")

    def test_extract_url(self):
        result = self.interpreter.interpret("open https://github.com", ["open", "https://github.com"])
        url_entities = [e for e in result.entities if e.entity_type == EntityType.URL]
        self.assertEqual(len(url_entities), 1)
        self.assertEqual(url_entities[0].text, "https://github.com")

    def test_extract_file_name(self):
        result = self.interpreter.interpret("open report.txt", ["open", "report.txt"])
        file_entities = [e for e in result.entities if e.entity_type == EntityType.FILE]
        self.assertEqual(len(file_entities), 1)
        self.assertEqual(file_entities[0].text, "report.txt")

    def test_extract_folder_name(self):
        result = self.interpreter.interpret("create a folder named projects", ["create", "a", "folder", "named", "projects"])
        folder_entities = [e for e in result.entities if e.entity_type == EntityType.FOLDER]
        self.assertEqual(len(folder_entities), 1)
        self.assertEqual(folder_entities[0].text, "projects")

    def test_extract_search_query(self):
        result = self.interpreter.interpret("search web python tutorials", ["search", "web", "python", "tutorials"])
        query_entities = [e for e in result.entities if e.entity_type == EntityType.SEARCH_QUERY]
        self.assertEqual(len(query_entities), 1)
        self.assertEqual(query_entities[0].text, "python tutorials")


class TestConfidence(unittest.TestCase):
    """Confidence scoring tests."""

    @classmethod
    def setUpClass(cls):
        cls.interpreter = NaturalLanguageInterpreter()

    def test_high_confidence_clear_command(self):
        result = self.interpreter.interpret("open chrome", ["open", "chrome"])
        self.assertGreaterEqual(result.confidence, 0.75)

    def test_low_confidence_ambiguous(self):
        result = self.interpreter.interpret("maybe open something", ["maybe", "open", "something"])
        self.assertLess(result.confidence, 0.75)

    def test_confidence_in_result(self):
        result = self.interpreter.interpret("open chrome", ["open", "chrome"])
        self.assertIsInstance(result.confidence, float)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)


class TestSecurity(unittest.TestCase):
    """Security tests - no shell execution."""

    def test_no_subprocess_import(self):
        import ast

        source = (
            os.path.join(os.path.dirname(__file__), "..", "core", "natural_language.py")
        )
        with open(source, "r") as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertNotIn(
                        alias.name,
                        {"subprocess", "shutil", "pickle", "marshal", "shelve"},
                        f"forbidden import: {alias.name}",
                    )

    def test_no_eval_exec(self):
        import ast

        source = (
            os.path.join(os.path.dirname(__file__), "..", "core", "natural_language.py")
        )
        with open(source, "r") as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name):
                    self.assertNotIn(
                        func.id,
                        {"eval", "exec", "__import__", "compile"},
                        f"forbidden call: {func.id}()",
                    )
                for kw in node.keywords:
                    self.assertNotEqual(
                        kw.arg, "shell", "shell= keyword forbidden"
                    )

    def test_no_file_system_side_effects(self):
        import ast

        source = (
            os.path.join(os.path.dirname(__file__), "..", "core", "natural_language.py")
        )
        with open(source, "r") as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute):
                    self.assertNotIn(
                        func.attr,
                        {"remove", "rename", "unlink", "rmdir", "makedirs", "mkdir"},
                        f"forbidden file operation: {func.attr}()",
                    )


class TestMultiCommandCompatibility(unittest.TestCase):
    """Multi-command compatibility tests."""

    @classmethod
    def setUpClass(cls):
        cls.interpreter = NaturalLanguageInterpreter()

    def test_search_cats_and_dogs_stays_one_command(self):
        result = self.interpreter.interpret("search web cats and dogs", ["search", "web", "cats", "and", "dogs"])
        self.assertEqual(result.intent, "SEARCH_WEB")
        self.assertEqual(result.target, "cats and dogs")

    def test_open_chrome_and_open_calculator(self):
        # Multi-command splitting is done by multi_command_parser, not NL layer
        result1 = self.interpreter.interpret("open chrome", ["open", "chrome"])
        self.assertEqual(result1.intent, "OPEN_APPLICATION")
        self.assertEqual(result1.target, "chrome")

        result2 = self.interpreter.interpret("open calculator", ["open", "calculator"])
        self.assertEqual(result2.intent, "OPEN_APPLICATION")
        self.assertEqual(result2.target, "calculator")

    def test_spell_checker_does_not_split_commands(self):
        result = self.interpreter.interpret("open chrome and open calculator", ["open", "chrome", "and", "open", "calculator"])
        # NL layer should not split - that's multi_command_parser's job
        self.assertIsNotNone(result)


class TestStep27Acceptance(unittest.TestCase):
    def setUp(self):
        self.interpreter = NaturalLanguageInterpreter()

    def test_required_natural_language_examples(self):
        examples = {
            "Could you please launch Google Chrome for me?": ("OPEN_APPLICATION", "google chrome"),
            "Can you start my browser?": ("OPEN_BROWSER", None),
            "Please search the web for Python tutorials.": ("SEARCH_WEB", "Python tutorials"),
            "Show me my recent commands.": ("HISTORY", None),
            "What commands can you do?": ("LIST_COMMANDS", None),
            "open google chrome": ("OPEN_APPLICATION", "google chrome"),
            "open file explorer": ("OPEN_APPLICATION", "file explorer"),
            "create a folder named projects": ("CREATE_FOLDER", "projects"),
            "make a folder called projects": ("CREATE_FOLDER", "projects"),
            "delete report.txt": ("DELETE_FILE", "report.txt"),
            "remove report.txt": ("DELETE_FILE", "report.txt"),
            "open report.txt": ("OPEN_FILE", "report.txt"),
            "find report.txt": ("SEARCH_FILE", "report.txt"),
            "search for report.txt": ("SEARCH_FILE", "report.txt"),
            "search web cats and dogs": ("SEARCH_WEB", "cats and dogs"),
        }
        for text, expected in examples.items():
            with self.subTest(text=text):
                result = self.interpreter.interpret(text)
                self.assertTrue(result.success, result.reason)
                self.assertEqual((result.intent, result.target), expected)

    def test_ambiguous_and_shell_like_requests_are_unknown(self):
        for text in (
            "Could you maybe do something with my computer?",
            "maybe open something",
            "I think maybe you should turn everything off",
            "run powershell payload",
            "open cmd and run whoami",
            "delete everything",
        ):
            with self.subTest(text=text):
                self.assertEqual(self.interpreter.interpret(text).intent, "UNKNOWN")

    def test_every_successful_intent_is_registered(self):
        for text in ("open chrome", "search web cats", "shutdown computer", "what can you do"):
            result = self.interpreter.interpret(text)
            if result.success:
                self.assertTrue(is_valid_intent(result.intent))

    def test_natural_shutdown_still_parks_for_confirmation(self):
        import io
        import main
        from contextlib import redirect_stdout
        from core.session_state import SessionState

        original_session = main.session
        main.session = SessionState()
        try:
            output = io.StringIO()
            with patch("main.get_input", return_value="Could you please shut down my computer?"), \
                 patch("main.execute_command") as execute, redirect_stdout(output):
                main._process_once()
            execute.assert_not_called()
            self.assertIn("Are you sure", output.getvalue())
            status, pending = main.session.resolve_pending_confirmation()
            self.assertEqual(status, "active")
            self.assertEqual(pending.intent, "shutdown_computer")
        finally:
            main.session = original_session


class TestBackwardCompatibility(unittest.TestCase):
    """Backward compatibility tests."""

    def test_correct_tokens_api(self):
        from core.natural_language import correct_tokens
        result = correct_tokens(["opn", "chrme"])
        self.assertEqual(result, ["open", "chrome"])

    def test_correct_token_api(self):
        from core.natural_language import correct_token
        result = correct_token("opn")
        self.assertEqual(result, "open")


class TestErrorHandling(unittest.TestCase):
    """Spell fallback remains independent from interpretation."""

    def setUp(self):
        from core.spell_checker import correct_tokens
        self.correct_tokens = correct_tokens

    def test_missing_model_falls_back(self):
        with patch("core.spell_model.SpellModel.load") as mock_load:
            mock_load.side_effect = FileNotFoundError("model not found")
            result = self.correct_tokens(["opn", "chrme"])
            self.assertEqual(result, ["open", "chrome"])

    def test_corrupt_model_falls_back(self):
        with patch("core.spell_model.SpellModel.load") as mock_load:
            mock_load.side_effect = ValueError("corrupt model")
            result = self.correct_tokens(["opn", "chrme"])
            self.assertEqual(result, ["open", "chrome"])

    def test_prediction_exception_falls_back(self):
        with patch("core.spell_model.SpellModel.load") as mock_load:
            mock_model = MagicMock()
            mock_model.correct.side_effect = RuntimeError("prediction failed")
            mock_load.return_value = mock_model
            result = self.correct_tokens(["opn", "chrme"])
            self.assertEqual(result, ["open", "chrome"])


class TestPerformance(unittest.TestCase):
    """Performance tests."""

    @classmethod
    def setUpClass(cls):
        cls.interpreter = NaturalLanguageInterpreter()

    def test_correction_latency(self):
        import time
        from core.spell_checker import correct_tokens

        start = time.perf_counter()
        tokens = correct_tokens(["opn", "chrme"])
        result = self.interpreter.interpret("opn chrme", tokens)
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 1.0)  # Should be under 1 second
        self.assertEqual(result.intent, "OPEN_APPLICATION")

    def test_repeated_correction_uses_cache(self):
        import time
        from core.spell_checker import correct_tokens

        tokens1 = correct_tokens(["opn", "chrme"])
        result1 = self.interpreter.interpret("opn chrme", tokens1)
        start = time.perf_counter()
        tokens2 = correct_tokens(["opn", "chrme"])
        result2 = self.interpreter.interpret("opn chrme", tokens2)
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 0.1)  # Should be very fast (cached)
        self.assertEqual(result1.intent, result2.intent)


if __name__ == "__main__":
    unittest.main()
