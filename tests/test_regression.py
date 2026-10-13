"""
Regression tests for bugs encountered during earlier development
(Step 21 / Phase 23).

Each test pins one historical bug. If any of these behaviours break
again, a test here fails first.
"""

import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

# Add parent directory to path so we can import core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import main
from core.application_manager import get_application
from core.command_history import CommandHistory, set_command_history
from core.command_parser import parse_command
from core.command_validator import validate_command
from core.intent import detect_intent
from core.multi_command_parser import split_commands
from core.session_state import SessionState
from core.spell_checker import correct_tokens
from core.input import normalize_input
from core.tokenizer import tokenize


def _pipeline(text):
    """Run normalize -> tokenize -> spell correct -> intent -> parse."""
    tokens = tokenize(normalize_input(text))
    corrected = correct_tokens(tokens)
    intent = detect_intent(corrected)
    command = parse_command(corrected, intent)
    return corrected, intent, command


class _RegressionTestCase(unittest.TestCase):
    """Fresh session/history with a captured-stdout turn."""

    def setUp(self):
        self._original_session = main.session
        main.session = SessionState()
        self.history = CommandHistory(max_entries=50)
        set_command_history(self.history)
        self._buffer = io.StringIO()

    def tearDown(self):
        main.session = self._original_session
        set_command_history(None)

    def _run_turn(self, text):
        with patch("main.get_input", return_value=text):
            with redirect_stdout(self._buffer):
                main._process_once()
        return self._buffer.getvalue()


class TestSpellCheckerRegressions(unittest.TestCase):
    """Historical spell-correction bugs (Phases 5 and 23)."""

    def test_opn_google_chrpme_becomes_open_google_chrome(self):
        # 1. "opn google chrpme" -> open google chrome
        corrected, intent, command = _pipeline("opn google chrpme")
        self.assertEqual(corrected, ["open", "google", "chrome"])
        self.assertEqual(intent.value, "open_application")
        self.assertEqual(command["target"], "google chrome")

    def test_opn_chrokme_becomes_open_chrome(self):
        # 2. "opn chrokme" -> open chrome
        corrected, intent, command = _pipeline("opn chrokme")
        self.assertEqual(corrected, ["open", "chrome"])
        self.assertEqual(command["target"], "chrome")

    def test_open_calc_never_becomes_call(self):
        # 3. "open calc" must stay "calc" — never "call"
        corrected, _, command = _pipeline("open calc")
        self.assertEqual(corrected, ["open", "calc"])
        self.assertNotIn("call", corrected)
        self.assertEqual(command["target"], "calc")
        # And the alias resolves to the calculator application
        app = get_application(command["target"])
        self.assertIsNotNone(app)
        self.assertEqual(app.name, "calculator")

    def test_open_notepad_never_becomes_notehead(self):
        # 4. "open notepad" must stay "notepad" — never "notehead"
        corrected, _, command = _pipeline("open notepad")
        self.assertEqual(corrected, ["open", "notepad"])
        self.assertNotIn("notehead", corrected)
        self.assertEqual(command["target"], "notepad")


class TestIntentRegressions(unittest.TestCase):
    """Historical intent-detection bugs (Phases 6 and 23)."""

    def test_open_file_explorer_is_open_application(self):
        # 5. "open file explorer" -> OPEN_APPLICATION / Explorer
        _, intent, command = _pipeline("open file explorer")
        self.assertEqual(intent.value, "open_application")
        self.assertEqual(command["target"], "file explorer")
        app = get_application(command["target"])
        self.assertIsNotNone(app)
        self.assertEqual(app.name, "explorer")

    def test_open_google_chrome_is_not_open_file(self):
        # 6. "open google chrome" -> OPEN_APPLICATION, not OPEN_FILE
        _, intent, command = _pipeline("open google chrome")
        self.assertEqual(intent.value, "open_application")
        self.assertNotEqual(intent.value, "open_url")
        self.assertEqual(command["target"], "google chrome")

    def test_open_browser_is_open_browser(self):
        # 7. "open browser" -> OPEN_BROWSER (not an application)
        _, intent, command = _pipeline("open browser")
        self.assertEqual(intent.value, "open_browser")
        self.assertIsNone(command["target"])

    def test_search_web_cats_and_dogs_stays_one_command(self):
        # 8. "search web cats and dogs" -> ONE command
        segments = split_commands(normalize_input("search web cats and dogs"))
        self.assertEqual(len(segments), 1)
        _, intent, command = _pipeline("search web cats and dogs")
        self.assertEqual(intent.value, "search_web")
        # The query keeps its words (the known "cats"->"cat" quirk is
        # out of scope — the point is that "and" never splits it).
        self.assertIn("and", command["target"])

    def test_search_the_web_for_variant_stays_one_command(self):
        segments = split_commands(
            normalize_input("search the web for cats and dogs")
        )
        self.assertEqual(len(segments), 1)

    def test_create_folder_rock_and_roll_stays_one_command(self):
        segments = split_commands(
            normalize_input("create folder rock and roll")
        )
        self.assertEqual(len(segments), 1)


class TestConfirmationRegressions(_RegressionTestCase):
    """Chain/confirmation boundary bugs (Phases 14, 15, 23)."""

    def test_open_chrome_and_shutdown_stops_at_confirmation(self):
        # 9. chrome executes, shutdown becomes pending, chain stops
        with patch("core.multi_command_executor.execute_command") as mock_exec:
            mock_exec.return_value = {
                "success": True, "message": "Chrome opened.", "data": {},
            }
            output = self._run_turn("open chrome and shutdown computer")
        self.assertEqual(mock_exec.call_count, 1)
        self.assertEqual(mock_exec.call_args.args[0]["target"], "chrome")
        status, pending = main.session.resolve_pending_confirmation()
        self.assertEqual(status, "active")
        self.assertEqual(pending.intent, "shutdown_computer")
        self.assertIn("Are you sure you want to shut down", output)

    def test_yes_executes_only_the_pending_shutdown(self):
        # 10. "yes" -> executes only the pending shutdown command
        with patch("core.multi_command_executor.execute_command") as mock_exec:
            mock_exec.return_value = {
                "success": True, "message": "Chrome opened.", "data": {},
            }
            self._run_turn("open chrome and shutdown computer")
        with patch("main.execute_command") as mock_execute:
            mock_execute.return_value = {
                "success": True, "message": "Shut down (mocked).", "data": {},
            }
            self._run_turn("yes")
        mock_execute.assert_called_once()
        self.assertEqual(
            mock_execute.call_args.args[0]["intent"], "shutdown_computer"
        )
        self.assertEqual(mock_execute.call_args.kwargs, {"confirmed": True})


class TestHistoryRegressions(_RegressionTestCase):
    """History recording/exemption bugs (Phase 23)."""

    def test_history_command_adds_no_history_entry(self):
        # 11. "history" displays history without creating an entry
        with patch("main.execute_command"):
            self._run_turn("help")
        self.assertEqual(self.history.count(), 1)
        output = self._run_turn("history")
        self.assertIn("1. help", output)
        self.assertEqual(self.history.count(), 1)
        self.assertEqual(self.history.get_history()[-1].intent, "help")

    def test_clear_history_command_adds_no_history_entry(self):
        # 12. "clear history" clears and records nothing about itself
        with patch("main.execute_command"):
            self._run_turn("help")
        self.assertEqual(self.history.count(), 1)
        output = self._run_turn("clear history")
        self.assertIn("Command history cleared.", output)
        self.assertEqual(self.history.count(), 0)
        # Follow-up read shows the documented empty message, not "[]"
        output = self._run_turn("history")
        self.assertIn("There is no command history yet.", output)
        self.assertEqual(self.history.count(), 0)

    def test_delete_history_alias_clears_history(self):
        """'delete history' is an alias of CLEAR_HISTORY (Phase 16)."""
        with patch("main.execute_command"):
            self._run_turn("help")
        self.assertEqual(self.history.count(), 1)
        _, intent, _ = _pipeline("delete history")
        self.assertEqual(intent.value, "clear_history")
        self._run_turn("delete history")
        self.assertEqual(self.history.count(), 0)


class TestConfigurationRegressions(_RegressionTestCase):
    """Malformed configuration must never weaken safety (Phase 23)."""

    def test_malformed_config_keeps_safety_defaults_enabled(self):
        # 13. invalid/malformed configuration -> safety stays on
        def junk(*args, **kwargs):
            return {"broken": True}

        with patch("core.config_manager.get_value", side_effect=junk), \
                patch("core.session_state.get_value", side_effect=junk), \
                patch("main.execute_command") as mock_execute:
            output = self._run_turn("shutdown computer")
        # Confirmation is still required; nothing executed
        mock_execute.assert_not_called()
        status, _ = main.session.resolve_pending_confirmation()
        self.assertEqual(status, "active")
        self.assertIn("Are you sure", output)

    def test_junk_timeout_value_falls_back_to_default(self):
        from core.session_state import _confirmation_timeout_seconds

        for junk in ("sixty", None, 0, -1, True, 2.5):
            with self.subTest(junk=junk):
                with patch(
                    "core.session_state.get_value", return_value=junk
                ):
                    self.assertEqual(_confirmation_timeout_seconds(), 60.0)


class TestValidatorRegressions(unittest.TestCase):
    """Required-target and UNKNOWN handling stays intact."""

    def test_unknown_is_not_executable(self):
        _, intent, command = _pipeline("frobnicate the wibble")
        self.assertEqual(intent.value, "unknown")
        result = validate_command(command)
        self.assertFalse(result["valid"])

    def test_required_target_commands_still_need_targets(self):
        result = validate_command(
            {"intent": "open_application", "target": None, "arguments": {}}
        )
        self.assertFalse(result["valid"])


if __name__ == "__main__":
    unittest.main()
