"""
Final Release Command Matrix (Step 24 â€” Phases 5-8).

Covers every registered command through the complete pipeline:
    input -> normalize -> tokenize -> spell -> intent -> parse
    -> validate -> execute -> response -> history

Also covers natural-input regression (Phase 6), multi-command
behavior (Phase 7), and confirmation security (Phase 8).

Safety model: the executor is mocked for every test. Dangerous
commands (shutdown/restart/delete) are verified to PARK as a pending
confirmation â€” they are never executed. No test performs a real
system operation.
"""

import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

# Make project modules importable (mirrors every other test file)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import main
from core.command_history import CommandHistory, set_command_history
from core.command_registry import get_all_commands
from core.session_state import SessionState


class _ReleaseTestCase(unittest.TestCase):
    """Fresh session + bounded history per test; executor fully mocked."""

    def setUp(self):
        self._original_session = main.session
        main.session = SessionState()
        self.history = CommandHistory(max_entries=100)
        set_command_history(self.history)
        self._buffer = io.StringIO()
        self._exec_patcher = patch(
            "main.execute_command",
            return_value={"success": True, "message": "ok (mocked)",
                          "data": {}},
        )
        self._multi_patcher = patch(
            "core.multi_command_executor.execute_command",
            return_value={"success": True, "message": "ok (mocked)",
                          "data": {}},
        )
        self._exec_patcher.start()
        self._multi_patcher.start()

    def tearDown(self):
        self._exec_patcher.stop()
        self._multi_patcher.stop()
        main.session = self._original_session
        set_command_history(None)

    def _run_turn(self, text):
        with patch("main.get_input", return_value=text):
            with redirect_stdout(self._buffer):
                main._process_once()
        return self._buffer.getvalue()

    def _has_pending(self):
        return main.session.get_pending_confirmation() is not None

    def _run_turn_real(self, text):
        """Run a turn with the REAL executor (read-only commands)."""
        self._exec_patcher.stop()
        self._multi_patcher.stop()
        try:
            return self._run_turn(text)
        finally:
            self._exec_patcher.start()
            self._multi_patcher.start()

    def _executed_commands(self):
        """Return all commands passed to either mocked executor."""
        cmds = []
        for mock in (main.execute_command,
                     sys.modules["core.multi_command_executor"].execute_command):
            cmds.extend(c.args[0] for c in mock.call_args_list if c.args)
        return cmds


# ---------------------------------------------------------------------------
# Phase 4/5 â€” Command inventory and full pipeline matrix
# ---------------------------------------------------------------------------

class TestCommandInventory(_ReleaseTestCase):
    """Every registered command is complete and detectable."""

    def test_every_command_has_required_metadata(self):
        commands = get_all_commands()
        for name, cmd in commands.items():
            with self.subTest(command=name):
                self.assertTrue(cmd.intent)
                self.assertIsInstance(cmd.aliases, tuple)
                self.assertTrue(cmd.description)
                self.assertIn(cmd.category,
                              {"application", "file", "system", "browser",
                               "help", "general"})

    def test_no_orphaned_commands(self):
        """Every non-UNKNOWN command is detectable by the intent layer."""
        from core.intent import detect_intent
        from core.spell_checker import correct_tokens
        from core.tokenizer import tokenize

        commands = get_all_commands()
        for name, cmd in commands.items():
            if name == "UNKNOWN":
                continue
            with self.subTest(command=name):
                # The primary alias must be detectable
                alias = cmd.aliases[0]
                tokens = correct_tokens(tokenize(alias))
                intent = detect_intent(tokens)
                self.assertEqual(
                    intent.value,
                    cmd.intent.lower(),
                    f"{name}: alias {alias!r} detected as "
                    f"{intent.value}, expected {cmd.intent.lower()}",
                )


class TestCommandPipelineMatrix(_ReleaseTestCase):
    """Each command flows through every pipeline stage correctly."""

    def _assert_executed(self, text, intent, target=None):
        self._buffer.seek(0)
        self._buffer.truncate(0)
        main.execute_command.reset_mock()
        sys.modules["core.multi_command_executor"].execute_command.reset_mock()
        output = self._run_turn(text)
        commands = self._executed_commands()
        self.assertEqual(len(commands), 1, f"{text!r} executed {commands}")
        cmd = commands[0]
        self.assertEqual(cmd["intent"], intent)
        if target is not None:
            self.assertEqual(cmd["target"], target)
        self.assertIn("ok (mocked)", output)
        return cmd

    # -- Application commands ------------------------------------------------
    def test_open_application(self):
        self._assert_executed("open chrome", "open_application", "chrome")

    def test_close_application(self):
        self._assert_executed("close chrome", "close_application", "chrome")

    # -- System commands -----------------------------------------------------
    def test_lock_computer(self):
        self._assert_executed("lock computer", "lock_computer")

    def test_shutdown_computer_parks(self):
        """Shutdown must park as confirmation, never execute."""
        self._buffer.seek(0)
        self._buffer.truncate(0)
        main.execute_command.reset_mock()
        output = self._run_turn("shutdown computer")
        main.execute_command.assert_not_called()
        self.assertIn("yes", output.lower())
        self.assertTrue(self._has_pending())

    def test_restart_computer_parks(self):
        self._buffer.seek(0)
        self._buffer.truncate(0)
        main.execute_command.reset_mock()
        output = self._run_turn("restart computer")
        main.execute_command.assert_not_called()
        self.assertIn("yes", output.lower())
        self.assertTrue(self._has_pending())

    # -- File commands -------------------------------------------------------
    def test_create_folder(self):
        self._assert_executed("create folder test", "create_folder", "test")

    def test_delete_file_parks(self):
        """Delete must park as confirmation, never execute."""
        self._buffer.seek(0)
        self._buffer.truncate(0)
        main.execute_command.reset_mock()
        output = self._run_turn("delete file test.txt")
        main.execute_command.assert_not_called()
        self.assertIn("yes", output.lower())
        self.assertTrue(self._has_pending())

    def test_open_file(self):
        self._assert_executed("open file report.pdf", "open_file",
                              "report.pdf")

    def test_search_file(self):
        self._assert_executed("search file report", "search_file", "report")

    # -- Browser commands ----------------------------------------------------
    def test_open_browser(self):
        self._assert_executed("open browser", "open_browser")

    def test_search_web(self):
        self._assert_executed("search web python tutorials", "search_web")

    def test_open_url(self):
        self._assert_executed("open url https://github.com", "open_url",
                              "https://github.com")

    # -- Help commands -------------------------------------------------------
    def test_help(self):
        self._buffer.seek(0)
        self._buffer.truncate(0)
        output = self._run_turn_real("help")
        self.assertIn("TITAN can help you with:", output)

    def test_list_commands(self):
        self._buffer.seek(0)
        self._buffer.truncate(0)
        output = self._run_turn_real("commands")
        self.assertIn("Available commands:", output)

    def test_history_command(self):
        self._run_turn("open chrome")
        self._buffer.seek(0)
        self._buffer.truncate(0)
        output = self._run_turn_real("history")
        self.assertIn("open chrome", output)

    def test_clear_history_command(self):
        self._run_turn("open chrome")
        self._buffer.seek(0)
        self._buffer.truncate(0)
        output = self._run_turn_real("clear history")
        self.assertIn("cleared", output.lower())

    # -- Startup commands (Step 23) ------------------------------------------
    def test_enable_startup_parks(self):
        self._buffer.seek(0)
        self._buffer.truncate(0)
        main.execute_command.reset_mock()
        output = self._run_turn("enable startup")
        main.execute_command.assert_not_called()
        self.assertIn("yes", output.lower())
        self.assertTrue(self._has_pending())

    def test_disable_startup_parks(self):
        self._buffer.seek(0)
        self._buffer.truncate(0)
        main.execute_command.reset_mock()
        output = self._run_turn("disable startup")
        main.execute_command.assert_not_called()
        self.assertIn("yes", output.lower())
        self.assertTrue(self._has_pending())

    def test_startup_status(self):
        self._buffer.seek(0)
        self._buffer.truncate(0)
        output = self._run_turn_real("startup status")
        self.assertIn("does not start", output)


# ---------------------------------------------------------------------------
# Phase 6 â€” Natural input regression
# ---------------------------------------------------------------------------

class TestNaturalInputRegression(_ReleaseTestCase):
    """Realistic user inputs and common mistakes stay deterministic."""

    def _intent_of(self, text):
        from core.intent import detect_intent
        from core.spell_checker import correct_tokens
        from core.tokenizer import tokenize
        return detect_intent(correct_tokens(tokenize(text))).value

    def test_realistic_application_inputs(self):
        cases = {
            "open chrome": "open_application",
            "open google chrome": "open_application",
            "open calc": "open_application",
            "open calculator": "open_application",
            "open notepad": "open_application",
            "open file explorer": "open_application",
            "open browser": "open_browser",
        }
        for text, expected in cases.items():
            with self.subTest(input=text):
                self.assertEqual(self._intent_of(text), expected)

    def test_realistic_browser_inputs(self):
        cases = {
            "search web python tutorials": "search_web",
            "search web cats and dogs": "search_web",
            "open https://github.com": "open_url",
        }
        for text, expected in cases.items():
            with self.subTest(input=text):
                self.assertEqual(self._intent_of(text), expected)

    def test_realistic_system_inputs(self):
        cases = {
            "lock computer": "lock_computer",
            "shutdown computer": "shutdown_computer",
            "restart computer": "restart_computer",
            "create folder test": "create_folder",
        }
        for text, expected in cases.items():
            with self.subTest(input=text):
                self.assertEqual(self._intent_of(text), expected)

    def test_realistic_help_inputs(self):
        cases = {
            "help": "help",
            "what can you do": "help",
            "commands": "list_commands",
            "history": "history",
            "clear history": "clear_history",
        }
        for text, expected in cases.items():
            with self.subTest(input=text):
                self.assertEqual(self._intent_of(text), expected)

    def test_common_misspellings(self):
        cases = {
            "opn chrome": "open_application",
            "opn chrokme": "open_application",
            "opn google chrpme": "open_application",
            "shut down computer": "shutdown_computer",
            "reboot computer": "restart_computer",
            "search the web for python tutorials": "search_web",
        }
        for text, expected in cases.items():
            with self.subTest(input=text):
                self.assertEqual(self._intent_of(text), expected)

    def test_search_web_cats_and_dogs_is_one_command(self):
        """'search web cats and dogs' must not split on 'and'."""
        self._buffer.seek(0)
        self._buffer.truncate(0)
        main.execute_command.reset_mock()
        self._run_turn("search web cats and dogs")
        commands = self._executed_commands()
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0]["intent"], "search_web")


# ---------------------------------------------------------------------------
# Phase 7 â€” Multi-command behavior
# ---------------------------------------------------------------------------

class TestMultiCommandRelease(_ReleaseTestCase):
    """Chains execute sequentially, in order, with correct history."""

    def test_two_commands_execute_in_order(self):
        self._buffer.seek(0)
        self._buffer.truncate(0)
        main.execute_command.reset_mock()
        self._run_turn("open chrome and open calculator")
        commands = self._executed_commands()
        self.assertEqual(len(commands), 2)
        self.assertEqual(commands[0]["target"], "chrome")
        self.assertEqual(commands[1]["target"], "calculator")

    def test_then_separator(self):
        self._buffer.seek(0)
        self._buffer.truncate(0)
        main.execute_command.reset_mock()
        self._run_turn("open chrome then open calculator")
        commands = self._executed_commands()
        self.assertEqual(len(commands), 2)

    def test_chain_with_search(self):
        self._buffer.seek(0)
        self._buffer.truncate(0)
        main.execute_command.reset_mock()
        self._run_turn("open chrome and search web python tutorials")
        commands = self._executed_commands()
        self.assertEqual(len(commands), 2)
        self.assertEqual(commands[0]["intent"], "open_application")
        self.assertEqual(commands[1]["intent"], "search_web")

    def test_history_records_individual_commands(self):
        self._run_turn("open chrome and open calculator")
        entries = self.history.get_history()
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0].intent, "open_application")
        self.assertEqual(entries[1].intent, "open_application")

    def test_dangerous_command_stops_chain_and_parks(self):
        """A dangerous command parks; later commands do not execute."""
        self._buffer.seek(0)
        self._buffer.truncate(0)
        main.execute_command.reset_mock()
        output = self._run_turn("open chrome and shutdown computer")
        # Only the first (safe) command executed; shutdown parked
        commands = self._executed_commands()
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0]["intent"], "open_application")
        self.assertTrue(self._has_pending())
        self.assertIn("yes", output.lower())

    def test_restart_in_chain_parks(self):
        self._buffer.seek(0)
        self._buffer.truncate(0)
        main.execute_command.reset_mock()
        self._run_turn("open chrome and restart computer")
        commands = self._executed_commands()
        self.assertEqual(len(commands), 1)
        self.assertTrue(self._has_pending())


# ---------------------------------------------------------------------------
# Phase 8 â€” Confirmation security
# ---------------------------------------------------------------------------

class TestConfirmationSecurity(_ReleaseTestCase):
    """Confirmation authorizes only the exact pending command."""

    def test_yes_authorizes_only_shutdown(self):
        self._run_turn("shutdown computer")
        self._buffer.seek(0)
        self._buffer.truncate(0)
        main.execute_command.reset_mock()
        output = self._run_turn("yes")
        commands = self._executed_commands()
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0]["intent"], "shutdown_computer")
        self.assertFalse(self._has_pending())

    def test_yes_with_extra_command_does_not_authorize_both(self):
        """'yes and delete file test.txt' must NOT authorize the shutdown.

        The input is not an exact confirmation phrase, so the pending
        confirmation is cancelled and the remainder is reprocessed as
        new commands — nothing dangerous executes.
        """
        self._run_turn("shutdown computer")
        self.assertTrue(self._has_pending())
        self._buffer.seek(0)
        self._buffer.truncate(0)
        main.execute_command.reset_mock()
        self._run_turn("yes and delete file test.txt")
        # The shutdown was NOT authorized — nothing dangerous executed
        commands = self._executed_commands()
        self.assertEqual(
            [c for c in commands if c["intent"] in {"shutdown_computer",
                                                    "restart_computer",
                                                    "delete_file"}],
            [],
        )
        # The pending confirmation was cancelled, not consumed as approval
        self.assertFalse(self._has_pending())

    def test_cancel_executes_nothing(self):
        self._run_turn("shutdown computer")
        self._buffer.seek(0)
        self._buffer.truncate(0)
        main.execute_command.reset_mock()
        output = self._run_turn("no")
        main.execute_command.assert_not_called()
        self.assertFalse(self._has_pending())
        self.assertIn("cancelled", output.lower())

    def test_unrelated_input_cancels_confirmation(self):
        self._run_turn("shutdown computer")
        self._buffer.seek(0)
        self._buffer.truncate(0)
        main.execute_command.reset_mock()
        self._run_turn("open chrome")
        # The confirmation was cancelled; open chrome executed normally
        commands = self._executed_commands()
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0]["intent"], "open_application")
        self.assertFalse(self._has_pending())

    def test_expired_confirmation_executes_nothing(self):
        import time as _time

        self._run_turn("shutdown computer")
        # Force expiry
        main.session._pending_expires_at = _time.monotonic() - 1
        self._buffer.seek(0)
        self._buffer.truncate(0)
        main.execute_command.reset_mock()
        output = self._run_turn("yes")
        main.execute_command.assert_not_called()
        self.assertIn("expired", output.lower())

    def test_yes_with_no_pending_confirmation_executes_nothing(self):
        self._buffer.seek(0)
        self._buffer.truncate(0)
        main.execute_command.reset_mock()
        self._run_turn("yes")
        main.execute_command.assert_not_called()


if __name__ == "__main__":
    unittest.main()
