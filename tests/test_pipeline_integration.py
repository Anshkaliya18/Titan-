"""
End-to-end pipeline integration tests (Step 21).

Each test drives the real pipeline (normalization, tokenization,
spell correction, intent detection, parsing, validation, execution
routing, response formatting, session state, and history) while the
actual effectful leaf (the executor's system calls) is mocked. No
application is launched, no browser is opened, and no system state
changes during these tests.
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
from core.command_history import CommandHistory, set_command_history
from core.session_state import SessionState


class _PipelineTestCase(unittest.TestCase):
    """Shared harness: fresh session/history per test, captured stdout."""

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
        """Run one full user turn through main._process_once()."""
        with patch("main.get_input", return_value=text):
            with redirect_stdout(self._buffer):
                main._process_once()
        return self._buffer.getvalue()


class TestSingleCommandPipeline(_PipelineTestCase):
    """Input flows through every stage to the executor."""

    def _success_result(self, message="Done."):
        return {"success": True, "message": message, "data": {}}

    def test_misspelled_input_reaches_every_stage(self):
        with patch("main.execute_command") as mock_execute:
            mock_execute.return_value = self._success_result("Chrome opened.")
            output = self._run_turn("opn google chrpme")

        # Stage: execution received the fully corrected, parsed command
        mock_execute.assert_called_once()
        command = mock_execute.call_args.args[0]
        self.assertEqual(
            command,
            {"intent": "open_application", "target": "google chrome",
             "arguments": {}},
        )

        # Stage records in session state: raw tokens before correction,
        # corrected tokens after, intent, and the parsed command.
        snap = main.session.snapshot()
        self.assertEqual(snap["last_input"], "opn google chrpme")
        self.assertEqual(snap["last_tokens"], ["opn", "google", "chrpme"])
        self.assertEqual(
            snap["last_corrected_tokens"], ["open", "google", "chrome"]
        )
        self.assertEqual(snap["last_intent"], "open_application")
        self.assertEqual(snap["last_command"]["target"], "google chrome")

        # History recorded the attempt
        entries = self.history.get_history()
        self.assertEqual(len(entries), 1)
        self.assertTrue(entries[0].success)
        self.assertEqual(entries[0].intent, "open_application")

        # Response reached the user
        self.assertIn("Chrome opened.", output)

    def test_normalized_whitespace_input(self):
        with patch("main.execute_command") as mock_execute:
            mock_execute.return_value = self._success_result()
            self._run_turn("  OPEN   CHROME  ")
        self.assertEqual(
            mock_execute.call_args.args[0]["target"], "chrome"
        )

    def test_validation_failure_never_reaches_executor(self):
        with patch("main.execute_command") as mock_execute:
            output = self._run_turn("blah blah")
        mock_execute.assert_not_called()
        # A failed attempt is still recorded in history
        entries = self.history.get_history()
        self.assertEqual(len(entries), 1)
        self.assertFalse(entries[0].success)
        self.assertIn("Unknown command.", output)

    def test_response_comes_from_result_message(self):
        with patch("main.execute_command") as mock_execute:
            mock_execute.return_value = {
                "success": True,
                "message": "Calculator opened successfully.",
                "data": {},
            }
            output = self._run_turn("open calculator")
        self.assertIn("Calculator opened successfully.", output)
        self.assertEqual(
            mock_execute.call_args.args[0]["target"], "calculator"
        )

    def test_unknown_intent_is_recorded_but_not_executed(self):
        with patch("main.execute_command") as mock_execute:
            self._run_turn("frobnicate the wibble")
        mock_execute.assert_not_called()
        self.assertEqual(self.history.count(), 1)
        self.assertFalse(self.history.get_history()[0].success)


class TestMultiCommandPipeline(_PipelineTestCase):
    """One input, several commands: order, stop rules, history."""

    def test_chain_executes_in_order(self):
        with patch("core.multi_command_executor.execute_command") as mock_exec:
            mock_exec.return_value = {
                "success": True, "message": "ok", "data": {},
            }
            output = self._run_turn("open chrome and open calculator")

        self.assertEqual(mock_exec.call_count, 2)
        self.assertEqual(
            mock_exec.call_args_list[0].args[0]["target"], "chrome"
        )
        self.assertEqual(
            mock_exec.call_args_list[1].args[0]["target"], "calculator"
        )
        # Both responses appear, in execution order
        lines = [line for line in output.splitlines() if line.strip()]
        self.assertIn("ok", lines)

        # History: one entry per command, sharing the full input text
        entries = self.history.get_history()
        self.assertEqual(len(entries), 2)
        self.assertEqual(
            {e.input_text for e in entries},
            {"open chrome and open calculator"},
        )
        self.assertNotEqual(entries[0].id, entries[1].id)

    def test_chain_stops_at_first_failure(self):
        with patch("core.multi_command_executor.execute_command") as mock_exec:
            mock_exec.return_value = {
                "success": False, "message": "First failed.", "data": {},
            }
            self._run_turn("open chrome and open calculator")
        # Pre-validation passes for both, execution stops after the
        # first failure — the second command never runs.
        self.assertEqual(mock_exec.call_count, 1)

    def test_invalid_chain_executes_nothing(self):
        with patch("core.multi_command_executor.execute_command") as mock_exec:
            self._run_turn("blah blah and open chrome")
        # Whole-chain pre-validation fails on the unknown command, so
        # even the valid command never executes.
        mock_exec.assert_not_called()

    def test_dangerous_command_stops_chain_and_parks(self):
        with patch("core.multi_command_executor.execute_command") as mock_exec:
            mock_exec.return_value = {
                "success": True, "message": "Chrome opened.", "data": {},
            }
            output = self._run_turn("open chrome and shutdown computer")

        # The safe command executed; the dangerous one parked.
        self.assertEqual(mock_exec.call_count, 1)
        self.assertEqual(
            mock_exec.call_args.args[0]["target"], "chrome"
        )
        status, pending = main.session.resolve_pending_confirmation()
        self.assertEqual(status, "active")
        self.assertEqual(pending.intent, "shutdown_computer")

        # History reflects both: resolved success + awaiting outcome
        entries = self.history.get_history()
        self.assertEqual(len(entries), 2)
        self.assertTrue(entries[0].success)
        self.assertIsNone(entries[1].success)
        self.assertIn("Are you sure you want to shut down", output)


class TestMainLoopIntegration(_PipelineTestCase):
    """The main loop keeps running after input and exits cleanly."""

    def test_main_loop_processes_turn_then_exits(self):
        with patch("main.get_input") as mock_input:
            mock_input.side_effect = [
                "opn google chrpme",
                KeyboardInterrupt,
            ]
            with patch("main.execute_command") as mock_execute:
                mock_execute.return_value = {
                    "success": True, "message": "Chrome opened.", "data": {},
                }
                with redirect_stdout(self._buffer):
                    main.main()

        output = self._buffer.getvalue()
        self.assertIn("Chrome opened.", output)
        self.assertIn("Goodbye!", output)
        self.assertEqual(mock_execute.call_count, 1)
        self.assertEqual(self.history.count(), 1)

    def test_bad_input_does_not_crash_the_loop(self):
        """An exception in one turn is answered safely, then the loop
        would continue (we stop it with KeyboardInterrupt)."""
        with patch("main.get_input") as mock_input:
            mock_input.side_effect = [
                RuntimeError("boom"),
                KeyboardInterrupt,
            ]
            with redirect_stdout(self._buffer):
                main.main()
        output = self._buffer.getvalue()
        self.assertIn("Something went wrong.", output)
        self.assertIn("Goodbye!", output)


if __name__ == "__main__":
    unittest.main()
