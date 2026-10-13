"""
Confirmation-flow integration tests (Step 21 / Phase 14).

Drives the complete round-trip through main._process_once():

    "shutdown computer" -> recognized, validated, PARKED (nothing
                           executes), prompt shown
    "yes"               -> ONLY the stored command executes
    "no"                -> cancelled, nothing executes

Critical security guarantee: a pending confirmation can authorize
exactly the stored command and nothing else. Complex input such as
"yes and delete file test.txt" is never treated as authorization.
"""

import io
import os
import sys
import time
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

# Add parent directory to path so we can import core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import main
from core.command_history import CommandHistory, set_command_history
from core.session_state import (
    POSITIVE_CONFIRMATION_PHRASES,
    NEGATIVE_CONFIRMATION_PHRASES,
    SessionState,
)


class _ConfirmationTestCase(unittest.TestCase):
    """Fresh session/history per test, captured stdout, mocked effects."""

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

    def _park_shutdown(self):
        """Run the dangerous turn; assert nothing executed."""
        with patch("main.execute_command") as mock_execute:
            output = self._run_turn("shutdown computer")
        mock_execute.assert_not_called()
        status, pending = main.session.resolve_pending_confirmation()
        self.assertEqual(status, "active")
        self.assertEqual(pending.intent, "shutdown_computer")
        return output


class TestParkingAndPrompts(_ConfirmationTestCase):

    def test_shutdown_parks_and_prompts_without_executing(self):
        output = self._park_shutdown()
        self.assertIn(
            "Are you sure you want to shut down the computer?", output
        )
        # The parked history entry never claims success
        entries = self.history.get_history()
        self.assertEqual(len(entries), 1)
        self.assertIsNone(entries[0].success)

    def test_restart_parks_with_shared_prompt_wording(self):
        with patch("main.execute_command") as mock_execute:
            output = self._run_turn("restart computer")
        mock_execute.assert_not_called()
        self.assertIn(
            "Are you sure you want to restart the computer?", output
        )
        status, _ = main.session.resolve_pending_confirmation()
        self.assertEqual(status, "active")

    def test_delete_file_parks_with_target_in_prompt(self):
        with patch("main.execute_command") as mock_execute:
            output = self._run_turn("delete file test.txt")
        mock_execute.assert_not_called()
        self.assertIn("delete 'test.txt'", output)
        status, pending = main.session.resolve_pending_confirmation()
        self.assertEqual(status, "active")
        self.assertEqual(pending.command["target"], "test.txt")

    def test_empty_turn_does_not_cancel_pending(self):
        self._park_shutdown()
        output = self._run_turn("")
        self.assertIn("No command entered.", output)
        status, _ = main.session.resolve_pending_confirmation()
        self.assertEqual(status, "active")


class TestPositiveAnswers(_ConfirmationTestCase):
    """Every positive phrase executes the stored command only."""

    def test_all_positive_phrases_execute_stored_command(self):
        for phrase in sorted(POSITIVE_CONFIRMATION_PHRASES):
            with self.subTest(phrase=phrase):
                self._park_shutdown()
                with patch("main.execute_command") as mock_execute:
                    mock_execute.return_value = {
                        "success": True,
                        "message": "System shutting down (mocked).",
                        "data": {},
                    }
                    output = self._run_turn(phrase)
                mock_execute.assert_called_once()
                command = mock_execute.call_args.args[0]
                self.assertEqual(command["intent"], "shutdown_computer")
                self.assertEqual(mock_execute.call_args.kwargs,
                                 {"confirmed": True})
                self.assertIn("System shutting down (mocked).", output)
                # One-shot: pending cleared after the answer
                self.assertEqual(
                    main.session.resolve_pending_confirmation(),
                    ("none", None),
                )
                # Parked history entry resolved to the real outcome
                entries = self.history.get_history()
                self.assertTrue(entries[-1].success)
                # Reset for the next phrase
                main.session = SessionState()
                self.history = CommandHistory(max_entries=50)
                set_command_history(self.history)

    def test_history_entry_resolves_to_real_outcome(self):
        self._park_shutdown()
        with patch("main.execute_command") as mock_execute:
            mock_execute.return_value = {
                "success": True, "message": "Mocked.", "data": {},
            }
            self._run_turn("yes")
        entry = self.history.get_history()[0]
        self.assertTrue(entry.success)
        self.assertEqual(entry.message, "Mocked.")

    def test_failed_execution_resolves_to_failure(self):
        self._park_shutdown()
        with patch("main.execute_command") as mock_execute:
            mock_execute.return_value = {
                "success": False, "message": "Could not shut down.", "data": {},
            }
            self._run_turn("yes")
        entry = self.history.get_history()[0]
        self.assertFalse(entry.success)
        self.assertEqual(entry.message, "Could not shut down.")


class TestNegativeAnswers(_ConfirmationTestCase):
    """Every negative phrase cancels; nothing executes."""

    def test_all_negative_phrases_cancel(self):
        for phrase in sorted(NEGATIVE_CONFIRMATION_PHRASES):
            with self.subTest(phrase=phrase):
                self._park_shutdown()
                with patch("main.execute_command") as mock_execute:
                    output = self._run_turn(phrase)
                mock_execute.assert_not_called()
                self.assertEqual(
                    main.session.resolve_pending_confirmation(),
                    ("none", None),
                )
                # History entry resolved to a real failure outcome
                entries = self.history.get_history()
                self.assertFalse(entries[-1].success)
                # Reset for the next phrase
                main.session = SessionState()
                self.history = CommandHistory(max_entries=50)
                set_command_history(self.history)

    def test_cancel_message_is_intent_specific(self):
        self._park_shutdown()
        with patch("main.execute_command"):
            output = self._run_turn("no")
        self.assertIn("Shutdown cancelled.", output)


class TestTimeoutAndExpiry(_ConfirmationTestCase):

    def test_expired_confirmation_is_not_executed(self):
        self._park_shutdown()
        main.session._pending_expires_at = time.monotonic() - 1
        with patch("main.execute_command") as mock_execute:
            output = self._run_turn("yes")
        mock_execute.assert_not_called()
        self.assertIn("expired", output)
        self.assertEqual(
            main.session.resolve_pending_confirmation(), ("none", None)
        )
        # The parked history entry resolves to failure, never success
        self.assertFalse(self.history.get_history()[-1].success)

    def test_expired_confirmation_then_unrelated_input(self):
        self._park_shutdown()
        main.session._pending_expires_at = time.monotonic() - 1
        with patch("main.execute_command") as mock_execute:
            mock_execute.return_value = {
                "success": True, "message": "Chrome opened.", "data": {},
            }
            output = self._run_turn("open chrome")
        # Unrelated input after expiry cancels safely and processes
        # the new command — it is never treated as an answer.
        self.assertEqual(mock_execute.call_count, 1)
        self.assertEqual(
            mock_execute.call_args.args[0]["target"], "chrome"
        )
        self.assertIn("Chrome opened.", output)


class TestUnrelatedInput(_ConfirmationTestCase):

    def test_unrelated_command_cancels_and_processes_normally(self):
        self._park_shutdown()
        with patch("main.execute_command") as mock_execute:
            mock_execute.return_value = {
                "success": True, "message": "Chrome opened.", "data": {},
            }
            output = self._run_turn("open chrome")
        # Only the new command executed — the parked shutdown never did
        self.assertEqual(mock_execute.call_count, 1)
        self.assertEqual(
            mock_execute.call_args.args[0]["target"], "chrome"
        )
        self.assertEqual(
            main.session.resolve_pending_confirmation(), ("none", None)
        )
        self.assertIn("Chrome opened.", output)

    def test_partial_match_is_not_an_answer(self):
        """'yes please' is not an exact confirmation phrase."""
        self._park_shutdown()
        with patch("main.execute_command") as mock_execute:
            self._run_turn("yes please")
        mock_execute.assert_not_called()
        # Pending confirmation was cancelled by unrelated input
        self.assertEqual(
            main.session.resolve_pending_confirmation(), ("none", None)
        )


class TestCriticalBypassAttempts(_ConfirmationTestCase):
    """Security regression tests for confirmation authorization."""

    def test_yes_with_extra_text_is_not_authorization(self):
        """Pending shutdown + 'yes and delete file test.txt' must NOT
        authorize either operation.

        The input is not an exact confirmation phrase, so the pending
        confirmation is cancelled; the remainder fails whole-chain
        pre-validation ('yes' is not a valid command), so nothing at
        all executes — in particular, no file is deleted.
        """
        self._park_shutdown()
        with patch("main.execute_command") as mock_execute, \
                patch("core.multi_command_executor.execute_command") as mock_multi, \
                patch("core.executor.delete_file") as mock_delete:
            output = self._run_turn("yes and delete file test.txt")

        mock_execute.assert_not_called()
        mock_multi.assert_not_called()
        mock_delete.assert_not_called()
        # Pending confirmation cancelled, not consumed as approval
        self.assertEqual(
            main.session.resolve_pending_confirmation(), ("none", None)
        )
        # The turn still received a user-facing response
        self.assertTrue(output.strip())

    def test_confirmation_cannot_be_reused(self):
        self._park_shutdown()
        with patch("main.execute_command") as mock_execute:
            mock_execute.return_value = {
                "success": True, "message": "Mocked.", "data": {},
            }
            self._run_turn("yes")
            # Second "yes" has nothing to confirm
            output = self._run_turn("yes")
        self.assertEqual(mock_execute.call_count, 1)
        self.assertIn("There is nothing to confirm.", output)

    def test_yes_without_pending_never_executes(self):
        with patch("main.execute_command") as mock_execute:
            output = self._run_turn("yes")
        mock_execute.assert_not_called()
        self.assertIn("There is nothing to confirm.", output)

    def test_confirmation_executes_stored_command_not_new_input(self):
        """The answer 'yes' executes the stored command object, with
        its original target — not anything derived from 'yes'."""
        self._park_shutdown()
        stored = main.session.snapshot()["pending_confirmation"]["command"]
        with patch("main.execute_command") as mock_execute:
            mock_execute.return_value = {
                "success": True, "message": "Mocked.", "data": {},
            }
            self._run_turn("y")
        mock_execute.assert_called_once()
        self.assertEqual(mock_execute.call_args.args[0], stored)
        self.assertEqual(mock_execute.call_args.args[0]["intent"],
                         "shutdown_computer")


if __name__ == "__main__":
    unittest.main()
