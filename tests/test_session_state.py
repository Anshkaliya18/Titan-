"""
Session State Tests (Step 18)

Covers the in-memory SessionState: initialization, per-turn updates,
snapshot isolation, clearing/resetting, pending-confirmation safety
and expiration, plus main.py's session-level confirmation round-trip.

Safety rules verified here:
- "yes" executes nothing unless a pending confirmation exists.
- Only the EXACT stored command runs, and only after revalidation.
- Expired confirmations can never execute.
- Unrelated input cancels the pending confirmation, never answers it.
- Attack input ("yes && delete ...") is never interpreted as a
  shell command — there is no shell interpretation anywhere.
- Session state is never written to disk.

No real dangerous operation runs: the executor is mocked throughout.
"""

import io
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import core.session_state as ss
from core.session_state import (
    DEFAULT_CONFIRMATION_TIMEOUT_SECONDS,
    NEGATIVE_CONFIRMATION_PHRASES,
    POSITIVE_CONFIRMATION_PHRASES,
    SessionState,
    classify_confirmation_input,
)

EXPECTED_STATE_KEYS = {
    "session_id",
    "started_at",
    "last_input",
    "last_tokens",
    "last_corrected_tokens",
    "last_intent",
    "last_command",
    "last_commands",
    "last_result",
    "last_results",
    "last_response",
    "pending_confirmation",
    "context",
}

SHUTDOWN_COMMAND = {
    "intent": "shutdown_computer",
    "target": None,
    "arguments": {},
}


class TestSessionInitialization(unittest.TestCase):
    """A fresh session starts empty with a unique, aware identity."""

    def test_session_id_exists(self):
        session = SessionState()
        self.assertIsInstance(session.session_id, str)
        self.assertTrue(session.session_id)

    def test_session_ids_are_unique(self):
        self.assertNotEqual(
            SessionState().session_id, SessionState().session_id
        )

    def test_started_at_exists_and_is_timezone_aware(self):
        started = SessionState().started_at
        self.assertIsInstance(started, datetime)
        self.assertIsNotNone(started.tzinfo)
        self.assertEqual(started.utcoffset(), timezone.utc.utcoffset(None))

    def test_state_starts_empty(self):
        snap = SessionState().snapshot()
        self.assertEqual(set(snap.keys()), EXPECTED_STATE_KEYS)
        for key in EXPECTED_STATE_KEYS - {"session_id", "started_at", "context"}:
            self.assertIsNone(snap[key], f"{key} should start None")
        self.assertEqual(snap["context"], [])

    def test_pending_confirmation_starts_none(self):
        session = SessionState()
        self.assertIsNone(session.get_pending_confirmation())
        self.assertEqual(
            session.resolve_pending_confirmation(), ("none", None)
        )


class TestStateUpdates(unittest.TestCase):
    """Every per-turn field can be recorded and read back."""

    def test_setters_update_values(self):
        session = SessionState()
        session.set_last_input("open chrome")
        session.set_last_tokens(["open", "chrome"])
        session.set_last_corrected_tokens(["open", "chrome"])
        session.set_last_intent("open_application")
        session.set_last_command(
            {"intent": "open_application", "target": "chrome", "arguments": {}}
        )
        session.set_last_result(
            {"success": True, "message": "ok", "data": {}}
        )
        session.set_last_response("Chrome opened successfully.")

        snap = session.snapshot()
        self.assertEqual(snap["last_input"], "open chrome")
        self.assertEqual(snap["last_tokens"], ["open", "chrome"])
        self.assertEqual(snap["last_corrected_tokens"], ["open", "chrome"])
        self.assertEqual(snap["last_intent"], "open_application")
        self.assertEqual(snap["last_command"]["target"], "chrome")
        self.assertTrue(snap["last_result"]["success"])
        self.assertEqual(snap["last_response"], "Chrome opened successfully.")

    def test_set_last_tokens_copies_the_list(self):
        session = SessionState()
        tokens = ["open", "chrome"]
        session.set_last_tokens(tokens)
        tokens.append("hacked")
        self.assertEqual(session.get("last_tokens"), ["open", "chrome"])

    def test_set_last_command_deep_copies(self):
        session = SessionState()
        command = {"intent": "shutdown_computer", "arguments": {"a": [1]}}
        session.set_last_command(command)
        command["arguments"]["a"].append(2)
        self.assertEqual(
            session.get("last_command")["arguments"], {"a": [1]}
        )

    def test_set_last_commands_stores_the_ordered_list(self):
        """Multi-command turns record every command in input order."""
        session = SessionState()
        commands = [
            {"intent": "open_application", "target": "chrome"},
            {"intent": "search_web", "target": "python tutorials"},
        ]
        session.set_last_commands(commands)
        recorded = session.get("last_commands")
        self.assertEqual(len(recorded), 2)
        self.assertEqual(recorded[0]["target"], "chrome")
        self.assertEqual(recorded[1]["target"], "python tutorials")

    def test_set_last_commands_deep_copies(self):
        session = SessionState()
        commands = [{"intent": "help", "arguments": {"nested": [1]}}]
        session.set_last_commands(commands)
        commands[0]["arguments"]["nested"].append(2)
        self.assertEqual(
            session.get("last_commands")[0]["arguments"], {"nested": [1]}
        )

    def test_set_last_results_stores_entries_in_order(self):
        """Per-command results are recorded in execution order."""
        session = SessionState()
        results = [
            {"command": "open chrome", "success": True, "message": "A.",
             "data": {}},
            {"command": "open calc", "success": True, "message": "B.",
             "data": {}},
        ]
        session.set_last_results(results)
        recorded = session.get("last_results")
        self.assertEqual([r["message"] for r in recorded], ["A.", "B."])

    def test_set_last_results_deep_copies(self):
        session = SessionState()
        results = [{"command": "x", "success": True, "message": "m",
                    "data": {"n": [1]}}]
        session.set_last_results(results)
        results[0]["data"]["n"].append(2)
        self.assertEqual(
            session.get("last_results")[0]["data"], {"n": [1]}
        )

    def test_set_last_commands_and_results_accept_none(self):
        session = SessionState()
        session.set_last_commands([{"intent": "help"}])
        session.set_last_results([{"success": True}])
        session.set_last_commands(None)
        session.set_last_results(None)
        self.assertIsNone(session.get("last_commands"))
        self.assertIsNone(session.get("last_results"))

    def test_get_returns_a_copy_of_mutable_values(self):
        session = SessionState()
        session.set_last_result(
            {"success": False, "message": "m", "data": {}}
        )
        session.get("last_result")["success"] = True
        self.assertFalse(session.get("last_result")["success"])

    def test_get_unknown_key_returns_default(self):
        self.assertIsNone(SessionState().get("nope"))
        self.assertEqual(SessionState().get("nope", 5), 5)

    def test_set_unknown_key_raises(self):
        with self.assertRaises(KeyError):
            SessionState().set("not_a_field", 1)

    def test_set_session_id_and_started_at_rejected(self):
        session = SessionState()
        with self.assertRaises(ValueError):
            session.set("session_id", "forged")
        with self.assertRaises(ValueError):
            session.set("started_at", "forged")
        # Values unchanged
        self.assertNotEqual(session.session_id, "forged")

    def test_set_pending_confirmation_directly_rejected(self):
        """Direct writes would bypass the expiry bookkeeping."""
        with self.assertRaises(ValueError):
            SessionState().set("pending_confirmation", "whatever")


class TestSnapshot(unittest.TestCase):
    """snapshot() is safe to inspect and impossible to mutate through."""

    def test_snapshot_contains_expected_fields(self):
        self.assertEqual(
            set(SessionState().snapshot().keys()), EXPECTED_STATE_KEYS
        )

    def test_snapshot_is_a_copy(self):
        session = SessionState()
        session.set_last_tokens(["a"])
        session.set_last_command(
            {"intent": "help", "arguments": {"nested": ["x"]}}
        )
        snap = session.snapshot()
        snap["last_tokens"].append("b")
        snap["last_command"]["arguments"]["nested"].append("y")
        snap["last_intent"] = "forged"

        internal = session.snapshot()
        self.assertEqual(internal["last_tokens"], ["a"])
        self.assertEqual(
            internal["last_command"]["arguments"], {"nested": ["x"]}
        )
        self.assertIsNone(internal["last_intent"])

    def test_snapshot_renders_timestamps_as_strings(self):
        session = SessionState()
        session.set_pending_confirmation(SHUTDOWN_COMMAND)
        snap = session.snapshot()
        self.assertIsInstance(snap["started_at"], str)
        self.assertIsInstance(snap["pending_confirmation"]["created_at"], str)

    def test_snapshot_pending_structure(self):
        session = SessionState()
        session.set_pending_confirmation(SHUTDOWN_COMMAND)
        pending = session.snapshot()["pending_confirmation"]
        self.assertEqual(pending["intent"], "shutdown_computer")
        self.assertEqual(pending["command"], SHUTDOWN_COMMAND)


class TestClearAndReset(unittest.TestCase):
    """clear() keeps identity; reset_session() makes a new one."""

    def _populated(self):
        session = SessionState()
        session.set_last_input("shutdown computer")
        session.set_last_tokens(["shutdown", "computer"])
        session.set_last_corrected_tokens(["shutdown", "computer"])
        session.set_last_intent("shutdown_computer")
        session.set_last_command(SHUTDOWN_COMMAND)
        session.set_last_result({"success": True, "message": "m", "data": {}})
        session.set_last_response("r")
        session.set_pending_confirmation(SHUTDOWN_COMMAND)
        return session

    def test_clear_resets_command_state(self):
        session = self._populated()
        session.clear()
        snap = session.snapshot()
        for key in EXPECTED_STATE_KEYS - {"session_id", "started_at", "context"}:
            self.assertIsNone(snap[key], f"{key} should be cleared")
        self.assertEqual(snap["context"], [])

    def test_clear_preserves_identity(self):
        session = self._populated()
        session_id = session.session_id
        started_at = session.started_at
        session.clear()
        self.assertEqual(session.session_id, session_id)
        self.assertEqual(session.started_at, started_at)

    def test_clear_removes_pending_confirmation(self):
        session = self._populated()
        session.clear()
        self.assertEqual(
            session.resolve_pending_confirmation(), ("none", None)
        )

    def test_reset_session_creates_new_identity(self):
        session = self._populated()
        old_id = session.session_id
        old_started = session.started_at
        session.reset_session()
        self.assertNotEqual(session.session_id, old_id)
        self.assertGreaterEqual(session.started_at, old_started)

    def test_reset_session_clears_command_state(self):
        session = self._populated()
        session.reset_session()
        snap = session.snapshot()
        for key in EXPECTED_STATE_KEYS - {"session_id", "started_at", "context"}:
            self.assertIsNone(snap[key], f"{key} should be None after reset")
        self.assertEqual(snap["context"], [])


class TestConfirmationClassification(unittest.TestCase):
    """Exact, deterministic phrase matching — no fuzzy logic."""

    def test_positive_phrases(self):
        for phrase in POSITIVE_CONFIRMATION_PHRASES:
            with self.subTest(phrase=phrase):
                self.assertEqual(
                    classify_confirmation_input(phrase), "positive"
                )

    def test_negative_phrases(self):
        for phrase in NEGATIVE_CONFIRMATION_PHRASES:
            with self.subTest(phrase=phrase):
                self.assertEqual(
                    classify_confirmation_input(phrase), "negative"
                )

    def test_case_and_whitespace_normalized(self):
        self.assertEqual(classify_confirmation_input("  YES  "), "positive")
        self.assertEqual(classify_confirmation_input("Do It"), "positive")
        self.assertEqual(classify_confirmation_input("CANCEL IT"), "negative")

    def test_complex_input_is_not_a_confirmation(self):
        # PART 15: "yes please open chrome" must NOT count as an answer
        for text in (
            "yes please open chrome",
            "open chrome",
            "yess",
            "nope",
            "confirm shutdown and delete files",
            "",
        ):
            with self.subTest(text=text):
                self.assertIsNone(classify_confirmation_input(text))

    def test_non_string_input_is_not_a_confirmation(self):
        self.assertIsNone(classify_confirmation_input(None))
        self.assertIsNone(classify_confirmation_input(42))


class TestPendingConfirmation(unittest.TestCase):
    """The pending confirmation stores an exact, private command."""

    def test_set_pending_stores_lowercased_intent(self):
        session = SessionState()
        pending = session.set_pending_confirmation(
            {"intent": "SHUTDOWN_COMPUTER", "target": None, "arguments": {}}
        )
        self.assertEqual(pending.intent, "shutdown_computer")

    def test_set_pending_deep_copies_the_command(self):
        """Later mutations of the caller's dict cannot alter what runs."""
        session = SessionState()
        command = dict(SHUTDOWN_COMMAND, arguments={"depth": ["original"]})
        session.set_pending_confirmation(command)
        command["target"] = "forged"
        command["arguments"]["depth"].append("tampered")
        stored = session.get_pending_confirmation().command
        self.assertIsNone(stored["target"])
        self.assertEqual(stored["arguments"], {"depth": ["original"]})

    def test_get_pending_returns_the_confirmation(self):
        session = SessionState()
        session.set_pending_confirmation(SHUTDOWN_COMMAND)
        pending = session.get_pending_confirmation()
        self.assertEqual(pending.intent, "shutdown_computer")
        status, resolved = session.resolve_pending_confirmation()
        self.assertEqual(status, "active")
        self.assertEqual(resolved.intent, "shutdown_computer")

    def test_cancel_pending_clears_it(self):
        session = SessionState()
        session.set_pending_confirmation(SHUTDOWN_COMMAND)
        self.assertTrue(session.cancel_pending_confirmation())
        self.assertIsNone(session.get_pending_confirmation())
        self.assertEqual(
            session.resolve_pending_confirmation(), ("none", None)
        )
        self.assertFalse(session.cancel_pending_confirmation())

    def test_get_pending_returns_a_private_copy(self):
        session = SessionState()
        session.set_pending_confirmation(SHUTDOWN_COMMAND)
        session.get_pending_confirmation().command["target"] = "forged"
        self.assertIsNone(
            session.get_pending_confirmation().command["target"]
        )


class TestConfirmationExpiration(unittest.TestCase):
    """Confirmations expire and can never fire afterwards."""

    def test_default_timeout_is_60(self):
        self.assertEqual(DEFAULT_CONFIRMATION_TIMEOUT_SECONDS, 60)

    def test_timeout_uses_configured_value(self):
        with patch.object(ss, "get_value", return_value=15):
            self.assertEqual(ss._confirmation_timeout_seconds(), 15.0)

    def test_timeout_falls_back_on_malformed_config(self):
        # A malformed setting must never make confirmations immortal
        # (huge), vanish instantly (0/neg), or tick too fast (bool).
        for bad in ("60", 0, -5, True, None, 60.5):
            with self.subTest(value=bad):
                with patch.object(ss, "get_value", return_value=bad):
                    self.assertEqual(
                        ss._confirmation_timeout_seconds(),
                        float(DEFAULT_CONFIRMATION_TIMEOUT_SECONDS),
                    )

    def test_expired_confirmation_is_cleared_and_unusable(self):
        session = SessionState()
        timeout = ss._confirmation_timeout_seconds()
        with patch.object(ss.time, "monotonic", return_value=1000.0):
            session.set_pending_confirmation(SHUTDOWN_COMMAND)
        with patch.object(
            ss.time, "monotonic", return_value=1000.0 + timeout - 0.5
        ):
            status, pending = session.resolve_pending_confirmation()
            self.assertEqual(status, "active")
            self.assertIsNotNone(pending)
        with patch.object(
            ss.time, "monotonic", return_value=1000.0 + timeout
        ):
            self.assertEqual(
                session.resolve_pending_confirmation(), ("expired", None)
            )
        # Expired confirmations stay gone — later lookups see "none"
        with patch.object(
            ss.time, "monotonic", return_value=1000.0 + timeout + 1
        ):
            self.assertIsNone(session.get_pending_confirmation())
            self.assertEqual(
                session.resolve_pending_confirmation(), ("none", None)
            )


class TestMainConfirmationFlow(unittest.TestCase):
    """main.py's session-level confirmation round-trip.

    Every turn runs the real pipeline; only the executor is mocked so
    no dangerous operation can ever execute.
    """

    def setUp(self):
        import main

        self.main = main
        self._original_session = main.session
        main.session = SessionState()

    def tearDown(self):
        self.main.session = self._original_session

    def _run_turn(self, text):
        buffer = io.StringIO()
        with patch("main.get_input", return_value=text):
            with redirect_stdout(buffer):
                self.main._process_once()
        return buffer.getvalue()

    def _pending(self):
        return self.main.session.resolve_pending_confirmation()

    # -- Creating a pending confirmation ------------------------------

    def test_shutdown_creates_pending_and_executes_nothing(self):
        with patch("main.execute_command") as mock_execute:
            output = self._run_turn("shutdown computer")
        mock_execute.assert_not_called()
        self.assertIn(
            "Are you sure you want to shut down the computer?", output
        )
        status, pending = self._pending()
        self.assertEqual(status, "active")
        self.assertEqual(pending.intent, "shutdown_computer")

    def test_pending_turn_records_state(self):
        with patch("main.execute_command"):
            self._run_turn("shutdown computer")
        snap = self.main.session.snapshot()
        self.assertEqual(snap["last_input"], "shutdown computer")
        self.assertEqual(snap["last_intent"], "shutdown_computer")
        self.assertIsNotNone(snap["last_command"])
        self.assertIn("Are you sure", snap["last_response"])
        self.assertTrue(
            snap["last_result"]["data"]["pending_confirmation"]
        )

    def test_delete_file_prompt_contains_target(self):
        with patch("main.execute_command") as mock_execute:
            output = self._run_turn("delete file test.txt")
        mock_execute.assert_not_called()
        self.assertIn("delete 'test.txt'", output)
        _, pending = self._pending()
        self.assertEqual(pending.intent, "delete_file")
        self.assertEqual(pending.command["target"], "test.txt")

    # -- Accepting -----------------------------------------------------

    def test_yes_executes_exact_stored_command_after_revalidation(self):
        with patch("main.execute_command"):
            self._run_turn("shutdown computer")
        stored = self.main.session.snapshot()["pending_confirmation"][
            "command"
        ]

        with patch("main.execute_command") as mock_execute:
            mock_execute.return_value = {
                "success": True,
                "message": "Mock shutdown executed.",
                "data": {},
            }
            output = self._run_turn("yes")

        mock_execute.assert_called_once()
        self.assertEqual(mock_execute.call_args.args[0], stored)
        self.assertEqual(mock_execute.call_args.kwargs, {"confirmed": True})
        self.assertIn("Mock shutdown executed.", output)
        self.assertEqual(self._pending(), ("none", None))

    def test_delete_yes_confirms_the_stored_target(self):
        with patch("main.execute_command"):
            self._run_turn("delete file test.txt")
        with patch("main.execute_command") as mock_execute:
            mock_execute.return_value = {
                "success": True,
                "message": "Deleted (mocked).",
                "data": {},
            }
            self._run_turn("yes")
        mock_execute.assert_called_once()
        self.assertEqual(
            mock_execute.call_args.args[0]["target"], "test.txt"
        )
        self.assertEqual(mock_execute.call_args.kwargs, {"confirmed": True})

    def test_confirmation_accept_is_logged(self):
        with patch("main.execute_command"):
            self._run_turn("shutdown computer")
        with self.assertLogs("main", level="INFO") as logs:
            with patch("main.execute_command") as mock_execute:
                mock_execute.return_value = {
                    "success": True, "message": "m", "data": {},
                }
                self._run_turn("confirmed")
        self.assertIn(
            "Confirmation accepted: shutdown_computer", "\n".join(logs.output)
        )

    def test_pending_creation_is_logged(self):
        with self.assertLogs("core.session_state", level="DEBUG") as logs:
            with patch("main.execute_command"):
                self._run_turn("shutdown computer")
        self.assertIn(
            "Pending confirmation created: shutdown_computer",
            "\n".join(logs.output),
        )

    # -- Cancelling ----------------------------------------------------

    def test_no_cancels_pending(self):
        with patch("main.execute_command"):
            self._run_turn("shutdown computer")
        with patch("main.execute_command") as mock_execute:
            output = self._run_turn("no")
        mock_execute.assert_not_called()
        self.assertIn("Shutdown cancelled.", output)
        self.assertEqual(self._pending(), ("none", None))

    def test_cancel_cancels_pending(self):
        with patch("main.execute_command"):
            self._run_turn("shutdown computer")
        with patch("main.execute_command") as mock_execute:
            output = self._run_turn("cancel")
        mock_execute.assert_not_called()
        self.assertIn("Shutdown cancelled.", output)
        self.assertEqual(self._pending(), ("none", None))

    def test_negative_variants_cancel(self):
        for phrase in ("n", "stop", "don't", "do not", "cancel it"):
            with self.subTest(phrase=phrase):
                main_session = SessionState()
                self.main.session = main_session
                with patch("main.execute_command"):
                    self._run_turn("shutdown computer")
                with patch("main.execute_command") as mock_execute:
                    output = self._run_turn(phrase)
                mock_execute.assert_not_called()
                self.assertIn("Shutdown cancelled.", output)
                self.assertEqual(self._pending(), ("none", None))

    def test_cancellation_is_logged(self):
        with patch("main.execute_command"):
            self._run_turn("shutdown computer")
        with self.assertLogs("main", level="INFO") as logs:
            with patch("main.execute_command"):
                self._run_turn("no")
        self.assertIn("Confirmation cancelled", "\n".join(logs.output))

    # -- Refusals ------------------------------------------------------

    def test_yes_without_pending_does_nothing(self):
        with patch("main.execute_command") as mock_execute:
            output = self._run_turn("yes")
        mock_execute.assert_not_called()
        self.assertIn("There is nothing to confirm.", output)
        self.assertEqual(self._pending(), ("none", None))

    def test_expired_confirmation_cannot_execute(self):
        timeout = ss._confirmation_timeout_seconds()
        with patch.object(ss.time, "monotonic", return_value=1000.0):
            with patch("main.execute_command"):
                self._run_turn("shutdown computer")
        with patch.object(
            ss.time, "monotonic", return_value=1000.0 + timeout + 1
        ):
            with patch("main.execute_command") as mock_execute:
                output = self._run_turn("yes")
        mock_execute.assert_not_called()
        self.assertIn("That confirmation has expired.", output)
        self.assertEqual(self._pending(), ("none", None))

    def test_unrelated_input_cancels_pending_and_runs_new_command(self):
        with patch("main.execute_command"):
            self._run_turn("shutdown computer")
        with patch("main.execute_command") as mock_execute:
            mock_execute.return_value = {
                "success": True,
                "message": "Chrome opened successfully.",
                "data": {},
            }
            output = self._run_turn("open chrome")

        # Only the NEW command runs — never the parked shutdown.
        mock_execute.assert_called_once()
        self.assertEqual(
            mock_execute.call_args.args[0]["intent"], "open_application"
        )
        self.assertIn("Chrome opened successfully.", output)
        self.assertEqual(self._pending(), ("none", None))

    def test_empty_input_leaves_pending_alone(self):
        with patch("main.execute_command"):
            self._run_turn("shutdown computer")
        output = self._run_turn("")
        self.assertIn("No command entered.", output)
        status, _ = self._pending()
        self.assertEqual(status, "active")

    def test_unrelated_cancellation_is_logged(self):
        with patch("main.execute_command"):
            self._run_turn("shutdown computer")
        with self.assertLogs("main", level="INFO") as logs:
            with patch("main.execute_command") as mock_execute:
                mock_execute.return_value = {
                    "success": True, "message": "m", "data": {},
                }
                self._run_turn("open chrome")
        self.assertIn(
            "Pending confirmation cancelled by unrelated input",
            "\n".join(logs.output),
        )

    # -- Revalidation --------------------------------------------------

    def test_revalidation_failure_blocks_execution(self):
        with patch("main.execute_command"):
            self._run_turn("shutdown computer")
        with patch(
            "main.validate_command",
            return_value={
                "valid": False,
                "message": "stale command",
                "errors": ["stale"],
            },
        ):
            with patch("main.execute_command") as mock_execute:
                output = self._run_turn("yes")
        mock_execute.assert_not_called()
        self.assertIn("That command is no longer valid.", output)
        self.assertEqual(self._pending(), ("none", None))

    # -- Security ------------------------------------------------------

    def test_attack_input_executes_nothing_and_kills_pending(self):
        """'yes && delete important_file' is never shell-interpreted."""
        with patch("main.execute_command"):
            self._run_turn("shutdown computer")
        with patch("main.execute_command") as mock_execute:
            output = self._run_turn("yes && delete important_file")
        mock_execute.assert_not_called()
        # No shell interpretation: the input is only ever tokenized,
        # spell-checked and validated — it fails as an unknown command.
        self.assertIn("Unknown command.", output)
        # The parked shutdown is cancelled, not remembered.
        self.assertEqual(self._pending(), ("none", None))

    def test_yes_never_executes_without_matching_pending(self):
        """Direct security check: shutdown/restart/delete need pending."""
        for phrase in ("yes", "y", "confirm", "proceed"):
            with self.subTest(phrase=phrase):
                self.main.session = SessionState()
                with patch("main.execute_command") as mock_execute:
                    self._run_turn(phrase)
                mock_execute.assert_not_called()

    def test_command_is_dict_only_never_shell_string(self):
        """The stored pending command is structured, not executable text."""
        session = SessionState()
        pending = session.set_pending_confirmation(SHUTDOWN_COMMAND)
        self.assertIsInstance(pending.command, dict)
        self.assertIn("intent", pending.command)


class TestUnknownAndHelpState(unittest.TestCase):
    """PARTS 25-26: unknown/help turns update state, never go stale."""

    def setUp(self):
        import main

        self.main = main
        self._original_session = main.session
        main.session = SessionState()

    def tearDown(self):
        self.main.session = self._original_session

    def _run_turn(self, text):
        buffer = io.StringIO()
        with patch("main.get_input", return_value=text):
            with redirect_stdout(buffer):
                self.main._process_once()
        return buffer.getvalue()

    def test_unknown_command_updates_all_state(self):
        output = self._run_turn("blah blah")
        snap = self.main.session.snapshot()
        self.assertEqual(snap["last_input"], "blah blah")
        self.assertEqual(snap["last_tokens"], ["blah", "blah"])
        self.assertEqual(snap["last_intent"], "unknown")
        self.assertEqual(snap["last_command"]["intent"], "unknown")
        self.assertFalse(snap["last_result"]["success"])
        self.assertEqual(snap["last_response"], "Unknown command.")
        self.assertIn("Unknown command.", output)

    def test_unknown_does_not_keep_previous_command_state(self):
        self._run_turn("help")
        self.assertEqual(
            self.main.session.snapshot()["last_intent"], "help"
        )
        self._run_turn("blah blah")
        snap = self.main.session.snapshot()
        self.assertEqual(snap["last_intent"], "unknown")
        self.assertNotEqual(snap["last_command"]["intent"], "help")
        self.assertFalse(snap["last_result"]["success"])

    def test_help_turn_is_recorded(self):
        self._run_turn("help")
        snap = self.main.session.snapshot()
        self.assertEqual(snap["last_intent"], "help")
        self.assertIsNotNone(snap["last_response"])
        self.assertTrue(snap["last_result"]["success"])

    def test_commands_turn_is_recorded(self):
        self._run_turn("commands")
        snap = self.main.session.snapshot()
        self.assertEqual(snap["last_intent"], "list_commands")
        self.assertTrue(snap["last_result"]["success"])

    def test_main_logs_session_lifecycle(self):
        with self.assertLogs("main", level="INFO") as logs:
            with patch(
                "main.get_input", side_effect=KeyboardInterrupt()
            ):
                with redirect_stdout(io.StringIO()):
                    self.main.main()
        joined = "\n".join(logs.output)
        self.assertIn("session started", joined)
        self.assertIn("session ended", joined)


class TestSessionPrivacy(unittest.TestCase):
    """Session state lives in memory only — never on disk."""

    FORBIDDEN_FILES = (
        "session.json",
        "memory.json",
        "history.json",
        "history.db",
    )

    def test_no_session_files_are_created(self):
        import main

        original_session = main.session
        main.session = SessionState()
        try:
            with patch("main.execute_command") as mock_execute:
                with patch(
                    "main.get_input",
                    side_effect=["shutdown computer", "no",
                                 KeyboardInterrupt()],
                ):
                    with redirect_stdout(io.StringIO()):
                        main.main()
            mock_execute.assert_not_called()
        finally:
            main.session = original_session

        root = Path(__file__).resolve().parent.parent
        for name in self.FORBIDDEN_FILES:
            self.assertFalse(
                (root / name).exists(),
                f"{name} must never be created by session state",
            )

    def test_session_state_has_no_persistence_api(self):
        session = SessionState()
        for method in ("save", "persist", "to_json", "write"):
            self.assertFalse(hasattr(session, method))


class TestSessionStateHistorySeparation(unittest.TestCase):
    """Session State and Command History are separate stores (Step 20).

    Neither one duplicates the other: Session State tracks the
    CURRENT turn's context and pending confirmation, while
    CommandHistory owns the bounded list of past commands.
    """

    def setUp(self):
        from core.command_history import CommandHistory, set_command_history

        self.session = SessionState()
        self.history = CommandHistory(max_entries=10)
        set_command_history(self.history)

    def tearDown(self):
        from core.command_history import set_command_history

        set_command_history(None)

    def test_snapshot_has_no_history_fields(self):
        """History entries never leak into Session State keys."""
        self.history.add_entry(
            input_text="open chrome", command_text="open chrome",
            intent="open_application", success=True, message="ok",
        )
        snap = self.session.snapshot()
        self.assertEqual(set(snap.keys()), EXPECTED_STATE_KEYS)
        for key in snap:
            self.assertNotIn("history", key)
        self.assertNotIn("last_history", snap)

    def test_session_state_has_no_history_api(self):
        for method in (
            "add_entry", "get_history", "clear_history", "add_history",
        ):
            self.assertFalse(hasattr(self.session, method))

    def test_clearing_history_does_not_touch_session_state(self):
        self.session.set_last_input("open chrome")
        self.history.add_entry(
            input_text="open chrome", command_text="open chrome"
        )
        self.history.clear_history()
        self.assertEqual(self.history.count(), 0)
        self.assertEqual(self.session.snapshot()["last_input"], "open chrome")

    def test_resetting_session_does_not_clear_history(self):
        self.history.add_entry(
            input_text="open chrome", command_text="open chrome"
        )
        self.session.reset_session()
        self.assertEqual(self.history.count(), 1)

    def test_single_history_instance_is_used_by_main(self):
        """main.py creates ONE CommandHistory for the whole session
        (PART 31), and every recording site reads it through the
        accessor — so installing an isolated instance here is what
        the pipeline (and the Executor) sees.
        """
        import main

        from core.command_history import CommandHistory, get_command_history

        self.assertIsInstance(main.history, CommandHistory)
        self.assertIs(get_command_history(), self.history)


if __name__ == "__main__":
    unittest.main()
