"""
Multi-Command Executor Tests (Step 19)

Covers the Multi-Command Execution Layer (planning reuse, sequential
execution, stop policies) and main.py's end-to-end multi-command flow,
including confirmation safety and command-injection resistance.

Every test mocks the Executor — no real application is ever launched,
no dangerous operation ever executes, and no shell is ever invoked.
"""

import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

# Add parent directory to path so we can import core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.command_history import CommandHistory
from core.exceptions import MultiCommandError
from core.executor import CONFIRMATION_INTENTS
from core.multi_command_executor import (
    MultiCommandOutcome,
    PlannedCommand,
    plan_command,
    run_multi_commands,
)
from core.multi_command_parser import (
    MAX_COMMANDS,
    TOO_MANY_COMMANDS_MESSAGE,
)
from core.response_manager import format_response
from core.session_state import SessionState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _success_recorder(executed):
    """Build an execute_command side effect that records call order."""

    def fake(command, confirmed=None):
        executed.append(
            {"intent": command.get("intent"),
             "target": command.get("target"),
             "confirmed": confirmed}
        )
        intent = command.get("intent")
        if intent == "help":
            message = "Jarvis can help you."
        elif intent == "list_commands":
            message = "Available commands listed."
        elif intent == "shutdown_computer":
            message = "Computer shutting down."
        elif intent == "restart_computer":
            message = "Computer restarting."
        elif intent == "delete_file":
            message = "File deleted."
        else:
            message = f"{command.get('target')} opened successfully."
        return {"success": True, "message": message, "data": {}}

    return fake


def _combined(outcome):
    """Build the combined result dict exactly like main.py does."""
    return {
        "success": outcome.success,
        "results": list(outcome.results),
        "data": {"command_count": len(outcome.plans),
                 "executed_count": outcome.executed_count,
                 "stop_reason": outcome.stop_reason},
    }


# ---------------------------------------------------------------------------
# Planning (reuse of the existing pipeline)
# ---------------------------------------------------------------------------


class TestPlanCommands(unittest.TestCase):
    """plan_command reuses normalize/tokenize/spell/intent/parse/validate."""

    def test_plan_of_single_command_is_valid(self):
        plan = plan_command("open chrome")
        self.assertIsInstance(plan, PlannedCommand)
        self.assertEqual(plan.text, "open chrome")
        self.assertEqual(plan.intent, "open_application")
        self.assertTrue(plan.valid)
        self.assertIn("chrome", plan.corrected_tokens)

    def test_plan_records_tokens_and_corrected_tokens(self):
        plan = plan_command("opn chrme")
        self.assertIn("opn", plan.tokens)
        self.assertIn("open", plan.corrected_tokens)

    def test_plan_detects_intent_per_command(self):
        self.assertEqual(plan_command("open chrome").intent,
                         "open_application")
        self.assertEqual(plan_command("search web python").intent,
                         "search_web")
        self.assertEqual(plan_command("shutdown computer").intent,
                         "shutdown_computer")

    def test_plan_of_malformed_command_is_marked_invalid(self):
        plan = plan_command("blah blah")
        self.assertFalse(plan.valid)
        self.assertEqual(plan.intent, "unknown")
        self.assertEqual(plan.validation["message"], "Unknown command.")

    def test_plan_structured_command_shape(self):
        """PART 34: reuse the existing Command representation."""
        plan = plan_command("open chrome")
        self.assertEqual(set(plan.command.keys()),
                         {"intent", "target", "arguments"})
        self.assertEqual(plan.command["intent"], "open_application")
        self.assertEqual(plan.command["target"], "chrome")
        self.assertEqual(plan.command["arguments"], {})


# ---------------------------------------------------------------------------
# Sequential execution and ordered results
# ---------------------------------------------------------------------------


class TestSequentialExecution(unittest.TestCase):
    """Commands execute one after another, in input order (PART 16)."""

    def test_commands_execute_in_order(self):
        executed = []
        with patch(
            "core.multi_command_executor.execute_command",
            side_effect=_success_recorder(executed),
        ):
            outcome = run_multi_commands(
                ["open chrome", "open calculator", "open notepad"]
            )
        targets = [call["target"] for call in executed]
        self.assertEqual(targets, ["chrome", "calculator", "notepad"])
        self.assertEqual(outcome.executed_count, 3)
        self.assertIsNone(outcome.stop_reason)
        self.assertTrue(outcome.success)

    def test_results_entry_structure(self):
        """PART 13: {command, success, message, data} per command."""
        with patch(
            "core.multi_command_executor.execute_command",
            side_effect=_success_recorder([]),
        ):
            outcome = run_multi_commands(
                ["open chrome", "search web python tutorials"]
            )
        self.assertEqual(len(outcome.results), 2)
        for entry in outcome.results:
            self.assertEqual(
                set(entry.keys()), {"command", "success", "message", "data"}
            )
        self.assertEqual(outcome.results[0]["command"], "open chrome")
        self.assertTrue(outcome.results[0]["success"])

    def test_results_preserve_execution_order(self):
        executed = []
        with patch(
            "core.multi_command_executor.execute_command",
            side_effect=_success_recorder(executed),
        ):
            outcome = run_multi_commands(
                ["open chrome", "open calculator", "open notepad"]
            )
        commands = [entry["command"] for entry in outcome.results]
        self.assertEqual(
            commands, ["open chrome", "open calculator", "open notepad"]
        )

    def test_response_joins_messages_in_order(self):
        """PART 23/24: one line per command, execution order kept."""
        with patch(
            "core.multi_command_executor.execute_command",
            side_effect=_success_recorder([]),
        ):
            outcome = run_multi_commands(
                ["open chrome", "open calculator", "open notepad"]
            )
        response = format_response(_combined(outcome))
        self.assertEqual(
            response.splitlines(),
            [
                "chrome opened successfully.",
                "calculator opened successfully.",
                "notepad opened successfully.",
            ],
        )

    def test_combined_success_flag(self):
        with patch(
            "core.multi_command_executor.execute_command",
            side_effect=_success_recorder([]),
        ):
            outcome = run_multi_commands(["open chrome", "open notepad"])
        combined = _combined(outcome)
        self.assertTrue(combined["success"])
        self.assertEqual(combined["data"]["command_count"], 2)
        self.assertEqual(combined["data"]["executed_count"], 2)

    def test_plans_are_all_pre_validated(self):
        with patch("core.multi_command_executor.execute_command"):
            outcome = run_multi_commands(
                ["open chrome", "open calculator"]
            )
        self.assertEqual(len(outcome.plans), 2)
        self.assertTrue(all(plan.valid for plan in outcome.plans))


# ---------------------------------------------------------------------------
# Pre-validation (PART 21/22)
# ---------------------------------------------------------------------------


class TestPreValidation(unittest.TestCase):
    """The whole chain is validated BEFORE anything executes."""

    def test_malformed_second_command_prevents_all_execution(self):
        executed = []
        with patch(
            "core.multi_command_executor.execute_command",
            side_effect=_success_recorder(executed),
        ):
            outcome = run_multi_commands(["open chrome", "open"])
        self.assertEqual(executed, [])
        self.assertEqual(outcome.stop_reason, "validation")
        self.assertFalse(outcome.success)
        self.assertEqual(len(outcome.results), 1)
        self.assertEqual(outcome.results[0]["command"], "open")
        self.assertFalse(outcome.results[0]["success"])

    def test_malformed_third_command_prevents_earlier_execution(self):
        executed = []
        with patch(
            "core.multi_command_executor.execute_command",
            side_effect=_success_recorder(executed),
        ):
            outcome = run_multi_commands(
                ["open chrome", "open calculator", "search web"]
            )
        self.assertEqual(executed, [])
        self.assertEqual(outcome.stop_reason, "validation")

    def test_validation_failure_message_is_the_real_message(self):
        with patch("core.multi_command_executor.execute_command"):
            outcome = run_multi_commands(["open chrome", "blah blah"])
        self.assertEqual(outcome.results[0]["message"], "Unknown command.")
        response = format_response(_combined(outcome))
        self.assertEqual(response, "Unknown command.")

    def test_invalid_plans_still_recorded_in_plans(self):
        with patch("core.multi_command_executor.execute_command"):
            outcome = run_multi_commands(["open chrome", "open"])
        self.assertEqual(len(outcome.plans), 2)
        self.assertTrue(outcome.plans[0].valid)
        self.assertFalse(outcome.plans[1].valid)


# ---------------------------------------------------------------------------
# Stop on failure (PART 17)
# ---------------------------------------------------------------------------


class TestStopOnFailure(unittest.TestCase):
    """A failed command ends the chain — later commands never run."""

    def test_third_command_does_not_execute_after_failure(self):
        calls = []

        def fail_second(command, confirmed=None):
            calls.append(command.get("target"))
            if len(calls) == 2:
                return {"success": False,
                        "message": "I couldn't open Firefox.",
                        "data": {}}
            return {"success": True,
                    "message": f"{command.get('target')} opened.",
                    "data": {}}

        with patch(
            "core.multi_command_executor.execute_command",
            side_effect=fail_second,
        ):
            outcome = run_multi_commands(
                ["open chrome", "open firefox", "open notepad"]
            )
        self.assertEqual(calls, ["chrome", "firefox"])
        self.assertEqual(outcome.stop_reason, "failure")
        self.assertFalse(outcome.success)

    def test_failure_stops_response_after_previous_messages(self):
        def fail_second(command, confirmed=None):
            if command.get("target") == "firefox":
                return {"success": False,
                        "message": "I couldn't open Firefox.",
                        "data": {}}
            return {"success": True,
                    "message": "Chrome opened successfully.",
                    "data": {}}

        with patch(
            "core.multi_command_executor.execute_command",
            side_effect=fail_second,
        ):
            outcome = run_multi_commands(
                ["open chrome", "open firefox", "open notepad"]
            )
        response = format_response(_combined(outcome))
        self.assertEqual(
            response.splitlines(),
            ["Chrome opened successfully.", "I couldn't open Firefox."],
        )

    def test_failed_entry_recorded_in_results(self):
        def fail_second(command, confirmed=None):
            if command.get("target") == "firefox":
                return {"success": False,
                        "message": "I couldn't open Firefox.",
                        "data": {}}
            return {"success": True, "message": "ok.", "data": {}}

        with patch(
            "core.multi_command_executor.execute_command",
            side_effect=fail_second,
        ):
            outcome = run_multi_commands(
                ["open chrome", "open firefox", "open notepad"]
            )
        self.assertEqual(len(outcome.results), 2)
        self.assertTrue(outcome.results[0]["success"])
        self.assertFalse(outcome.results[1]["success"])
        self.assertEqual(outcome.final_plan.text, "open firefox")


# ---------------------------------------------------------------------------
# Stop on confirmation (PART 18/19/20)
# ---------------------------------------------------------------------------


class TestConfirmationStop(unittest.TestCase):
    """A dangerous command parks — the layer never executes it."""

    def test_dangerous_intent_is_not_executed(self):
        executed = []
        with patch(
            "core.multi_command_executor.execute_command",
            side_effect=_success_recorder(executed),
        ):
            outcome = run_multi_commands(["open chrome", "shutdown computer"])
        self.assertEqual([c["intent"] for c in executed],
                         ["open_application"])
        self.assertEqual(outcome.stop_reason, "confirmation")
        self.assertIsNotNone(outcome.confirmation)

    def test_confirmation_holds_the_exact_structured_command(self):
        with patch("core.multi_command_executor.execute_command"):
            outcome = run_multi_commands(
                ["open chrome", "shutdown computer"]
            )
        self.assertEqual(outcome.confirmation["intent"],
                         "shutdown_computer")
        self.assertIsNone(outcome.confirmation["target"])

    def test_prompt_is_the_last_response_line(self):
        with patch("core.multi_command_executor.execute_command"):
            outcome = run_multi_commands(
                ["open chrome", "shutdown computer"]
            )
        response = format_response(_combined(outcome))
        self.assertIn(
            "Are you sure you want to shut down the computer?", response
        )

    def test_remaining_commands_are_not_executed_after_confirmation(self):
        executed = []
        with patch(
            "core.multi_command_executor.execute_command",
            side_effect=_success_recorder(executed),
        ):
            outcome = run_multi_commands(
                ["open chrome", "shutdown computer", "open notepad"]
            )
        self.assertEqual([c["target"] for c in executed], ["chrome"])
        self.assertEqual(outcome.executed_count, 1)
        self.assertEqual(outcome.final_plan.text, "shutdown computer")

    def test_every_dangerous_intent_stops_the_chain(self):
        for intent_text in ("shutdown computer", "restart computer",
                            "delete file test.txt", "enable startup",
                            "disable startup"):
            with self.subTest(intent=intent_text):
                executed = []
                with patch(
                    "core.multi_command_executor.execute_command",
                    side_effect=_success_recorder(executed),
                ):
                    outcome = run_multi_commands(
                        ["open chrome", intent_text]
                    )
                self.assertEqual(outcome.stop_reason, "confirmation")
                self.assertEqual(outcome.executed_count, 1)

    def test_confirmation_intents_match_the_executor_policy(self):
        """The chain stops on exactly the executor's dangerous set.

        Step 23 (Phase 17): enable/disable startup change the user's
        Windows configuration, so they share the confirmation policy.
        """
        self.assertEqual(
            CONFIRMATION_INTENTS,
            {
                "shutdown_computer",
                "restart_computer",
                "delete_file",
                "enable_startup",
                "disable_startup",
            },
        )

    def test_confirmation_entry_marks_pending(self):
        with patch("core.multi_command_executor.execute_command"):
            outcome = run_multi_commands(
                ["open chrome", "restart computer"]
            )
        self.assertTrue(
            outcome.results[1]["data"].get("pending_confirmation")
        )


# ---------------------------------------------------------------------------
# Security (PART 42/47)
# ---------------------------------------------------------------------------


class TestSecurity(unittest.TestCase):
    """No shell, no injection, confirmation never bypassed."""

    def _run_with_shell_spy(self, texts):
        """Run the chain with shell APIs patched to detect any call."""
        executed = []
        with patch("os.system") as mock_system, \
                patch("subprocess.run") as mock_run, \
                patch("subprocess.Popen") as mock_popen, \
                patch(
                    "core.multi_command_executor.execute_command",
                    side_effect=_success_recorder(executed),
                ):
            outcome = run_multi_commands(texts)
        shell_calls = (mock_system.call_count + mock_run.call_count
                       + mock_popen.call_count)
        return outcome, executed, shell_calls

    def test_shutdown_with_shell_operators_never_executes_shutdown(self):
        outcome, executed, shell_calls = self._run_with_shell_spy(
            ["open chrome", "shutdown && delete file"]
        )
        self.assertEqual(shell_calls, 0)
        self.assertEqual([c["intent"] for c in executed],
                         ["open_application"])
        self.assertEqual(outcome.stop_reason, "confirmation")

    def test_shell_operator_input_never_reaches_a_shell(self):
        _, _, shell_calls = self._run_with_shell_spy(
            ["open chrome", "shutdown /s /t 0 && rd /s /q C:\\"]
        )
        self.assertEqual(shell_calls, 0)

    def test_delete_in_chain_requires_confirmation(self):
        outcome, executed, shell_calls = self._run_with_shell_spy(
            ["open chrome", "delete file C:\\important.txt"]
        )
        self.assertEqual(shell_calls, 0)
        # Only Chrome executed; the delete parked for confirmation.
        self.assertEqual(len(executed), 1)
        self.assertEqual(executed[0]["intent"], "open_application")
        self.assertEqual(outcome.confirmation["intent"], "delete_file")

    def test_malformed_delete_target_stops_before_any_execution(self):
        """A delete without the 'file' keyword is UNKNOWN → validation
        stops the whole chain — nothing at all executes."""
        executed = []
        with patch(
            "core.multi_command_executor.execute_command",
            side_effect=_success_recorder(executed),
        ):
            outcome = run_multi_commands(
                ["open chrome", "delete C:\\important.txt"]
            )
        self.assertEqual(executed, [])
        self.assertEqual(outcome.stop_reason, "validation")

    def test_confirmation_is_stored_structured_command_not_input(self):
        """The parked command is the parsed dict, never raw text."""
        with patch("core.multi_command_executor.execute_command"):
            outcome = run_multi_commands(
                ["open chrome", "shutdown computer"]
            )
        self.assertIsInstance(outcome.confirmation, dict)
        self.assertEqual(outcome.confirmation["intent"],
                         "shutdown_computer")


# ---------------------------------------------------------------------------
# Limits (defense in depth)
# ---------------------------------------------------------------------------


class TestExecutionLimits(unittest.TestCase):
    """The layer re-checks the chain-size limit."""

    def test_too_many_texts_raise(self):
        texts = [f"open app{i}" for i in range(MAX_COMMANDS + 1)]
        with self.assertRaises(MultiCommandError) as ctx:
            run_multi_commands(texts)
        self.assertEqual(str(ctx.exception), TOO_MANY_COMMANDS_MESSAGE)

    def test_empty_input_returns_failed_outcome(self):
        outcome = run_multi_commands([])
        self.assertIsInstance(outcome, MultiCommandOutcome)
        self.assertFalse(outcome.success)
        self.assertEqual(outcome.results, [])

    def test_non_list_input_returns_failed_outcome(self):
        outcome = run_multi_commands(None)
        self.assertFalse(outcome.success)


# ---------------------------------------------------------------------------
# main.py end-to-end multi-command flow (PART 40/43)
# ---------------------------------------------------------------------------


class TestMainMultiCommandFlow(unittest.TestCase):
    """main._process_once with multi-command input.

    Both executor references are mocked so nothing real runs, and the
    recorded order proves sequential execution.
    """

    def setUp(self):
        import main

        self.main = main
        self._original_session = main.session
        main.session = SessionState()
        self.executed = []
        self.fake = _success_recorder(self.executed)

    def tearDown(self):
        self.main.session = self._original_session

    def _run_turn(self, text):
        buffer = io.StringIO()
        with patch("main.get_input", return_value=text), \
                patch("main.execute_command", side_effect=self.fake), \
                patch(
                    "core.multi_command_executor.execute_command",
                    side_effect=self.fake,
                ), \
                redirect_stdout(buffer):
            self.main._process_once()
        return buffer.getvalue()

    def test_chain_executes_in_order(self):
        output = self._run_turn("open chrome and open calculator")
        targets = [c["target"] for c in self.executed]
        self.assertEqual(targets, ["chrome", "calculator"])
        self.assertIn("chrome opened successfully.", output)
        self.assertIn("calculator opened successfully.", output)

    def test_responses_appear_in_execution_order(self):
        output = self._run_turn(
            "open chrome and open calculator and open notepad"
        )
        lines = [line for line in output.splitlines() if line.strip()]
        self.assertEqual(
            lines,
            [
                "chrome opened successfully.",
                "calculator opened successfully.",
                "notepad opened successfully.",
            ],
        )

    def test_single_command_unchanged(self):
        output = self._run_turn("open chrome")
        self.assertEqual(len(self.executed), 1)
        self.assertEqual(self.executed[0]["target"], "chrome")
        self.assertEqual(output.count("chrome opened successfully."), 1)

    def test_search_query_is_not_split(self):
        output = self._run_turn("search web cats and dogs")
        self.assertEqual(len(self.executed), 1)
        self.assertEqual(self.executed[0]["intent"], "search_web")

    def test_malformed_later_command_blocks_everything(self):
        output = self._run_turn("open chrome and open")
        self.assertEqual(self.executed, [])
        self.assertIn("requires a target", output)

    def test_session_records_multi_command_state(self):
        self._run_turn("open chrome and search web python tutorials")
        snap = self.main.session.snapshot()
        self.assertEqual(snap["last_input"],
                         "open chrome and search web python tutorials")
        self.assertEqual(len(snap["last_commands"]), 2)
        self.assertEqual(len(snap["last_results"]), 2)
        self.assertEqual(snap["last_result"]["data"]["command_count"], 2)
        self.assertIsNone(snap["last_result"]["data"]["stop_reason"])
        # Pipeline fields reflect the command the turn ended on.
        self.assertEqual(snap["last_intent"], "search_web")
        # Search query content is preserved (Step 26)
        self.assertEqual(snap["last_command"]["target"],
                         "python tutorials")
        self.assertEqual(
            snap["last_response"].splitlines(),
            [
                "chrome opened successfully.",
                "python tutorials opened successfully.",
            ],
        )

    def test_empty_turn_is_not_a_chain(self):
        output = self._run_turn("")
        self.assertIn("No command entered.", output)
        self.assertEqual(self.executed, [])

    def test_too_many_commands_get_safe_message(self):
        text = " and ".join(f"open app{i}" for i in range(11))
        output = self._run_turn(text)
        self.assertEqual(self.executed, [])
        self.assertIn(
            "I can process up to 10 commands at a time.", output
        )

    def test_overlong_command_gets_safe_message(self):
        output = self._run_turn("open " + "a" * 600)
        self.assertEqual(self.executed, [])
        self.assertIn("Command is too long to process.", output)


# ---------------------------------------------------------------------------
# main.py confirmation round-trip with chains (PART 41)
# ---------------------------------------------------------------------------


class TestMainMultiConfirmationFlow(unittest.TestCase):
    """Chains that reach a dangerous command park; answers affect only
    the parked command — the chain is never re-run."""

    def setUp(self):
        import main

        self.main = main
        self._original_session = main.session
        main.session = SessionState()
        self.executed = []
        self.fake = _success_recorder(self.executed)

    def tearDown(self):
        self.main.session = self._original_session

    def _run_turn(self, text):
        buffer = io.StringIO()
        with patch("main.get_input", return_value=text), \
                patch("main.execute_command", side_effect=self.fake), \
                patch(
                    "core.multi_command_executor.execute_command",
                    side_effect=self.fake,
                ), \
                redirect_stdout(buffer):
            self.main._process_once()
        return buffer.getvalue()

    def _pending(self):
        return self.main.session.resolve_pending_confirmation()

    def test_yes_executes_only_the_stored_shutdown(self):
        first = self._run_turn("open chrome and shutdown computer")
        self.assertIn("Are you sure", first)
        status, pending = self._pending()
        self.assertEqual(status, "active")
        self.assertEqual(pending.intent, "shutdown_computer")
        # Only Chrome ran so far.
        self.assertEqual([c["target"] for c in self.executed], ["chrome"])

        second = self._run_turn("yes")
        self.assertIn("Computer shutting down.", second)
        # Chrome did NOT run again — only the stored shutdown.
        intents = [c["intent"] for c in self.executed]
        self.assertEqual(intents,
                         ["open_application", "shutdown_computer"])
        self.assertTrue(self.executed[1]["confirmed"])
        status, _ = self._pending()
        self.assertEqual(status, "none")

    def test_chain_after_confirmation_is_never_resumed(self):
        first = self._run_turn(
            "open chrome and shutdown computer and open notepad"
        )
        self.assertIn("Are you sure", first)
        self.assertNotIn("notepad opened", first)
        self.assertNotIn(
            "notepad", [c["target"] for c in self.executed if c["target"]]
        )

        self._run_turn("yes")
        targets = [c["target"] for c in self.executed]
        self.assertEqual(targets, ["chrome", None])
        self.assertNotIn("notepad", targets)

    def test_no_cancels_the_parked_shutdown(self):
        self._run_turn("open chrome and shutdown computer")
        output = self._run_turn("no")
        self.assertIn("Shutdown cancelled.", output)
        status, _ = self._pending()
        self.assertEqual(status, "none")
        intents = [c["intent"] for c in self.executed]
        self.assertEqual(intents, ["open_application"])

    def test_restart_chain_requires_confirmation(self):
        output = self._run_turn("open chrome and restart computer")
        self.assertIn("Are you sure you want to restart", output)
        status, pending = self._pending()
        self.assertEqual(status, "active")
        self.assertEqual(pending.intent, "restart_computer")
        self.assertEqual(
            [c["intent"] for c in self.executed], ["open_application"]
        )

    def test_delete_chain_requires_confirmation_and_keeps_safety(self):
        output = self._run_turn(
            "open chrome and delete file C:\\important.txt"
        )
        self.assertIn("Are you sure you want to delete", output)
        status, pending = self._pending()
        self.assertEqual(status, "active")
        self.assertEqual(pending.intent, "delete_file")
        # The delete was never executed — only Chrome ran.
        self.assertEqual(
            [c["intent"] for c in self.executed], ["open_application"]
        )

    def test_yes_with_no_pending_after_chain_does_nothing(self):
        self._run_turn("open chrome and open calculator")
        executed_before = len(self.executed)
        output = self._run_turn("yes")
        self.assertIn("There is nothing to confirm.", output)
        self.assertEqual(len(self.executed), executed_before)

    def test_unrelated_input_cancels_parked_chain_command(self):
        self._run_turn("open chrome and shutdown computer")
        executed_before = len(self.executed)
        output = self._run_turn("help")
        self.assertIn("Jarvis can help you.", output)
        status, _ = self._pending()
        self.assertEqual(status, "none")
        # Help ran; the shutdown never did.
        self.assertEqual(len(self.executed), executed_before + 1)
        self.assertEqual(self.executed[-1]["intent"], "help")


# ---------------------------------------------------------------------------
# main.py injection attempts (PART 42)
# ---------------------------------------------------------------------------


class TestMainInjectionSafety(unittest.TestCase):
    """Attack inputs never become shell execution."""

    def setUp(self):
        import main

        self.main = main
        self._original_session = main.session
        main.session = SessionState()

    def tearDown(self):
        self.main.session = self._original_session
        # Parked confirmations from attack inputs must not survive.
        self.main.session.cancel_pending_confirmation()

    def _run_turn(self, text, fake):
        buffer = io.StringIO()
        with patch("main.get_input", return_value=text), \
                patch("main.execute_command", side_effect=fake), \
                patch(
                    "core.multi_command_executor.execute_command",
                    side_effect=fake,
                ), \
                patch("os.system") as mock_system, \
                patch("subprocess.run") as mock_run, \
                patch("subprocess.Popen") as mock_popen, \
                redirect_stdout(buffer):
            self.main._process_once()
        return buffer.getvalue(), (
            mock_system.call_count + mock_run.call_count
            + mock_popen.call_count
        )

    def test_shutdown_and_delete_operators_never_execute(self):
        executed = []
        output, shell_calls = self._run_turn(
            "open chrome and shutdown && delete file",
            _success_recorder(executed),
        )
        self.assertEqual(shell_calls, 0)
        # Chrome ran; the bare "shutdown" is now correctly recognized
        # as a dangerous command and parks for confirmation — the
        # injected "delete file" is never reached.
        self.assertEqual(
            [c["intent"] for c in executed], ["open_application"]
        )
        _, pending = self.main.session.resolve_pending_confirmation()
        self.assertIsNotNone(pending)
        self.assertEqual(pending.intent, "shutdown_computer")

    def test_powershell_after_semicolon_is_not_shell_execution(self):
        executed = []
        output, shell_calls = self._run_turn(
            "open chrome; powershell Remove-Item x",
            _success_recorder(executed),
        )
        self.assertEqual(shell_calls, 0)
        # It stays a single OPEN_APPLICATION command — the manager's
        # allowlist (not a shell) decides what happens.
        self.assertEqual(len(executed), 1)
        self.assertEqual(executed[0]["intent"], "open_application")

    def test_shell_chain_operators_are_never_interpreted(self):
        executed = []
        output, shell_calls = self._run_turn(
            "open chrome and open calc & taskkill /f /im notepad.exe",
            _success_recorder(executed),
        )
        self.assertEqual(shell_calls, 0)


# ---------------------------------------------------------------------------
# History recording in the layer (Step 20 PART 32)
# ---------------------------------------------------------------------------


class TestLayerHistoryRecording(unittest.TestCase):
    """run_multi_commands records every processed command as it runs.

    The layer receives the session's history explicitly — recording
    is observational and optional, and can never change execution.
    """

    def setUp(self):
        self.history = CommandHistory(max_entries=50)
        self.executed = []
        self.fake = _success_recorder(self.executed)

    def _run(self, texts, input_text=None, fake=None):
        with patch(
            "core.multi_command_executor.execute_command",
            side_effect=fake or self.fake,
        ):
            return run_multi_commands(
                texts, history=self.history, input_text=input_text
            )

    def test_records_each_executed_command(self):
        outcome = self._run(["open chrome", "open calculator"])
        entries = self.history.get_history()
        self.assertEqual(len(entries), 2)
        self.assertEqual(
            [e.command_text for e in entries],
            ["open chrome", "open calculator"],
        )
        self.assertEqual([e.id for e in entries], [1, 2])
        self.assertEqual(
            [e.success for e in entries], [True, True]
        )
        self.assertIsNone(outcome.stop_reason)

    def test_entries_share_the_full_original_input(self):
        full = "open chrome and open calculator"
        self._run(
            ["open chrome", "open calculator"], input_text=full
        )
        for entry in self.history.get_history():
            self.assertEqual(entry.input_text, full)

    def test_input_text_defaults_to_joined_segments(self):
        self._run(["open chrome", "open calculator"])
        entries = self.history.get_history()
        self.assertEqual(
            entries[0].input_text, "open chrome and open calculator"
        )

    def test_without_history_parameter_records_nothing(self):
        """Default behavior is unchanged: no history, no side effects."""
        with patch(
            "core.multi_command_executor.execute_command",
            side_effect=self.fake,
        ):
            run_multi_commands(["open chrome"])
        self.assertEqual(self.history.count(), 0)

    def test_pre_validation_failure_records_only_invalid_command(self):
        outcome = self._run(["open chrome", "open"])
        entries = self.history.get_history()
        # Nothing executed — only the command that failed is recorded.
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].command_text, "open")
        self.assertIs(entries[0].success, False)
        self.assertEqual(outcome.stop_reason, "validation")
        self.assertEqual(self.executed, [])

    def test_failure_entry_recorded_before_chain_stops(self):
        def flaky(command, confirmed=None):
            result = self.fake(command, confirmed)
            if command.get("target") == "calculator":
                result = {"success": False,
                          "message": "calculator failed.",
                          "data": {}}
            return result

        outcome = self._run(
            ["open chrome", "open calculator", "open notepad"],
            fake=flaky,
        )
        entries = self.history.get_history()
        self.assertEqual(
            [e.command_text for e in entries],
            ["open chrome", "open calculator"],
        )
        self.assertIs(entries[1].success, False)
        self.assertEqual(outcome.stop_reason, "failure")
        # Notepad never ran → no entry.
        self.assertNotIn(
            "notepad", [c["target"] for c in self.executed]
        )

    def test_confirmation_recorded_as_awaiting(self):
        outcome = self._run(["open chrome", "shutdown computer"])
        entries = self.history.get_history()
        self.assertEqual(len(entries), 2)
        self.assertIs(entries[0].success, True)
        self.assertIsNone(entries[1].success)
        self.assertEqual(entries[1].command_text, "shutdown computer")
        self.assertIn("Are you sure", entries[1].message)
        self.assertIsNotNone(outcome.confirmation)
        self.assertEqual(
            [c["intent"] for c in self.executed], ["open_application"]
        )

    def test_history_command_in_chain_is_exempt_but_executed(self):
        self._run(["open chrome", "history"])
        entries = self.history.get_history()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].command_text, "open chrome")
        self.assertEqual(
            [c["intent"] for c in self.executed],
            ["open_application", "history"],
        )

    def test_recorded_text_is_sanitized(self):
        self._run(
            ["open chrome", "search web token=abc123"],
            input_text="open chrome and search web token=abc123",
        )
        entries = self.history.get_history()
        self.assertEqual(len(entries), 2)
        self.assertNotIn("abc123", entries[1].command_text)
        self.assertNotIn("abc123", entries[1].input_text)
        # Execution saw the raw query (tokenization may re-space it).
        self.assertIn("abc123", self.executed[1]["target"])

    def test_recording_cannot_break_the_chain(self):
        """A broken history never changes what executes (PART 33)."""
        broken = CommandHistory(max_entries=5)
        with patch(
            "core.multi_command_executor.execute_command",
            side_effect=self.fake,
        ), patch.object(
            broken, "add_entry", side_effect=RuntimeError("boom")
        ):
            outcome = run_multi_commands(
                ["open chrome", "open calculator"],
                history=broken,
                input_text="open chrome and open calculator",
            )
        # Both commands still executed and the chain completed.
        self.assertEqual(
            [c["target"] for c in self.executed],
            ["chrome", "calculator"],
        )
        self.assertIsNone(outcome.stop_reason)
        self.assertEqual(outcome.executed_count, 2)


if __name__ == "__main__":
    unittest.main()
