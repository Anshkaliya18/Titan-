"""
Command History Tests (Step 20)

Covers the bounded in-memory CommandHistory (entries, ordering,
recent, limit, IDs, clear, immutability, sanitization), the
HISTORY / CLEAR_HISTORY routing through the Executor, main.py's
single/multi/confirmation integration, and the security rules:
history never executes, replays, modifies, or persists anything.

Every test mocks the Executor where a real command would run — no
real application is ever launched and no shell is ever invoked.
"""

import io
import os
import sys
import time
import unittest
from contextlib import redirect_stdout
from dataclasses import FrozenInstanceError
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

# Add parent directory to path so we can import core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.command_history import (
    DEFAULT_MAX_ENTRIES,
    HARD_MAX_ENTRIES,
    HISTORY_EXEMPT_INTENTS,
    REDACTED,
    CommandHistory,
    HistoryEntry,
    entry_to_dict,
    get_command_history,
    record_entry,
    sanitize_history_text,
    set_command_history,
)
from core.executor import execute_command
from core.response_manager import format_response
from core.session_state import SessionState

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The original executor reference — captured before any patching so
# history/clear-history commands can be routed to the real handler
# inside mocked chains.
from core.executor import execute_command as real_execute_command  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_exec(executed, fail_targets=()):
    """Build an execute_command side effect that records call order."""

    def fake(command, confirmed=None):
        intent = command.get("intent")
        target = command.get("target")
        executed.append({
            "intent": intent,
            "target": target,
            "confirmed": confirmed,
            "command": dict(command),
        })
        if intent == "help":
            message, success = "Jarvis can help you.", True
        elif intent == "list_commands":
            message, success = "Available commands listed.", True
        elif intent == "shutdown_computer":
            message, success = "Computer shutting down.", True
        elif intent == "restart_computer":
            message, success = "Computer restarting.", True
        elif intent == "history":
            message, success = "Command history retrieved.", True
        elif intent == "clear_history":
            message, success = "Command history cleared.", True
        elif intent == "unknown":
            message, success = "Unknown command.", False
        elif target in fail_targets:
            message, success = f"{target} failed.", False
        else:
            message, success = f"{target} opened successfully.", True
        return {"success": success, "message": message, "data": {}}

    return fake


def _routing_fake(fake):
    """Execute history commands for real; everything else via `fake`."""

    def side_effect(command, confirmed=None):
        if command.get("intent") in ("history", "clear_history"):
            return real_execute_command(command, confirmed)
        return fake(command, confirmed)

    return side_effect


def _history_files():
    """Return any persisted-history files (there must be none)."""
    names = ("history.json", "history.db", "history.csv", "history.txt",
             "history.sqlite")
    found = []
    for base in (PROJECT_ROOT, PROJECT_ROOT / "config",
                 PROJECT_ROOT / "logs", PROJECT_ROOT / "core",
                 PROJECT_ROOT / "tests"):
        for name in names:
            path = base / name
            if path.exists():
                found.append(path)
    return found


class _MainHistoryTest(unittest.TestCase):
    """Base: isolated session + isolated history for main.py flows."""

    def setUp(self):
        import main

        self.main = main
        self._original_session = main.session
        main.session = SessionState()
        self.history = CommandHistory(max_entries=50)
        set_command_history(self.history)
        self.executed = []
        self.fake = _fake_exec(self.executed)

    def tearDown(self):
        self.main.session = self._original_session
        set_command_history(None)

    def _run_turn(self, text):
        """Run one turn with both executor references mocked."""
        buffer = io.StringIO()
        side_effect = _routing_fake(self.fake)
        with patch("main.get_input", return_value=text), \
                patch("main.execute_command", side_effect=side_effect), \
                patch(
                    "core.multi_command_executor.execute_command",
                    side_effect=side_effect,
                ), \
                redirect_stdout(buffer):
            self.main._process_once()
        return buffer.getvalue()

    def _entries(self):
        return self.history.get_history()


# ---------------------------------------------------------------------------
# Sanitization (PART 22/23)
# ---------------------------------------------------------------------------


class TestSanitizeHistoryText(unittest.TestCase):
    """Deterministic redaction of obvious secrets."""

    def test_password_equals_redacted(self):
        self.assertEqual(
            sanitize_history_text("password=abc123"),
            f"password={REDACTED}",
        )

    def test_password_is_redacted(self):
        """Spec example: the secret value must not survive storage."""
        result = sanitize_history_text("search web my password is abc123")
        self.assertNotIn("abc123", result)
        self.assertEqual(
            result, f"search web my password is {REDACTED}"
        )

    def test_token_colon_redacted(self):
        self.assertEqual(
            sanitize_history_text("token: ghp_12345"),
            f"token: {REDACTED}",
        )

    def test_api_key_variants_redacted(self):
        for text in ("api_key=xyz", "api key = xyz", "apikey: xyz"):
            result = sanitize_history_text(text)
            self.assertNotIn("xyz", result, text)
            self.assertIn(REDACTED, result, text)

    def test_private_key_redacted(self):
        result = sanitize_history_text("private key: AAAAB3")
        self.assertNotIn("AAAAB3", result)

    def test_authorization_redacted_to_end(self):
        result = sanitize_history_text("secret is hunter2 and more")
        self.assertNotIn("hunter2", result)
        self.assertNotIn("and more", result)

    def test_case_insensitive(self):
        result = sanitize_history_text("PASSWORD=abc")
        self.assertEqual(result, f"PASSWORD={REDACTED}")

    def test_normal_commands_unchanged(self):
        for text in (
            "open chrome",
            "search web python tutorials",
            "create folder passwords",
            "open calculator",
            "lock computer",
        ):
            self.assertEqual(sanitize_history_text(text), text)

    def test_secret_named_file_unchanged(self):
        """No separator after the keyword → conservative no-op."""
        for text in (
            "delete file token.txt",
            "open file credentials.json",
            "search file cookie.log",
        ):
            self.assertEqual(sanitize_history_text(text), text)

    def test_keyword_without_value_unchanged(self):
        self.assertEqual(sanitize_history_text("password="), "password=")
        self.assertEqual(sanitize_history_text("password"), "password")

    def test_empty_and_non_string(self):
        self.assertEqual(sanitize_history_text(""), "")
        self.assertEqual(sanitize_history_text(42), 42)
        self.assertIsNone(sanitize_history_text(None))


# ---------------------------------------------------------------------------
# Entry structure (PART 2)
# ---------------------------------------------------------------------------


class TestHistoryEntry(unittest.TestCase):
    """HistoryEntry is a structured, immutable record."""

    def setUp(self):
        self.history = CommandHistory(max_entries=10)
        self.entry = self.history.add_entry(
            input_text="open chrome",
            command_text="open chrome",
            intent="open_application",
            success=True,
            message="Chrome opened successfully.",
        )

    def test_entry_is_a_history_entry(self):
        self.assertIsInstance(self.entry, HistoryEntry)

    def test_field_types(self):
        self.assertIsInstance(self.entry.id, int)
        self.assertIsInstance(self.entry.timestamp, datetime)
        self.assertIsInstance(self.entry.input_text, str)
        self.assertIsInstance(self.entry.command_text, str)
        self.assertEqual(self.entry.intent, "open_application")
        self.assertIs(self.entry.success, True)
        self.assertEqual(self.entry.message, "Chrome opened successfully.")

    def test_timestamp_is_timezone_aware(self):
        self.assertIsNotNone(self.entry.timestamp.tzinfo)

    def test_entry_is_immutable(self):
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            self.entry.success = False

    def test_entry_to_dict_shape(self):
        data = entry_to_dict(self.entry)
        self.assertEqual(
            set(data.keys()),
            {"id", "timestamp", "input", "command", "intent",
             "success", "message"},
        )
        self.assertEqual(data["command"], "open chrome")

    def test_entry_to_dict_returns_fresh_dict(self):
        data = entry_to_dict(self.entry)
        data["command"] = "tampered"
        self.assertEqual(self.entry.command_text, "open chrome")


# ---------------------------------------------------------------------------
# Adding and reading (PART 3/4/7)
# ---------------------------------------------------------------------------


class TestAddAndGet(unittest.TestCase):
    """Entries are added, ordered, and read back chronologically."""

    def setUp(self):
        self.history = CommandHistory(max_entries=100)

    def test_empty_history(self):
        self.assertEqual(self.history.count(), 0)
        self.assertEqual(self.history.get_history(), [])

    def test_add_one_entry(self):
        self.history.add_entry(
            input_text="open chrome", command_text="open chrome"
        )
        self.assertEqual(self.history.count(), 1)

    def test_fields_recorded(self):
        entry = self.history.add_entry(
            input_text="open chrome",
            command_text="open chrome",
            intent="open_application",
            success=True,
            message="Chrome opened successfully.",
        )
        self.assertEqual(entry.id, 1)
        self.assertEqual(entry.input_text, "open chrome")
        self.assertEqual(entry.command_text, "open chrome")
        self.assertEqual(entry.intent, "open_application")
        self.assertIs(entry.success, True)
        self.assertEqual(entry.message, "Chrome opened successfully.")

    def test_failed_command_recorded(self):
        entry = self.history.add_entry(
            input_text="open invalidapp",
            command_text="open invalidapp",
            intent="open_application",
            success=False,
            message="invalidapp is not installed.",
        )
        self.assertIs(entry.success, False)

    def test_multiple_entries_in_order(self):
        for target in ("chrome", "calculator", "notepad"):
            self.history.add_entry(
                input_text=f"open {target}",
                command_text=f"open {target}",
                intent="open_application",
                success=True,
                message=f"{target} opened successfully.",
            )
        entries = self.history.get_history()
        self.assertEqual(
            [e.command_text for e in entries],
            ["open chrome", "open calculator", "open notepad"],
        )
        self.assertEqual([e.id for e in entries], [1, 2, 3])

    def test_chain_entries_share_the_full_input(self):
        """PART 4: each command keeps its own granularity plus the
        original input as context."""
        full = "open chrome and open calculator"
        self.history.add_entry(
            input_text=full, command_text="open chrome",
            intent="open_application", success=True,
            message="Chrome opened successfully.",
        )
        self.history.add_entry(
            input_text=full, command_text="open calculator",
            intent="open_application", success=True,
            message="Calculator opened successfully.",
        )
        entries = self.history.get_history()
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0].input_text, full)
        self.assertEqual(entries[1].input_text, full)
        self.assertEqual(entries[0].command_text, "open chrome")
        self.assertEqual(entries[1].command_text, "open calculator")

    def test_sanitization_applied_on_store(self):
        entry = self.history.add_entry(
            input_text="search web my password is abc123",
            command_text="search web my password is abc123",
            intent="search_web",
            success=True,
            message="Searched for password=abc123",
        )
        self.assertNotIn("abc123", entry.input_text)
        self.assertNotIn("abc123", entry.command_text)
        self.assertNotIn("abc123", entry.message)
        self.assertIn(REDACTED, entry.command_text)

    def test_pathological_input_never_raises(self):
        entry = self.history.add_entry(
            input_text=None,
            command_text=123,
            intent=42,
            success="yes",
            message=object(),
        )
        self.assertEqual(entry.input_text, "")
        self.assertEqual(entry.command_text, "123")
        self.assertEqual(entry.intent, "42")
        self.assertIs(entry.success, True)
        self.assertIsInstance(entry.message, str)
        self.assertEqual(self.history.count(), 1)

    def test_blank_intent_stored_as_none(self):
        entry = self.history.add_entry(
            input_text="x", command_text="x", intent="  "
        )
        self.assertIsNone(entry.intent)


# ---------------------------------------------------------------------------
# Recent entries (PART 8)
# ---------------------------------------------------------------------------


class TestGetRecent(unittest.TestCase):
    """get_recent returns the tail of the history, safely."""

    def setUp(self):
        self.history = CommandHistory(max_entries=100)
        for i in range(5):
            self.history.add_entry(
                input_text=f"cmd{i}", command_text=f"cmd{i}"
            )

    def test_recent_two_returns_latest(self):
        recent = self.history.get_recent(2)
        self.assertEqual([e.command_text for e in recent], ["cmd3", "cmd4"])

    def test_recent_more_than_available(self):
        self.assertEqual(len(self.history.get_recent(99)), 5)

    def test_recent_zero_returns_empty(self):
        self.assertEqual(self.history.get_recent(0), [])

    def test_recent_negative_returns_empty(self):
        self.assertEqual(self.history.get_recent(-3), [])

    def test_recent_non_int_and_bool_return_empty(self):
        self.assertEqual(self.history.get_recent("2"), [])
        self.assertEqual(self.history.get_recent(True), [])
        self.assertEqual(self.history.get_recent(None), [])

    def test_recent_on_empty_history(self):
        self.assertEqual(CommandHistory().get_recent(5), [])


# ---------------------------------------------------------------------------
# Size limit and IDs (PART 5/9/38)
# ---------------------------------------------------------------------------


class TestSizeLimit(unittest.TestCase):
    """History is bounded; the oldest entry is dropped, IDs are not."""

    def test_limit_three_keeps_newest(self):
        history = CommandHistory(max_entries=3)
        for i in range(1, 5):
            history.add_entry(input_text=f"cmd{i}", command_text=f"cmd{i}")
        entries = history.get_history()
        self.assertEqual(history.count(), 3)
        self.assertEqual([e.command_text for e in entries],
                         ["cmd2", "cmd3", "cmd4"])

    def test_ids_not_renumbered_after_trim(self):
        """PART 9: IDs stay unique for the session — 2,3,4 not 1,2,3."""
        history = CommandHistory(max_entries=3)
        for i in range(1, 5):
            history.add_entry(input_text=f"cmd{i}", command_text=f"cmd{i}")
        self.assertEqual([e.id for e in history.get_history()], [2, 3, 4])

    def test_trim_logs_warning(self):
        history = CommandHistory(max_entries=1)
        with self.assertLogs("core.command_history", level="WARNING") as cap:
            history.add_entry(input_text="a", command_text="a")
            history.add_entry(input_text="b", command_text="b")
        self.assertTrue(
            any("History limit reached" in line for line in cap.output)
        )

    def test_valid_explicit_limit_kept(self):
        self.assertEqual(CommandHistory(max_entries=42).max_entries, 42)

    def test_invalid_explicit_limits_fall_back_to_default(self):
        for bad in (0, -5, "100", 1.5, True, None and 3):
            if bad is None:
                continue
            with self.assertLogs("core.command_history", level="WARNING"):
                history = CommandHistory(max_entries=bad)
            self.assertEqual(history.max_entries, DEFAULT_MAX_ENTRIES, bad)

    def test_explicit_limit_clamped_to_hard_max(self):
        self.assertEqual(
            CommandHistory(max_entries=HARD_MAX_ENTRIES + 1).max_entries,
            HARD_MAX_ENTRIES,
        )

    def test_config_value_used(self):
        with patch("core.command_history.get_value", return_value=7):
            self.assertEqual(CommandHistory().max_entries, 7)

    def test_malformed_config_falls_back(self):
        for bad in (0, -1, "big", True, 3.5):
            with patch("core.command_history.get_value", return_value=bad):
                with self.assertLogs(
                    "core.command_history", level="WARNING"
                ):
                    history = CommandHistory()
            self.assertEqual(history.max_entries, DEFAULT_MAX_ENTRIES, bad)

    def test_huge_config_clamped(self):
        with patch(
            "core.command_history.get_value", return_value=10_000_000
        ):
            history = CommandHistory()
        self.assertEqual(history.max_entries, HARD_MAX_ENTRIES)

    def test_default_limit_is_100(self):
        self.assertEqual(DEFAULT_MAX_ENTRIES, 100)
        with patch("core.command_history.get_value", return_value=100):
            self.assertEqual(CommandHistory().max_entries, 100)


# ---------------------------------------------------------------------------
# Clearing (PART 10)
# ---------------------------------------------------------------------------


class TestClearHistory(unittest.TestCase):
    """clear_history empties the list but never resets IDs."""

    def setUp(self):
        self.history = CommandHistory(max_entries=10)

    def test_clear_removes_everything(self):
        for i in range(3):
            self.history.add_entry(input_text=f"c{i}", command_text=f"c{i}")
        cleared = self.history.clear_history()
        self.assertEqual(cleared, 3)
        self.assertEqual(self.history.count(), 0)
        self.assertEqual(self.history.get_history(), [])

    def test_clear_empty_is_harmless(self):
        self.assertEqual(self.history.clear_history(), 0)
        self.assertEqual(self.history.count(), 0)

    def test_next_id_continues_after_clear(self):
        """PART 10: IDs stay unique across the whole session."""
        self.history.add_entry(input_text="a", command_text="a")
        self.history.add_entry(input_text="b", command_text="b")
        self.history.clear_history()
        entry = self.history.add_entry(input_text="c", command_text="c")
        self.assertEqual(entry.id, 3)

    def test_clear_logs_info(self):
        self.history.add_entry(input_text="a", command_text="a")
        with self.assertLogs("core.command_history", level="INFO") as cap:
            self.history.clear_history()
        self.assertTrue(
            any("Command history cleared" in line for line in cap.output)
        )

    def test_clear_does_not_touch_disk(self):
        self.history.add_entry(input_text="a", command_text="a")
        self.history.clear_history()
        self.assertEqual(_history_files(), [])


# ---------------------------------------------------------------------------
# Immutability (PART 7/42)
# ---------------------------------------------------------------------------


class TestImmutability(unittest.TestCase):
    """Callers can never corrupt the internal history."""

    def setUp(self):
        self.history = CommandHistory(max_entries=10)
        self.history.add_entry(input_text="open chrome",
                               command_text="open chrome",
                               intent="open_application", success=True,
                               message="ok")
        self.history.add_entry(input_text="open calc",
                               command_text="open calc",
                               intent="open_application", success=True,
                               message="ok")

    def test_returned_list_is_a_copy(self):
        entries = self.history.get_history()
        entries.clear()
        self.assertEqual(self.history.count(), 2)
        self.assertEqual(len(self.history.get_history()), 2)

    def test_each_call_returns_a_new_list(self):
        first = self.history.get_history()
        second = self.history.get_history()
        self.assertIsNot(first, second)

    def test_entries_are_frozen(self):
        entry = self.history.get_history()[0]
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            entry.command_text = "tampered"
        self.assertEqual(
            self.history.get_history()[0].command_text, "open chrome"
        )

    def test_get_recent_is_a_copy(self):
        recent = self.history.get_recent(1)
        recent.clear()
        self.assertEqual(self.history.count(), 2)

    def test_mutating_dict_copy_does_not_affect_history(self):
        entry = self.history.get_history()[0]
        data = entry_to_dict(entry)
        data["command"] = "tampered"
        data["success"] = False
        self.assertEqual(entry.command_text, "open chrome")
        self.assertIs(entry.success, True)

    def test_get_entry_returns_immutable_entry(self):
        entry = self.history.get_entry(1)
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            entry.intent = "unknown"


# ---------------------------------------------------------------------------
# Outcome updates (PART 28 support)
# ---------------------------------------------------------------------------


class TestUpdateOutcome(unittest.TestCase):
    """A parked entry is resolved to its real outcome later."""

    def setUp(self):
        self.history = CommandHistory(max_entries=10)
        self.pending = self.history.add_entry(
            input_text="shutdown computer",
            command_text="shutdown computer",
            intent="shutdown_computer",
            success=None,
            message="Are you sure you want to shut down the computer?",
        )

    def test_resolve_to_success(self):
        updated = self.history.update_outcome(
            self.pending.id, True, "Computer shutting down."
        )
        self.assertIs(updated.success, True)
        self.assertEqual(updated.message, "Computer shutting down.")
        self.assertIs(self.history.get_history()[0].success, True)

    def test_resolve_to_failure(self):
        updated = self.history.update_outcome(
            self.pending.id, False, "Shutdown cancelled."
        )
        self.assertIs(updated.success, False)
        self.assertEqual(updated.message, "Shutdown cancelled.")

    def test_message_none_keeps_existing(self):
        updated = self.history.update_outcome(self.pending.id, False, None)
        self.assertEqual(
            updated.message, "Are you sure you want to shut down the computer?"
        )

    def test_unknown_id_returns_none(self):
        self.assertIsNone(self.history.update_outcome(999, True, "x"))
        self.assertIsNone(self.history.update_outcome(True, True, "x"))
        self.assertEqual(self.history.count(), 1)

    def test_update_does_not_touch_other_entries(self):
        other = self.history.add_entry(
            input_text="open chrome", command_text="open chrome",
            intent="open_application", success=True, message="ok",
        )
        self.history.update_outcome(self.pending.id, True, "done")
        entries = self.history.get_history()
        self.assertIs(entries[1].success, True)
        self.assertEqual(entries[1].id, other.id)
        self.assertEqual(entries[1].message, "ok")

    def test_update_message_is_sanitized(self):
        updated = self.history.update_outcome(
            self.pending.id, True, "done with token=abc123"
        )
        self.assertNotIn("abc123", updated.message)


# ---------------------------------------------------------------------------
# Structured result (PART 21)
# ---------------------------------------------------------------------------


class TestStructuredResult(unittest.TestCase):
    """get_history_result follows the {success, message, data} shape."""

    def test_empty_history_result(self):
        result = CommandHistory().get_history_result()
        self.assertTrue(result["success"])
        self.assertEqual(result["message"], "There is no command history yet.")
        self.assertEqual(result["data"]["kind"], "history")
        self.assertEqual(result["data"]["entries"], [])

    def test_non_empty_history_result(self):
        history = CommandHistory()
        history.add_entry(
            input_text="open chrome", command_text="open chrome",
            intent="open_application", success=True, message="ok",
        )
        result = history.get_history_result()
        self.assertTrue(result["success"])
        self.assertEqual(result["message"], "Command history retrieved.")
        entries = result["data"]["entries"]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["id"], 1)
        self.assertEqual(entries[0]["command"], "open chrome")

    def test_entries_are_plain_dicts(self):
        history = CommandHistory()
        history.add_entry(input_text="x", command_text="x")
        entries = history.get_history_result()["data"]["entries"]
        for entry in entries:
            self.assertIsInstance(entry, dict)
            self.assertNotIsInstance(entry, HistoryEntry)

    def test_result_never_contains_session_internals(self):
        history = CommandHistory()
        history.add_entry(input_text="x", command_text="x")
        text = format_response(history.get_history_result())
        self.assertNotIn("session", text.lower())
        self.assertNotIn("HistoryEntry", text)
        self.assertNotIn("timestamp", text)


# ---------------------------------------------------------------------------
# Accessors and the recording helper
# ---------------------------------------------------------------------------


class TestAccessorsAndRecord(unittest.TestCase):
    """Session-wide accessor and the never-raising record helper."""

    def tearDown(self):
        set_command_history(None)

    def test_exempt_intents(self):
        self.assertEqual(
            HISTORY_EXEMPT_INTENTS, frozenset({"history", "clear_history"})
        )

    def test_get_creates_on_first_use(self):
        set_command_history(None)
        history = get_command_history()
        self.assertIsInstance(history, CommandHistory)
        self.assertIs(get_command_history(), history)

    def test_set_installs_instance(self):
        mine = CommandHistory(max_entries=5)
        set_command_history(mine)
        self.assertIs(get_command_history(), mine)

    def test_record_entry_with_none_history_is_noop(self):
        record_entry(None, input_text="x", command_text="x")

    def test_record_entry_adds(self):
        history = CommandHistory()
        record_entry(
            history, input_text="x", command_text="x",
            intent="unknown", success=False, message="nope",
        )
        self.assertEqual(history.count(), 1)

    def test_record_entry_never_raises(self):
        history = CommandHistory()
        with patch.object(
            history, "add_entry", side_effect=RuntimeError("boom")
        ):
            record_entry(history, input_text="x", command_text="x")

    def test_logs_never_contain_command_content(self):
        """PART 37: logs get IDs and counts — never the history text
        or any secret in it."""
        history = CommandHistory()
        with self.assertLogs("core.command_history", level="DEBUG") as cap:
            history.add_entry(
                input_text="search web password=abc123",
                command_text="search web password=abc123",
                intent="search_web",
                success=True,
                message="ok",
            )
            history.get_history()
        joined = "\n".join(cap.output)
        self.assertIn("History entry added", joined)
        self.assertIn("History entries retrieved", joined)
        self.assertNotIn("abc123", joined)
        self.assertNotIn("password", joined)
        self.assertNotIn("search web", joined)


# ---------------------------------------------------------------------------
# No persistence (PART 36)
# ---------------------------------------------------------------------------


class TestNoPersistence(unittest.TestCase):
    """History exists only in memory."""

    def test_no_history_files_created(self):
        history = CommandHistory()
        history.add_entry(input_text="open chrome",
                          command_text="open chrome")
        history.get_history()
        history.clear_history()
        self.assertEqual(_history_files(), [])

    def test_module_has_no_storage_libraries(self):
        import core.command_history as module

        self.assertNotIn("json", vars(module))
        self.assertNotIn("sqlite3", vars(module))
        self.assertNotIn("csv", vars(module))


# ---------------------------------------------------------------------------
# Executor routing (PART 17)
# ---------------------------------------------------------------------------


class TestExecutorHistoryRouting(unittest.TestCase):
    """HISTORY and CLEAR_HISTORY route to Command History only."""

    def setUp(self):
        self.history = CommandHistory(max_entries=20)
        set_command_history(self.history)

    def tearDown(self):
        set_command_history(None)

    def _command(self, intent):
        return {"intent": intent, "target": None, "arguments": {}}

    def test_history_returns_structured_result(self):
        self.history.add_entry(
            input_text="open chrome", command_text="open chrome",
            intent="open_application", success=True, message="ok",
        )
        result = execute_command(self._command("history"))
        self.assertTrue(result["success"])
        self.assertEqual(result["data"]["kind"], "history")
        self.assertEqual(len(result["data"]["entries"]), 1)

    def test_history_empty_message(self):
        result = execute_command(self._command("history"))
        self.assertEqual(result["message"], "There is no command history yet.")
        self.assertEqual(result["data"]["entries"], [])

    def test_clear_history_routes_and_clears(self):
        self.history.add_entry(input_text="a", command_text="a")
        result = execute_command(self._command("clear_history"))
        self.assertTrue(result["success"])
        self.assertEqual(result["message"], "Command history cleared.")
        self.assertEqual(result["data"], {})
        self.assertEqual(self.history.count(), 0)

    def test_history_is_read_only(self):
        self.history.add_entry(input_text="a", command_text="a")
        execute_command(self._command("history"))
        execute_command(self._command("history"))
        self.assertEqual(self.history.count(), 1)
        self.assertEqual(self.history.get_history()[0].command_text, "a")

    def test_history_makes_no_shell_calls(self):
        self.history.add_entry(input_text="a", command_text="a")
        with patch("os.system") as mock_system, \
                patch("subprocess.run") as mock_run, \
                patch("subprocess.Popen") as mock_popen:
            execute_command(self._command("history"))
            execute_command(self._command("clear_history"))
        self.assertEqual(
            mock_system.call_count + mock_run.call_count
            + mock_popen.call_count,
            0,
        )

    def test_clear_history_deletes_no_files(self):
        self.history.add_entry(input_text="a", command_text="a")
        with patch("os.remove") as mock_remove, \
                patch("os.unlink") as mock_unlink:
            execute_command(self._command("clear_history"))
        self.assertEqual(
            mock_remove.call_count + mock_unlink.call_count, 0
        )
        self.assertEqual(_history_files(), [])


# ---------------------------------------------------------------------------
# main.py single-command history (PART 25/26/34)
# ---------------------------------------------------------------------------


class TestMainSingleHistoryFlow(_MainHistoryTest):
    """Single commands are recorded; history commands are exempt."""

    def test_successful_command_recorded(self):
        self._run_turn("open chrome")
        entries = self._entries()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].command_text, "open chrome")
        self.assertEqual(entries[0].input_text, "open chrome")
        self.assertEqual(entries[0].intent, "open_application")
        self.assertIs(entries[0].success, True)

    def test_unknown_attempt_recorded_as_failed(self):
        """PART 26: history represents what the user attempted."""
        output = self._run_turn("blah blah")
        self.assertIn("Unknown command.", output)
        entries = self._entries()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].command_text, "blah blah")
        self.assertEqual(entries[0].intent, "unknown")
        self.assertIs(entries[0].success, False)

    def test_validation_failure_recorded(self):
        output = self._run_turn("open")
        self.assertIn("requires a target", output)
        entries = self._entries()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].command_text, "open")
        self.assertIs(entries[0].success, False)
        self.assertEqual(entries[0].message, "Open Application requires a target.")

    def test_empty_turn_records_nothing(self):
        self._run_turn("")
        self.assertEqual(self.history.count(), 0)

    def test_history_command_not_recorded_and_shows_list(self):
        """PART 34: open chrome, open calculator, history → 2 entries."""
        self._run_turn("open chrome")
        self._run_turn("open calculator")
        output = self._run_turn("history")
        self.assertIn("1. open chrome", output)
        self.assertIn("2. open calculator", output)
        self.assertNotIn("3. history", output)
        self.assertEqual(self.history.count(), 2)

    def test_clear_history_command_not_recorded(self):
        self._run_turn("open chrome")
        output = self._run_turn("clear history")
        self.assertIn("Command history cleared.", output)
        self.assertEqual(self.history.count(), 0)
        # And clearing does not immediately record itself.
        self.assertEqual(self._entries(), [])

    def test_history_after_clear_says_empty(self):
        self._run_turn("open chrome")
        self._run_turn("clear history")
        output = self._run_turn("history")
        self.assertIn("There is no command history yet.", output)
        self.assertNotIn("[", output)

    def test_history_display_order(self):
        self._run_turn("open chrome")
        self._run_turn("open calculator")
        self._run_turn("search web python tutorials")
        output = self._run_turn("history")
        lines = [line for line in output.splitlines()
                 if line and line[0].isdigit()]
        self.assertEqual(
            lines,
            [
                "1. open chrome",
                "2. open calculator",
                "3. search web python tutorials",
            ],
        )

    def test_multiple_commands_in_one_session_accumulate(self):
        self._run_turn("open chrome")
        self._run_turn("lock computer")
        self.assertEqual(
            [e.command_text for e in self._entries()],
            ["open chrome", "lock computer"],
        )
        self.assertEqual([e.id for e in self._entries()], [1, 2])


# ---------------------------------------------------------------------------
# main.py confirmation integration (PART 28/29)
# ---------------------------------------------------------------------------


class TestMainConfirmationHistory(_MainHistoryTest):
    """Pending commands are recorded truthfully and resolved later."""

    def test_parked_command_recorded_as_awaiting(self):
        self._run_turn("shutdown computer")
        entries = self._entries()
        self.assertEqual(len(entries), 1)
        self.assertIsNone(entries[0].success)
        self.assertEqual(entries[0].intent, "shutdown_computer")
        self.assertIn("Are you sure", entries[0].message)
        self.assertEqual(self.executed, [])

    def test_yes_resolves_entry_to_success(self):
        self._run_turn("shutdown computer")
        output = self._run_turn("yes")
        self.assertIn("Computer shutting down.", output)
        entries = self._entries()
        # No "yes" entry — the parked entry is updated in place.
        self.assertEqual(len(entries), 1)
        self.assertIs(entries[0].success, True)
        self.assertEqual(entries[0].message, "Computer shutting down.")
        self.assertEqual(len(self.executed), 1)
        self.assertIs(self.executed[0]["confirmed"], True)

    def test_no_resolves_entry_to_cancelled(self):
        self._run_turn("shutdown computer")
        output = self._run_turn("no")
        self.assertIn("Shutdown cancelled.", output)
        entries = self._entries()
        self.assertEqual(len(entries), 1)
        self.assertIs(entries[0].success, False)
        self.assertEqual(entries[0].message, "Shutdown cancelled.")
        self.assertEqual(self.executed, [])

    def test_expired_resolves_entry_to_failed(self):
        self._run_turn("shutdown computer")
        # Force the confirmation past its deadline.
        self.main.session._pending_expires_at = time.monotonic() - 1
        output = self._run_turn("yes")
        self.assertIn("That confirmation has expired.", output)
        entries = self._entries()
        self.assertEqual(len(entries), 1)
        self.assertIs(entries[0].success, False)
        self.assertEqual(entries[0].message, "That confirmation has expired.")
        self.assertEqual(self.executed, [])

    def test_unrelated_input_resolves_and_records_new_command(self):
        self._run_turn("shutdown computer")
        self._run_turn("help")
        entries = self._entries()
        # The parked entry resolved to cancelled; help recorded after.
        self.assertEqual(len(entries), 2)
        self.assertIs(entries[0].success, False)
        self.assertEqual(entries[0].message, "Confirmation cancelled.")
        self.assertEqual(entries[1].command_text, "help")
        self.assertIs(entries[1].success, True)
        # Only help ran — the parked shutdown never executed.
        self.assertEqual(
            [c["intent"] for c in self.executed], ["help"]
        )

    def test_nothing_to_confirm_changes_nothing(self):
        self._run_turn("open chrome")
        before = len(self._entries())
        self._run_turn("yes")
        self.assertEqual(len(self._entries()), before)
        self.assertEqual(len(self.executed), 1)


# ---------------------------------------------------------------------------
# main.py multi-command history (PART 4/27/32)
# ---------------------------------------------------------------------------


class TestMainMultiHistory(_MainHistoryTest):
    """Chains record one entry per processed command."""

    def test_chain_records_each_command(self):
        self._run_turn("open chrome and open calculator")
        entries = self._entries()
        self.assertEqual(len(entries), 2)
        self.assertEqual(
            [e.command_text for e in entries],
            ["open chrome", "open calculator"],
        )
        full = "open chrome and open calculator"
        self.assertEqual(entries[0].input_text, full)
        self.assertEqual(entries[1].input_text, full)
        self.assertEqual([e.id for e in entries], [1, 2])

    def test_pre_validation_failure_records_only_the_invalid_command(self):
        """Nothing executed → only the command that failed planning."""
        self._run_turn("open chrome and open")
        entries = self._entries()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].command_text, "open")
        self.assertIs(entries[0].success, False)
        self.assertEqual(self.executed, [])

    def test_execution_failure_records_individual_results(self):
        """PART 27: every command gets its own entry and result —
        the failed last command stops the chain with success=False.

        (The last target is 'calculator' rather than a made-up app
        name because the spell checker normalizes unknown words — a
        known quirk outside this step's scope.)
        """
        fake = _fake_exec(self.executed, fail_targets=("calculator",))
        side_effect = _routing_fake(fake)
        buffer = io.StringIO()
        with patch("main.get_input",
                   return_value="open chrome and open firefox and open calculator"), \
                patch("main.execute_command", side_effect=side_effect), \
                patch("core.multi_command_executor.execute_command",
                      side_effect=side_effect), \
                redirect_stdout(buffer):
            self.main._process_once()
        entries = self._entries()
        self.assertEqual(
            [e.command_text for e in entries],
            ["open chrome", "open firefox", "open calculator"],
        )
        self.assertIs(entries[0].success, True)
        self.assertIs(entries[1].success, True)
        self.assertIs(entries[2].success, False)

    def test_chain_records_before_stopping_on_failure(self):
        fake = _fake_exec(self.executed, fail_targets=("firefox",))
        side_effect = _routing_fake(fake)
        buffer = io.StringIO()
        with patch("main.get_input",
                   return_value="open chrome and open firefox and open notepad"), \
                patch("main.execute_command", side_effect=side_effect), \
                patch("core.multi_command_executor.execute_command",
                      side_effect=side_effect), \
                redirect_stdout(buffer):
            self.main._process_once()
        entries = self._entries()
        # Notepad never ran → no entry; firefox's failure is recorded.
        self.assertEqual(
            [e.command_text for e in entries],
            ["open chrome", "open firefox"],
        )
        self.assertIs(entries[1].success, False)

    def test_history_command_in_chain_is_exempt(self):
        self._run_turn("open chrome and history")
        entries = self._entries()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].command_text, "open chrome")

    def test_clear_history_in_chain_is_exempt(self):
        self._run_turn("open chrome")
        output = self._run_turn("open calculator and clear history")
        self.assertIn("Command history cleared.", output)
        self.assertEqual(self.history.count(), 0)

    def test_chain_parks_dangerous_command_as_awaiting(self):
        self._run_turn("open chrome and shutdown computer")
        entries = self._entries()
        self.assertEqual(len(entries), 2)
        self.assertIs(entries[0].success, True)
        self.assertIsNone(entries[1].success)
        self.assertEqual(entries[1].command_text, "shutdown computer")

    def test_chain_confirmation_round_trip_resolves_entry(self):
        self._run_turn("open chrome and shutdown computer")
        self._run_turn("yes")
        entries = self._entries()
        self.assertEqual(len(entries), 2)
        self.assertIs(entries[0].success, True)
        self.assertIs(entries[1].success, True)
        self.assertEqual(entries[1].message, "Computer shutting down.")
        intents = [c["intent"] for c in self.executed]
        self.assertEqual(intents, ["open_application", "shutdown_computer"])

    def test_chain_confirmation_declined_resolves_entry(self):
        self._run_turn("open chrome and shutdown computer")
        output = self._run_turn("no")
        self.assertIn("Shutdown cancelled.", output)
        entries = self._entries()
        self.assertIs(entries[1].success, False)
        self.assertEqual(entries[1].message, "Shutdown cancelled.")
        self.assertEqual(
            [c["intent"] for c in self.executed], ["open_application"]
        )


# ---------------------------------------------------------------------------
# Security (PART 22/24/44/45)
# ---------------------------------------------------------------------------


class TestHistorySecurity(_MainHistoryTest):
    """History never executes, replays, persists, or exposes secrets."""

    def test_history_never_replays_commands(self):
        """'history' is read-only: stored commands never run again."""
        self._run_turn("open chrome")
        self.assertEqual(len(self.executed), 1)
        self._run_turn("history")
        self._run_turn("command history")
        self._run_turn("recent commands")
        self.assertEqual(len(self.executed), 1)

    def test_repeat_command_requests_are_not_replayed(self):
        """PART 45: replay commands are not implemented — they fail
        as unknown commands and execute nothing."""
        self._run_turn("open chrome")
        for text in ("repeat last command", "run command 3",
                     "execute history 1"):
            output = self._run_turn(text)
            self.assertIn("Unknown command.", output)
        self.assertEqual(len(self.executed), 1)

    def test_history_makes_no_shell_calls(self):
        self._run_turn("open chrome")
        buffer = io.StringIO()
        with patch("main.get_input", return_value="history"), \
                patch("os.system") as mock_system, \
                patch("subprocess.run") as mock_run, \
                patch("subprocess.Popen") as mock_popen, \
                redirect_stdout(buffer):
            self.main._process_once()
        self.assertEqual(
            mock_system.call_count + mock_run.call_count
            + mock_popen.call_count,
            0,
        )

    def test_clear_history_never_touches_files(self):
        self._run_turn("open chrome")
        buffer = io.StringIO()
        with patch("main.get_input", return_value="clear history"), \
                patch("os.remove") as mock_remove, \
                patch("os.unlink") as mock_unlink, \
                patch("os.rmdir") as mock_rmdir, \
                redirect_stdout(buffer):
            self.main._process_once()
        self.assertEqual(
            mock_remove.call_count + mock_unlink.call_count
            + mock_rmdir.call_count,
            0,
        )
        self.assertEqual(self.history.count(), 0)

    def test_secret_is_executed_raw_but_stored_redacted(self):
        """PART 23/24: history sanitization never alters execution."""
        self._run_turn("search web my password is abc123")
        # The executor received the raw query unchanged...
        self.assertEqual(
            self.executed[0]["target"], "my password is abc123"
        )
        # ...but the stored history holds no secret.
        entry = self._entries()[0]
        self.assertNotIn("abc123", entry.input_text)
        self.assertNotIn("abc123", entry.command_text)
        self.assertIn(REDACTED, entry.command_text)
        # And the displayed history is redacted too.
        output = self._run_turn("history")
        self.assertNotIn("abc123", output)

    def test_history_does_not_bypass_confirmation(self):
        """A pending dangerous command is never executed by asking
        for history instead of answering."""
        self._run_turn("shutdown computer")
        self._run_turn("history")  # unrelated input cancels safely
        self.assertEqual(self.executed, [])
        entries = self._entries()
        self.assertIs(entries[0].success, False)
        self.assertEqual(entries[0].message, "Confirmation cancelled.")

    def test_history_turn_leaves_session_state_intact(self):
        self._run_turn("open chrome")
        snap_before = self.main.session.snapshot()
        self._run_turn("history")
        snap_after = self.main.session.snapshot()
        # History is a separate store — it does not duplicate history
        # fields into Session State (PART 30).
        self.assertEqual(
            set(snap_before.keys()), set(snap_after.keys())
        )
        self.assertNotIn("history", snap_after)


if __name__ == "__main__":
    unittest.main()
