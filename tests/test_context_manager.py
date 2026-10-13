"""Focused tests for bounded, typed session context."""

from core.context_manager import ContextManager
from core.multi_command_executor import run_multi_commands
from core.session_state import SessionState
from unittest.mock import patch


def _record(manager, intent, target, success=True):
    manager.record_success(
        {"intent": intent, "target": target, "arguments": {}},
        {"success": success},
    )


def test_application_reference_is_typed_and_latest_wins():
    manager = ContextManager()
    _record(manager, "OPEN_APPLICATION", "chrome")
    manager.advance()
    _record(manager, "OPEN_APPLICATION", "calculator")

    resolved = manager.resolve_reference("close it")
    assert resolved is not None
    assert (resolved.intent, resolved.target) == ("CLOSE_APPLICATION", "calculator")


def test_failed_command_does_not_replace_context():
    manager = ContextManager()
    _record(manager, "OPEN_APPLICATION", "chrome")
    manager.advance()
    _record(manager, "OPEN_APPLICATION", "calculator", success=False)

    assert manager.resolve_reference("close it").target == "chrome"


def test_delete_reference_only_resolves_a_file():
    manager = ContextManager()
    _record(manager, "OPEN_APPLICATION", "chrome")
    assert manager.resolve_reference("delete it") is None

    _record(manager, "OPEN_FILE", "notes.txt")
    resolved = manager.resolve_reference("delete it")
    assert resolved is not None
    assert (resolved.intent, resolved.target) == ("DELETE_FILE", "notes.txt")


def test_typed_open_reference_does_not_cross_entity_types():
    manager = ContextManager()
    _record(manager, "OPEN_APPLICATION", "chrome")
    assert manager.resolve_reference("open the file") is None
    assert manager.resolve_reference("close it").target == "chrome"


def test_search_again_uses_last_successful_query():
    manager = ContextManager()
    _record(manager, "SEARCH_WEB", "python tutorials")
    result = manager.resolve_reference("search again")
    assert result is not None
    assert result.intent == "SEARCH_WEB"
    assert result.arguments == {"query": "python tutorials"}


def test_created_folder_can_be_reopened_but_not_deleted_as_a_file():
    manager = ContextManager()
    _record(manager, "CREATE_FOLDER", "projects")
    opened = manager.resolve_reference("open it")
    assert opened is not None
    assert (opened.intent, opened.target) == ("OPEN_FILE", "projects")
    assert manager.resolve_reference("delete it") is None


def test_context_is_bounded_and_expires():
    manager = ContextManager()
    manager.max_items = 2
    manager.max_age_commands = 3
    _record(manager, "OPEN_APPLICATION", "chrome")
    manager.advance()
    _record(manager, "OPEN_APPLICATION", "calculator")
    manager.advance()
    _record(manager, "OPEN_APPLICATION", "notepad")
    assert len(manager.snapshot()) == 2
    manager.advance()
    manager.advance()
    manager.advance()
    assert manager.snapshot() == []


def test_snapshot_is_detached_and_clear_removes_context():
    manager = ContextManager()
    _record(manager, "OPEN_APPLICATION", "chrome")
    snapshot = manager.snapshot()
    snapshot[0]["name"] = "changed"
    assert manager.snapshot()[0]["name"] == "chrome"
    manager.clear()
    assert manager.snapshot() == []


def test_multi_command_context_resolves_sequentially():
    session = SessionState()
    calls = []

    def successful(command):
        calls.append(command.copy())
        return {"success": True, "message": "ok", "data": {}}

    with patch("core.multi_command_executor.execute_command", successful):
        outcome = run_multi_commands(
            ["open chrome", "close it"], context_manager=session.context
        )

    assert outcome.success
    assert [call["intent"] for call in calls] == [
        "open_application", "close_application"
    ]
    assert calls[1]["target"] == "chrome"
