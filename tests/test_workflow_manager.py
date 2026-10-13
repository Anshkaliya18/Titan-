from __future__ import annotations

import copy
import json

import pytest

import core.task_planner as planner_module
import core.workflow_manager as workflow_module
from core.task_planner import TaskPlanner
from core.workflow_manager import WorkflowError, WorkflowManager


@pytest.fixture
def setup_workflows(tmp_path, monkeypatch):
    settings = {
        "automation": {
            "enabled": True, "max_workflows": 50,
            "max_steps_per_workflow": 10,
            "max_workflow_name_length": 100,
            "max_description_length": 500,
        },
        "task_planning": {"enabled": True, "max_steps": 10,
                          "max_plan_input_length": 2000,
                          "allow_llm_planning": False,
                          "require_confirmation_for_plan": False},
    }

    def get_value(section, key, default=None):
        return settings.get(section, {}).get(key, default)

    monkeypatch.setattr(workflow_module, "get_value", get_value)
    monkeypatch.setattr(planner_module, "get_value", get_value)
    planner = TaskPlanner()
    manager = WorkflowManager(tmp_path / "workflows.json", planner)
    return manager, planner, settings


def command(intent="OPEN_APPLICATION", target="chrome"):
    return {"intent": intent, "target": target, "arguments": {}}


def test_create_persist_reload_list_show_update_delete(setup_workflows):
    manager, _, _ = setup_workflows
    created = manager.create_workflow("  Morning   Routine ", [command()], description="open chrome")
    assert created["name"] == "Morning Routine"
    assert created["enabled"] is True and created["schedule"] is None
    reloaded = WorkflowManager(manager.storage_path).list_workflows()
    assert reloaded[0]["id"] == created["id"]
    changed = manager.update_workflow("morning routine", description="open my browser")
    assert changed["description"] == "open my browser"
    assert manager.disable_workflow(created["id"])["enabled"] is False
    assert manager.enable_workflow("Morning Routine")["enabled"] is True
    assert manager.delete_workflow("morning routine") is True
    assert manager.get_workflow(created["id"]) is None


@pytest.mark.parametrize("name", ["", "   ", "a\nb", "password: abc"])
def test_rejects_invalid_or_secret_like_name(setup_workflows, name):
    manager, _, _ = setup_workflows
    with pytest.raises(WorkflowError):
        manager.create_workflow(name, [command()])


def test_rejects_duplicate_name_empty_or_oversized_workflow(setup_workflows):
    manager, _, settings = setup_workflows
    manager.create_workflow("Morning Routine", [command()])
    with pytest.raises(WorkflowError, match="already exists"):
        manager.create_workflow(" morning   routine ", [command()])
    with pytest.raises(WorkflowError, match="at least one"):
        manager.create_workflow("Empty", [])
    settings["automation"]["max_steps_per_workflow"] = 1
    with pytest.raises(WorkflowError, match="step limit"):
        manager.create_workflow("Too many", [command(), command("OPEN_APPLICATION", "notepad")])


@pytest.mark.parametrize("bad_step", [
    {"intent": "RUN_SHELL", "target": "powershell", "arguments": {"command": "whoami"}},
    {"intent": "OPEN_APPLICATION", "target": "C:\\evil.exe", "arguments": {}},
    {"intent": "OPEN_URL", "target": "file:///secret", "arguments": {}},
    {"intent": "OPEN_APPLICATION", "target": "chrome", "arguments": {"python": "exec('x')"}},
    {"intent": "OPEN_APPLICATION", "target": "chrome", "arguments": {}, "confirmed": True},
])
def test_rejects_unsafe_command_data(setup_workflows, bad_step):
    manager, _, _ = setup_workflows
    with pytest.raises(WorkflowError):
        manager.create_workflow("Bad", [bad_step])


def test_create_save_is_atomic_and_bounded(setup_workflows):
    manager, _, _ = setup_workflows
    manager.create_workflow("One", [command()])
    assert list(manager.storage_path.parent.glob(".workflows-*.tmp")) == []
    manager.storage_path.write_text("x" * (workflow_module.MAX_STORAGE_BYTES + 1), encoding="utf-8")
    with pytest.raises(WorkflowError, match="size limit"):
        manager.list_workflows()


def test_corrupted_and_tampered_storage_is_rejected_without_execution(setup_workflows):
    manager, _, _ = setup_workflows
    manager.create_workflow("Trusted", [command()])
    payload = json.loads(manager.storage_path.read_text(encoding="utf-8"))
    payload["workflows"][0]["steps"][0] = {
        "intent": "RUN_SHELL", "target": "cmd.exe", "arguments": {"command": "whoami"}
    }
    manager.storage_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(WorkflowError, match="validation failed"):
        manager.get_workflow("Trusted")
    manager.storage_path.write_text("{broken", encoding="utf-8")
    with pytest.raises(WorkflowError, match="corrupted"):
        manager.list_workflows()


def test_enable_and_run_revalidate_and_require_feature_enabled(setup_workflows):
    manager, _, settings = setup_workflows
    manager.create_workflow("Dev", [command()])
    settings["automation"]["enabled"] = False
    with pytest.raises(WorkflowError, match="disabled"):
        manager.run_workflow("Dev")
    settings["automation"]["enabled"] = True
    plan = manager.run_workflow("Dev")["plan"]
    assert plan.steps[0].command == command()
    manager.disable_workflow("Dev")
    with pytest.raises(WorkflowError, match="workflow is disabled"):
        manager.run_workflow("Dev")


def test_workflow_plan_runs_sequentially_through_existing_executor(monkeypatch, setup_workflows):
    from core.command_history import CommandHistory
    from core.context_manager import ContextManager
    from core.multi_command_executor import run_structured_commands

    manager, _, _ = setup_workflows
    manager.create_workflow("Dev", [command(), command("OPEN_APPLICATION", "notepad")])
    plan = manager.run_workflow("Dev")["plan"]
    calls = []

    def execute(item):
        calls.append(item["target"])
        return {"success": True, "message": "opened", "data": {}}

    monkeypatch.setattr("core.multi_command_executor.execute_command", execute)
    outcome = run_structured_commands(
        [{"text": step.source_text, "command": step.command} for step in plan.steps],
        history=CommandHistory(max_entries=10),
        input_text="run workflow Dev", context_manager=ContextManager(),
    )
    assert outcome.success
    assert calls == ["chrome", "notepad"]
    assert outcome.executed_count == 2


def test_explicit_workflow_cli_parser_and_coordinated_action(monkeypatch, setup_workflows):
    import main

    manager, planner, _ = setup_workflows
    monkeypatch.setattr(main, "workflow_manager", manager)
    monkeypatch.setattr(main, "task_planner", planner)
    monkeypatch.setattr(main, "get_value", lambda section, key, default=None: True if (section, key) == ("automation", "enabled") else default)
    output = []
    monkeypatch.setattr(main, "_display_response", output.append)
    request = main._parse_workflow_request(
        "Create a workflow called development setup that opens Chrome and GitHub."
    )
    assert request == ("create", "development setup", "opens Chrome and GitHub")
    main._handle_workflow_request(request, "create workflow development setup")
    saved = manager.get_workflow("development setup")
    assert [step["intent"] for step in saved["steps"]] == ["OPEN_APPLICATION", "OPEN_URL"]
    assert "Saved workflow" in output[0]
    assert main._expand_workflow_action(
        "opens Chrome, opens GitHub, and opens Notepad"
    ) == "open Chrome, open GitHub, and open Notepad"
    example_plan = planner.create_workflow_plan(main._expand_workflow_action(
        "opens Chrome, opens GitHub, and opens Notepad"
    ))
    assert [step.command["intent"] for step in example_plan.steps] == [
        "OPEN_APPLICATION", "OPEN_URL", "OPEN_APPLICATION"
    ]
    assert main._parse_workflow_request("run my development setup workflow") == (
        "run", "development setup", None
    )


def test_storage_rejects_unknown_fields_and_duplicate_ids(setup_workflows):
    manager, _, _ = setup_workflows
    saved = manager.create_workflow("Dev", [command()])
    payload = json.loads(manager.storage_path.read_text(encoding="utf-8"))
    payload["workflows"][0]["confirmed"] = True
    manager.storage_path.write_text(json.dumps(payload), encoding="utf-8")
    assert manager.validate_workflow({**saved, "confirmed": True})
    with pytest.raises(WorkflowError, match="structure"):
        manager.get_workflow("Dev")


def test_malformed_automation_settings_fall_back_safely():
    from core.config_manager import DEFAULT_CONFIG, _validate_and_fix

    config = copy.deepcopy(DEFAULT_CONFIG)
    config["automation"] = {
        "enabled": "yes", "scheduler_enabled": True,
        "skip_missed_runs": "false", "max_workflows": 100000,
        "max_steps_per_workflow": 99, "max_workflow_name_length": -1,
        "max_description_length": True, "max_runs_per_hour": 0,
    }
    fixed = _validate_and_fix(config)["automation"]
    assert fixed == {
        "enabled": False, "max_workflows": 50,
        "max_steps_per_workflow": 10,
        "max_workflow_name_length": 100,
        "max_description_length": 500,
        "max_runs_per_hour": 10,
        "scheduler_enabled": False, "skip_missed_runs": True,
    }
