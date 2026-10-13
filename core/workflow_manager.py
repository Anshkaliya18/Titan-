"""Bounded persistent storage for validated TITAN command workflows.

This module manages data only. It never invokes the Executor, managers, a
shell, or Python code. Every stored workflow is treated as untrusted and is
revalidated through TaskPlanner before use.
"""

from __future__ import annotations

import copy
import json
import os
import re
import tempfile
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config_manager import get_value
from core.task_planner import TaskPlanner
from utils import paths
from utils.logger import get_logger

logger = get_logger(__name__)

MAX_STORAGE_BYTES = 1_000_000
MAX_WORKFLOWS_HARD = 50
MAX_STEPS_HARD = 10  # The existing TaskPlanner has an independent hard cap of 10.
_WORKFLOW_ID = re.compile(r"^workflow-[0-9a-f]{32}$")
_SECRET_TEXT = re.compile(
    r"(?i)\b(?:password|passwd|api[_ -]?key|token|secret|credential|cookie)\b\s*(?:=|:|is)\s*\S+"
)


class WorkflowError(ValueError):
    """A safe, user-displayable workflow validation or storage error."""


class WorkflowManager:
    """Create and maintain bounded JSON workflows without executing them."""

    def __init__(self, storage_path: str | Path | None = None,
                 task_planner: TaskPlanner | None = None) -> None:
        self._path = Path(storage_path) if storage_path is not None else paths.data_root() / "data" / "workflows.json"
        self._planner = task_planner or TaskPlanner()

    @property
    def storage_path(self) -> Path:
        return self._path

    def create_workflow(self, name: str, steps: list[dict[str, Any]], *,
                        description: str = "") -> dict[str, Any]:
        normalized_name = self._validate_name(name)
        description = self._validate_description(description)
        self._validate_steps(steps)
        workflows = self._load()
        if any(_name_key(item["name"]) == _name_key(normalized_name) for item in workflows):
            raise WorkflowError("A workflow with that name already exists.")
        limits = _automation_limits()
        if len(workflows) >= limits["max_workflows"]:
            raise WorkflowError("The workflow storage limit has been reached.")
        now = _now()
        workflow = {
            "id": f"workflow-{uuid.uuid4().hex}",
            "name": normalized_name,
            "description": description,
            "steps": copy.deepcopy(steps),
            "enabled": True,
            "created_at": now,
            "updated_at": now,
            "schedule": None,
            "metadata": {},
        }
        self._validate_workflow(workflow)
        workflows.append(workflow)
        self._save(workflows)
        logger.info("Workflow created (id=%s, steps=%d)", workflow["id"], len(steps))
        return copy.deepcopy(workflow)

    def get_workflow(self, name_or_id: str) -> dict[str, Any] | None:
        workflows = self._load()
        item = self._find(workflows, name_or_id)
        return copy.deepcopy(item) if item else None

    def list_workflows(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._load())

    def update_workflow(self, name_or_id: str, *, name: str | None = None,
                        description: str | None = None,
                        steps: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        workflows = self._load()
        current = self._find(workflows, name_or_id)
        if current is None:
            raise WorkflowError("Workflow not found.")
        updated = copy.deepcopy(current)
        if name is not None:
            updated["name"] = self._validate_name(name)
        if description is not None:
            updated["description"] = self._validate_description(description)
        if steps is not None:
            self._validate_steps(steps)
            updated["steps"] = copy.deepcopy(steps)
        if any(item["id"] != current["id"] and _name_key(item["name"]) == _name_key(updated["name"]) for item in workflows):
            raise WorkflowError("A workflow with that name already exists.")
        updated["updated_at"] = _now()
        self._validate_workflow(updated)
        workflows[workflows.index(current)] = updated
        self._save(workflows)
        logger.info("Workflow updated (id=%s)", updated["id"])
        return copy.deepcopy(updated)

    def delete_workflow(self, name_or_id: str) -> bool:
        workflows = self._load()
        item = self._find(workflows, name_or_id)
        if item is None:
            return False
        workflows.remove(item)
        self._save(workflows)
        logger.info("Workflow deleted (id=%s)", item["id"])
        return True

    def enable_workflow(self, name_or_id: str) -> dict[str, Any]:
        return self._set_enabled(name_or_id, True)

    def disable_workflow(self, name_or_id: str) -> dict[str, Any]:
        return self._set_enabled(name_or_id, False)

    def run_workflow(self, name_or_id: str) -> dict[str, Any]:
        """Return a fresh plan after reloading/revalidating; never executes it."""
        if get_value("automation", "enabled", False) is not True:
            raise WorkflowError("Automation is disabled in config/settings.json.")
        workflow = self.get_workflow(name_or_id)
        if workflow is None:
            raise WorkflowError("Workflow not found.")
        self._validate_workflow(workflow)
        if workflow["enabled"] is not True:
            raise WorkflowError("That workflow is disabled. Enable it before running.")
        plan = self._planner.create_plan_from_commands(
            f"Run workflow {workflow['name']}", workflow["steps"],
            source_prefix=workflow["name"],
        )
        return {"workflow": workflow, "plan": plan}

    def validate_workflow(self, workflow: dict[str, Any]) -> tuple[str, ...]:
        try:
            self._validate_workflow(workflow)
        except WorkflowError as exc:
            return (str(exc),)
        return ()

    def _set_enabled(self, name_or_id: str, enabled: bool) -> dict[str, Any]:
        workflows = self._load()
        item = self._find(workflows, name_or_id)
        if item is None:
            raise WorkflowError("Workflow not found.")
        # Validate before enabling and also before persisting any state change.
        self._validate_workflow(item)
        item["enabled"] = enabled
        item["updated_at"] = _now()
        self._save(workflows)
        logger.info("Workflow %s (id=%s)", "enabled" if enabled else "disabled", item["id"])
        return copy.deepcopy(item)

    def _validate_name(self, name: str) -> str:
        if not isinstance(name, str):
            raise WorkflowError("Workflow name must be text.")
        if any(unicodedata.category(char).startswith("C") for char in name):
            raise WorkflowError("Workflow name contains unsupported control characters.")
        normalized = " ".join(name.split())
        limit = _automation_limits()["max_workflow_name_length"]
        if not normalized or len(normalized) > limit:
            raise WorkflowError(f"Workflow name must be between 1 and {limit} characters.")
        if any(unicodedata.category(char).startswith("C") for char in normalized):
            raise WorkflowError("Workflow name contains unsupported control characters.")
        if _SECRET_TEXT.search(normalized):
            raise WorkflowError("Workflow names cannot contain secret-like values.")
        return normalized

    def _validate_description(self, description: str) -> str:
        if not isinstance(description, str):
            raise WorkflowError("Workflow description must be text.")
        if any(unicodedata.category(char).startswith("C") for char in description):
            raise WorkflowError("Workflow description contains unsupported control characters.")
        normalized = " ".join(description.split())
        limit = _automation_limits()["max_description_length"]
        if len(normalized) > limit:
            raise WorkflowError(f"Workflow description exceeds {limit} characters.")
        if any(unicodedata.category(char).startswith("C") for char in normalized):
            raise WorkflowError("Workflow description contains unsupported control characters.")
        if _SECRET_TEXT.search(normalized):
            raise WorkflowError("Workflow descriptions cannot contain secret-like values.")
        return normalized

    def _validate_steps(self, steps: list[dict[str, Any]]) -> None:
        if not isinstance(steps, list) or not steps:
            raise WorkflowError("A workflow must contain at least one step.")
        limit = min(_automation_limits()["max_steps_per_workflow"], MAX_STEPS_HARD)
        if len(steps) > limit:
            raise WorkflowError(f"Workflow exceeds the {limit}-step limit.")
        if any(not isinstance(step, dict) or set(step) != {"intent", "target", "arguments"} for step in steps):
            raise WorkflowError("Each step must contain only intent, target, and arguments.")
        for step in steps:
            target = step.get("target")
            if isinstance(target, str) and _SECRET_TEXT.search(target):
                raise WorkflowError("Workflow targets cannot contain secret-like values.")
            arguments = step.get("arguments")
            if isinstance(arguments, dict):
                for key, value in arguments.items():
                    if not isinstance(key, str) or not isinstance(value, str) or len(value) > 512:
                        raise WorkflowError("Workflow arguments must be bounded text values.")
                    if any(unicodedata.category(char).startswith("C") for char in value):
                        raise WorkflowError("Workflow arguments contain unsupported control characters.")
                    if _SECRET_TEXT.search(value):
                        raise WorkflowError("Workflow arguments cannot contain secret-like values.")
        plan_steps = [
            {"step_id": index, "order": index, "command": step,
             "dependencies": [index - 1] if index > 1 else []}
            for index, step in enumerate(steps, start=1)
        ]
        fake = {
            "plan_id": "plan-" + "0" * 32,
            "original_request": "workflow validation",
            "steps": plan_steps,
        }
        result = self._planner.validate_plan(fake)
        if not result.valid:
            raise WorkflowError("Workflow validation failed: " + "; ".join(result.errors))

    def _validate_workflow(self, workflow: dict[str, Any]) -> None:
        required = {"id", "name", "description", "steps", "enabled", "created_at", "updated_at", "schedule", "metadata"}
        if not isinstance(workflow, dict) or set(workflow) != required:
            raise WorkflowError("Workflow data has an invalid structure.")
        if not isinstance(workflow["id"], str) or not _WORKFLOW_ID.fullmatch(workflow["id"]):
            raise WorkflowError("Workflow ID is invalid.")
        if self._validate_name(workflow["name"]) != workflow["name"]:
            raise WorkflowError("Workflow name is not normalized.")
        if self._validate_description(workflow["description"]) != workflow["description"]:
            raise WorkflowError("Workflow description is not normalized.")
        if type(workflow["enabled"]) is not bool or workflow["schedule"] is not None or workflow["metadata"] != {}:
            raise WorkflowError("Workflow contains unsupported state or schedule data.")
        for key in ("created_at", "updated_at"):
            if not isinstance(workflow[key], str) or len(workflow[key]) > 40:
                raise WorkflowError("Workflow timestamps are invalid.")
            try:
                datetime.fromisoformat(workflow[key])
            except ValueError:
                raise WorkflowError("Workflow timestamps are invalid.") from None
        self._validate_steps(workflow["steps"])

    def _load(self) -> list[dict[str, Any]]:
        try:
            if not self._path.exists():
                return []
            if self._path.stat().st_size > MAX_STORAGE_BYTES:
                raise WorkflowError("Workflow storage exceeds its size limit.")
            with self._path.open("r", encoding="utf-8") as handle:
                root = json.load(handle)
            if not isinstance(root, dict) or set(root) != {"version", "workflows"} or root["version"] != 1 or not isinstance(root["workflows"], list):
                raise WorkflowError("Workflow storage has an invalid schema.")
            workflows = root["workflows"]
            limits = _automation_limits()
            if len(workflows) > min(limits["max_workflows"], MAX_WORKFLOWS_HARD):
                raise WorkflowError("Workflow storage exceeds its workflow count limit.")
            ids: set[str] = set()
            names: set[str] = set()
            for item in workflows:
                self._validate_workflow(item)
                if item["id"] in ids or _name_key(item["name"]) in names:
                    raise WorkflowError("Workflow storage contains duplicate identifiers or names.")
                ids.add(item["id"])
                names.add(_name_key(item["name"]))
            return workflows
        except WorkflowError:
            logger.warning("Workflow storage rejected by validation")
            raise
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, KeyError, ValueError, RecursionError):
            logger.warning("Workflow storage could not be read or parsed")
            raise WorkflowError("Workflow storage is unavailable or corrupted.") from None

    def _save(self, workflows: list[dict[str, Any]]) -> None:
        payload = json.dumps({"version": 1, "workflows": workflows}, ensure_ascii=False, indent=2)
        encoded = payload.encode("utf-8")
        if len(encoded) > MAX_STORAGE_BYTES:
            raise WorkflowError("Workflow storage exceeds its size limit.")
        temporary: str | None = None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile("wb", dir=self._path.parent, prefix=".workflows-", suffix=".tmp", delete=False) as handle:
                temporary = handle.name
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self._path)
        except OSError:
            logger.warning("Workflow storage could not be written")
            raise WorkflowError("Workflow storage could not be saved.") from None
        finally:
            if temporary:
                try:
                    Path(temporary).unlink(missing_ok=True)
                except OSError:
                    pass

    @staticmethod
    def _find(workflows: list[dict[str, Any]], name_or_id: str) -> dict[str, Any] | None:
        if not isinstance(name_or_id, str):
            return None
        query = " ".join(name_or_id.split())
        for item in workflows:
            if item["id"] == query or _name_key(item["name"]) == _name_key(query):
                return item
        return None


def _automation_limits() -> dict[str, int]:
    caps = {"max_workflows": MAX_WORKFLOWS_HARD,
            "max_steps_per_workflow": MAX_STEPS_HARD,
            "max_workflow_name_length": 100,
            "max_description_length": 500}
    defaults = {"max_workflows": 50, "max_steps_per_workflow": 10,
                "max_workflow_name_length": 100, "max_description_length": 500}
    values: dict[str, int] = {}
    for key, cap in caps.items():
        value = get_value("automation", key, defaults[key])
        values[key] = min(value, cap) if type(value) is int and 1 <= value <= cap else defaults[key]
    return values


def _name_key(name: str) -> str:
    return " ".join(name.casefold().split())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
