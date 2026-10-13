"""Bounded structured task plans; this module never executes commands.

Plans contain only validated TITAN command dictionaries. Execution remains
with ``core.multi_command_executor`` and the ordinary Executor boundary.
"""

from __future__ import annotations

import copy
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from core.command_registry import get_command, is_valid_intent
from core.command_validator import validate_command
from core.config_manager import get_value
from core.exceptions import MultiCommandError
from core.multi_command_executor import plan_command
from core.multi_command_parser import split_commands
from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_MAX_STEPS = 10
HARD_MAX_STEPS = 10
DEFAULT_MAX_INPUT_LENGTH = 2000
HARD_MAX_INPUT_LENGTH = 2000
MAX_RETAINED_PLANS = 20
_PLAN_ID_RE = re.compile(r"^plan-[0-9a-f]{32}$")


class PlanStatus(str, Enum):
    PENDING = "PENDING"
    VALIDATED = "VALIDATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    BLOCKED = "BLOCKED"
    REQUIRES_CONFIRMATION = "REQUIRES_CONFIRMATION"


class StepStatus(str, Enum):
    PENDING = "PENDING"
    VALIDATED = "VALIDATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    REQUIRES_CONFIRMATION = "REQUIRES_CONFIRMATION"
    NOT_EXECUTED = "NOT_EXECUTED"


@dataclass
class PlanStep:
    step_id: int
    order: int
    command: dict[str, Any]
    description: str
    source_text: str = ""
    dependencies: list[int] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class TaskPlan:
    plan_id: str
    original_request: str
    steps: list[PlanStep]
    current_step: int | None = None
    status: PlanStatus = PlanStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-friendly plan snapshot."""
        return {
            "plan_id": self.plan_id,
            "original_request": self.original_request,
            "steps": [
                {
                    "step_id": step.step_id,
                    "order": step.order,
                    "command": copy.deepcopy(step.command),
                    "description": step.description,
                    "source_text": step.source_text,
                    "status": step.status.value,
                    "dependencies": list(step.dependencies),
                    "result": copy.deepcopy(step.result),
                    "error": step.error,
                }
                for step in self.steps
            ],
            "current_step": self.current_step,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


@dataclass(frozen=True)
class PlanValidation:
    valid: bool
    errors: tuple[str, ...] = ()


def _planning_config() -> dict[str, Any]:
    """Read bounded settings; malformed values always resolve safely."""
    enabled = get_value("task_planning", "enabled", False)
    require_confirmation = get_value(
        "task_planning", "require_confirmation_for_plan", False
    )
    max_steps = get_value("task_planning", "max_steps", DEFAULT_MAX_STEPS)
    max_input = get_value(
        "task_planning", "max_plan_input_length", DEFAULT_MAX_INPUT_LENGTH
    )
    return {
        "enabled": enabled if type(enabled) is bool else False,
        "require_confirmation_for_plan": (
            require_confirmation if type(require_confirmation) is bool else True
        ),
        "allow_llm_planning": get_value("task_planning", "allow_llm_planning", False)
        is True,
        "max_steps": min(max_steps, HARD_MAX_STEPS)
        if type(max_steps) is int and 1 <= max_steps <= HARD_MAX_STEPS
        else DEFAULT_MAX_STEPS,
        "max_plan_input_length": min(max_input, HARD_MAX_INPUT_LENGTH)
        if type(max_input) is int and 1 <= max_input <= HARD_MAX_INPUT_LENGTH
        else DEFAULT_MAX_INPUT_LENGTH,
    }


class TaskPlanner:
    """Create, validate, inspect, and cancel bounded plans in memory."""

    def __init__(self, *, context_manager=None) -> None:
        self._context_manager = context_manager
        self._plans: dict[str, TaskPlan] = {}

    def create_plan(self, request: str, *, preview: bool = False) -> TaskPlan:
        """Create an ordinary plan subject to task_planning opt-in."""
        return self._create_plan(request, preview=preview, workflow_request=False)

    def create_workflow_plan(self, request: str) -> TaskPlan:
        """Plan an explicit workflow definition under automation opt-in."""
        if get_value("automation", "enabled", False) is not True:
            raise ValueError("Automation is disabled.")
        return self._create_plan(request, preview=True, workflow_request=True)

    def _create_plan(self, request: str, *, preview: bool,
                     workflow_request: bool) -> TaskPlan:
        """Deterministically parse a request into existing TITAN commands."""
        config = _planning_config()
        if not config["enabled"] and not workflow_request:
            raise ValueError("Task planning is disabled.")
        if not isinstance(request, str) or not request.strip():
            raise ValueError("Task request must be a non-empty string.")
        request = request.strip()
        if len(request) > config["max_plan_input_length"]:
            raise ValueError("Task request exceeds the configured input limit.")
        if config["require_confirmation_for_plan"] and not preview and not workflow_request:
            # A plan-wide `yes` would authorize more than one command and
            # would violate the session's exact-command confirmation model.
            raise ValueError(
                "Task-plan execution is disabled by require_confirmation_for_plan; "
                "turn that setting off to run plans. Dangerous steps still require "
                "their own confirmation."
            )

        try:
            segments = split_commands(request)
        except MultiCommandError as exc:
            raise ValueError(str(exc)) from None
        if not segments:
            segments = [request]
        if len(segments) > config["max_steps"]:
            raise ValueError(
                f"Task plan has {len(segments)} steps; the configured maximum is "
                f"{config['max_steps']}."
            )

        preview_context = (
            self._context_manager.fork() if self._context_manager is not None else None
        )
        steps: list[PlanStep] = []
        for index, segment in enumerate(segments, start=1):
            command_text = _expand_known_site(segment)
            try:
                if preview_context is not None:
                    preview_context.advance()
                parsed = plan_command(
                    command_text, preview_context,
                    use_llm=config["allow_llm_planning"] and not workflow_request,
                    use_natural_language=True,
                )
            except Exception as exc:
                logger.info("Task planning failed (%s)", type(exc).__name__)
                raise ValueError("I couldn't interpret one of the requested steps.") from None
            if not parsed.valid:
                raise ValueError(
                    f"Step {index} is not a valid TITAN command: "
                    f"{parsed.validation.get('message', 'invalid command')}"
                )
            command = _canonical_command(parsed.command)
            steps.append(PlanStep(
                step_id=index,
                order=index,
                command=command,
                description=(get_command(command["intent"]).description),
                source_text=command_text,
                dependencies=[index - 1] if index > 1 else [],
                status=StepStatus.VALIDATED,
            ))
            if preview_context is not None and command["intent"] not in _DANGEROUS_INTENTS:
                preview_context.record_success(command, {"success": True})

        plan = TaskPlan(
            plan_id=f"plan-{uuid.uuid4().hex}",
            original_request=request,
            steps=steps,
            status=PlanStatus.VALIDATED,
        )
        validation = self.validate_plan(plan)
        if not validation.valid:
            raise ValueError("Task plan rejected: " + "; ".join(validation.errors))
        self._plans[plan.plan_id] = plan
        self._trim_plans()
        return copy.deepcopy(plan)

    def create_plan_from_commands(
        self, request: str, commands: list[dict[str, Any]], *, source_prefix: str = "workflow"
    ) -> TaskPlan:
        """Build a fresh validated TaskPlan from already stored command data.

        This is the workflow bridge: stored data is revalidated through this
        planner and the existing command validator each time it is used. It
        deliberately does not execute anything.
        """
        config = _planning_config()
        if not isinstance(request, str) or not request.strip() or len(request) > config["max_plan_input_length"]:
            raise ValueError("Task request is missing or exceeds the input limit.")
        if not isinstance(commands, list) or not commands:
            raise ValueError("Workflow must contain at least one command.")
        if len(commands) > config["max_steps"]:
            raise ValueError("Workflow exceeds the task planner step limit.")
        steps: list[PlanStep] = []
        for index, raw_command in enumerate(commands, start=1):
            if not isinstance(raw_command, dict) or set(raw_command) != {"intent", "target", "arguments"}:
                raise ValueError(f"Workflow step {index} is malformed.")
            command = _canonical_command(raw_command)
            definition = get_command(command["intent"])
            if definition is None or not validate_command(command)["valid"] or not _safe_target(command):
                raise ValueError(f"Workflow step {index} failed command validation.")
            target = command.get("target")
            description = definition.description + (f" ({target})" if target else "")
            steps.append(PlanStep(
                step_id=index, order=index, command=command,
                description=description,
                source_text=_command_history_text(command),
                dependencies=[index - 1] if index > 1 else [],
                status=StepStatus.VALIDATED,
            ))
        plan = TaskPlan(
            plan_id=f"plan-{uuid.uuid4().hex}",
            original_request=request.strip(), steps=steps,
            status=PlanStatus.VALIDATED,
        )
        validation = self.validate_plan(plan)
        if not validation.valid:
            raise ValueError("Workflow plan rejected: " + "; ".join(validation.errors))
        self._plans[plan.plan_id] = plan
        self._trim_plans()
        return copy.deepcopy(plan)

    def validate_plan(self, plan: TaskPlan | dict[str, Any]) -> PlanValidation:
        """Validate complete plan structure, registry intents, and DAG."""
        errors: list[str] = []
        if isinstance(plan, TaskPlan):
            plan_id = plan.plan_id
            request = plan.original_request
            steps = plan.steps
            step_values = [
                {
                    "step_id": step.step_id, "order": step.order,
                    "command": step.command, "dependencies": step.dependencies,
                    "source_text": step.source_text,
                }
                for step in steps
            ]
        elif isinstance(plan, dict):
            if set(plan) - {"plan_id", "original_request", "steps", "current_step", "status", "created_at", "completed_at"}:
                errors.append("Plan contains unknown fields.")
            plan_id = plan.get("plan_id")
            request = plan.get("original_request")
            steps = plan.get("steps")
            step_values = steps if isinstance(steps, list) else []
        else:
            return PlanValidation(False, ("Plan must be a TaskPlan or object.",))

        if not isinstance(plan_id, str) or not _PLAN_ID_RE.fullmatch(plan_id):
            errors.append("Plan ID is invalid.")
        config = _planning_config()
        if not isinstance(request, str) or not request.strip() or len(request) > config["max_plan_input_length"]:
            errors.append("Original request is missing or exceeds the input limit.")
        if not isinstance(step_values, list) or not step_values:
            errors.append("Plan must contain at least one step.")
            return PlanValidation(False, tuple(errors))
        if len(step_values) > config["max_steps"]:
            errors.append("Plan exceeds the configured maximum step count.")

        ids: list[int] = []
        graph: dict[int, list[int]] = {}
        for index, step in enumerate(step_values, start=1):
            if isinstance(step, PlanStep):
                step = {"step_id": step.step_id, "order": step.order,
                        "command": step.command, "dependencies": step.dependencies}
            if not isinstance(step, dict):
                errors.append(f"Step {index} must be an object.")
                continue
            if set(step) - {"step_id", "order", "command", "description", "source_text", "dependencies", "status", "result", "error"}:
                errors.append(f"Step {index} contains unknown fields.")
            step_id = step.get("step_id")
            order = step.get("order")
            if type(step_id) is not int or step_id < 1:
                errors.append(f"Step {index} has an invalid ID.")
                continue
            ids.append(step_id)
            if type(order) is not int or order != index:
                errors.append(f"Step {index} order is invalid.")
            command = step.get("command")
            if not isinstance(command, dict) or set(command) != {"intent", "target", "arguments"}:
                errors.append(f"Step {index} command must contain only intent, target, and arguments.")
            else:
                args = command.get("arguments")
                if not isinstance(args, dict):
                    errors.append(f"Step {index} arguments must be an object.")
                elif (command.get("intent") == "SEARCH_WEB"):
                    if set(args) - {"query"} or any(
                        value is not None and not isinstance(value, str)
                        for value in args.values()
                    ):
                        errors.append(f"Step {index} has unsupported arguments.")
                elif args:
                    errors.append(f"Step {index} has unsupported arguments.")
                intent = command.get("intent")
                if not isinstance(intent, str) or not is_valid_intent(intent) or intent.upper() == "UNKNOWN":
                    errors.append(f"Step {index} uses an unregistered intent.")
                elif not validate_command(command)["valid"]:
                    errors.append(f"Step {index} failed command validation.")
                elif not _safe_target(command):
                    errors.append(f"Step {index} has an unsafe target.")
            dependencies = step.get("dependencies", [])
            if not isinstance(dependencies, list) or any(type(dep) is not int for dep in dependencies):
                errors.append(f"Step {index} dependencies must be integer IDs.")
                continue
            if len(set(dependencies)) != len(dependencies) or step_id in dependencies:
                errors.append(f"Step {index} has duplicate or self-referential dependencies.")
            graph[step_id] = dependencies
        if len(set(ids)) != len(ids):
            errors.append("Plan contains duplicate step IDs.")
        known = set(ids)
        order_by_id = {
            step.get("step_id"): step.get("order")
            for step in step_values if isinstance(step, dict)
        }
        for step_id, dependencies in graph.items():
            if any(dep not in known for dep in dependencies):
                errors.append(f"Step {step_id} refers to an unknown dependency.")
            if any(
                dep in order_by_id
                and type(order_by_id[dep]) is int
                and type(order_by_id.get(step_id)) is int
                and order_by_id[dep] >= order_by_id[step_id]
                for dep in dependencies
            ):
                errors.append(f"Step {step_id} depends on a later step.")
        if len(known) == len(step_values) and _has_cycle(graph):
            errors.append("Plan dependencies contain a cycle.")
        return PlanValidation(not errors, tuple(errors))

    def get_plan(self, plan_id: str) -> TaskPlan | None:
        plan = self._plans.get(plan_id)
        return copy.deepcopy(plan) if plan else None

    def start_plan(self, plan_id: str) -> bool:
        """Mark a locally validated plan as running before delegation."""
        plan = self._plans.get(plan_id)
        if plan is None or plan.status != PlanStatus.VALIDATED:
            return False
        plan.status = PlanStatus.RUNNING
        plan.current_step = 1
        return True

    def stop_after_confirmation(self, plan_id: str) -> bool:
        """Mark unconfirmed later steps as not run and stop the plan."""
        plan = self._plans.get(plan_id)
        if plan is None:
            return False
        for step in plan.steps:
            if step.status in {StepStatus.PENDING, StepStatus.VALIDATED}:
                step.status = StepStatus.NOT_EXECUTED
        plan.status = PlanStatus.BLOCKED
        plan.completed_at = datetime.now(timezone.utc)
        return True

    def mark_unexecuted(self, plan_id: str, step_ids: set[int]) -> None:
        plan = self._plans.get(plan_id)
        if plan is None:
            return
        for step in plan.steps:
            if step.step_id in step_ids and step.status in {
                StepStatus.PENDING, StepStatus.VALIDATED,
            }:
                step.status = StepStatus.NOT_EXECUTED

    def get_plan_status(self, plan_id: str) -> dict[str, Any] | None:
        plan = self._plans.get(plan_id)
        if plan is None:
            return None
        completed = sum(step.status == StepStatus.COMPLETED for step in plan.steps)
        return {
            "plan_id": plan.plan_id,
            "status": plan.status.value,
            "current_step": plan.current_step,
            "total_steps": len(plan.steps),
            "completed_steps": completed,
        }

    def cancel_plan(self, plan_id: str) -> bool:
        plan = self._plans.get(plan_id)
        if plan is None or plan.status in {
            PlanStatus.COMPLETED, PlanStatus.FAILED, PlanStatus.CANCELLED,
            PlanStatus.REQUIRES_CONFIRMATION,
        }:
            return False
        plan.status = PlanStatus.CANCELLED
        plan.completed_at = datetime.now(timezone.utc)
        for step in plan.steps:
            if step.status in {StepStatus.PENDING, StepStatus.VALIDATED}:
                step.status = StepStatus.CANCELLED
        return True

    def update_step(self, plan_id: str, step_id: int, status: StepStatus,
                    result: dict[str, Any] | None = None, error: str | None = None) -> bool:
        """Apply execution outcome reported by the existing pipeline."""
        plan = self._plans.get(plan_id)
        if plan is None:
            return False
        step = next((item for item in plan.steps if item.step_id == step_id), None)
        if step is None:
            return False
        step.status = status
        step.result = copy.deepcopy(result)
        step.error = error
        plan.current_step = step_id
        if status == StepStatus.REQUIRES_CONFIRMATION:
            plan.status = PlanStatus.REQUIRES_CONFIRMATION
        elif status == StepStatus.FAILED:
            plan.status = PlanStatus.FAILED
            plan.completed_at = datetime.now(timezone.utc)
            for remaining in plan.steps:
                if remaining.step_id > step_id and remaining.status in {StepStatus.PENDING, StepStatus.VALIDATED}:
                    remaining.status = StepStatus.NOT_EXECUTED
        elif status in {StepStatus.COMPLETED, StepStatus.CANCELLED}:
            if status == StepStatus.CANCELLED:
                plan.status = PlanStatus.CANCELLED
            elif all(item.status == StepStatus.COMPLETED for item in plan.steps):
                plan.status = PlanStatus.COMPLETED
            elif plan.status != PlanStatus.REQUIRES_CONFIRMATION:
                plan.status = PlanStatus.RUNNING
            if plan.status in {PlanStatus.COMPLETED, PlanStatus.CANCELLED}:
                plan.completed_at = datetime.now(timezone.utc)
        return True

    def resolve_confirmation(self, plan_id: str, step_id: int, *,
                             accepted: bool, result: dict[str, Any] | None = None) -> bool:
        """Resolve one parked command without resuming later plan steps."""
        plan = self._plans.get(plan_id)
        if plan is None:
            return False
        step = next((item for item in plan.steps if item.step_id == step_id), None)
        if step is None or step.status != StepStatus.REQUIRES_CONFIRMATION:
            return False
        success = bool(result and result.get("success")) if accepted else False
        step.status = StepStatus.COMPLETED if success else (
            StepStatus.FAILED if accepted else StepStatus.CANCELLED
        )
        step.result = copy.deepcopy(result)
        if not success:
            step.error = "Confirmation was cancelled." if not accepted else "Command failed."
        for remaining in plan.steps:
            if remaining.step_id > step_id and remaining.status in {
                StepStatus.PENDING, StepStatus.VALIDATED,
            }:
                remaining.status = StepStatus.NOT_EXECUTED
        plan.current_step = step_id
        plan.status = PlanStatus.BLOCKED if success else (
            PlanStatus.CANCELLED if not accepted else PlanStatus.FAILED
        )
        plan.completed_at = datetime.now(timezone.utc)
        return True

    def _trim_plans(self) -> None:
        while len(self._plans) > MAX_RETAINED_PLANS:
            self._plans.pop(next(iter(self._plans)))


_DANGEROUS_INTENTS = {"SHUTDOWN_COMPUTER", "RESTART_COMPUTER", "DELETE_FILE",
                      "ENABLE_STARTUP", "DISABLE_STARTUP"}


def _canonical_command(command: dict[str, Any]) -> dict[str, Any]:
    return {
        "intent": str(command.get("intent", "")).upper(),
        "target": command.get("target"),
        "arguments": copy.deepcopy(command.get("arguments", {})),
    }


def _command_history_text(command: dict[str, Any]) -> str:
    """Render the validated operation in the same compact form as CLI history."""
    intent, target = command["intent"], command.get("target")
    phrases = {
        "OPEN_APPLICATION": "open", "CLOSE_APPLICATION": "close",
        "CREATE_FOLDER": "create folder", "DELETE_FILE": "delete file",
        "OPEN_FILE": "open file", "SEARCH_FILE": "search file",
        "OPEN_URL": "open", "SEARCH_WEB": "search web",
    }
    if intent in phrases:
        return phrases[intent] + (f" {target}" if target else "")
    fixed = {
        "OPEN_BROWSER": "open browser", "LOCK_COMPUTER": "lock computer",
        "SHUTDOWN_COMPUTER": "shutdown", "RESTART_COMPUTER": "restart",
        "ENABLE_STARTUP": "enable startup", "DISABLE_STARTUP": "disable startup",
        "STARTUP_STATUS": "startup status", "HELP": "help",
        "LIST_COMMANDS": "list commands", "HISTORY": "history",
        "CLEAR_HISTORY": "clear history",
    }
    return fixed.get(intent, intent.casefold().replace("_", " "))


def _expand_known_site(text: str) -> str:
    """Resolve the one unambiguous bare-site phrase from the Step 34 examples."""
    if re.fullmatch(r"(?:open|launch|start)\s+github", text.strip(), re.IGNORECASE):
        return "open https://github.com"
    return text


def _safe_target(command: dict[str, Any]) -> bool:
    intent, target = command.get("intent"), command.get("target")
    if target is not None and (not isinstance(target, str) or len(target) > 512
                               or any(ord(char) < 32 for char in target)):
        return False
    if intent in {"OPEN_APPLICATION", "CLOSE_APPLICATION"}:
        from core.application_manager import get_application
        return isinstance(target, str) and get_application(target) is not None
    if intent == "OPEN_URL":
        return isinstance(target, str) and target.casefold().startswith(("http://", "https://"))
    if intent not in {
        "CLOSE_APPLICATION", "CREATE_FOLDER", "DELETE_FILE", "OPEN_FILE",
        "SEARCH_FILE", "SEARCH_WEB",
    }:
        return target is None
    if intent != "SEARCH_WEB" and isinstance(target, str) and re.search(
        r"[\r\n\x00`|<>]|\$\(|&&|;\s*(?:powershell|cmd|python|del|rm)\b",
        target,
        flags=re.IGNORECASE,
    ):
        return False
    return True


def _has_cycle(graph: dict[int, list[int]]) -> bool:
    visiting: set[int] = set()
    visited: set[int] = set()

    def visit(node: int) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        if any(dep in graph and visit(dep) for dep in graph[node]):
            return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node) for node in graph)
