"""
Multi-Command Execution Layer
-----------------------------
Plans and executes a list of command segments sequentially.

    list of command strings
            ↓
    PLAN each: normalize → tokenize → spell check → intent →
               parse → validate            (existing pipeline)
            ↓
    EXECUTE sequentially through the ordinary Executor
            ↓
    combined structured result

Responsibilities:
- Reuse the existing pipeline for EVERY command — no duplicated
  intent detection, parsing, validation or execution logic.
- Pre-validate the whole chain before executing anything, so a
  malformed later command can never cause surprising partial runs.
- Stop on failure: a failed command ends the chain.
- Stop on confirmation: a dangerous command is NEVER executed here;
  it is returned as `confirmation` for the caller to park in
  SessionState, and the chain ends.

This layer does NOT split natural-language input (that is the
Multi-Command Parser's job) and NEVER touches a shell — every
segment flows through Command Registry → Parser → Validator →
Executor → managers like any single command.
"""

from dataclasses import dataclass, field

from core.command_history import (
    HISTORY_EXEMPT_INTENTS,
    CommandHistory,
    record_entry,
)
from core.command_parser import parse_command
from core.command_validator import validate_command
from core.exceptions import MultiCommandError
from core.executor import (
    CONFIRMATION_INTENTS,
    confirmation_prompt,
    execute_command,
)
from core.input import normalize_input
from core.intent import detect_intent as detect_intent_rules
from core.hybrid_intent import detect_hybrid_intent
from core.intent import Intent
from core.context_manager import ContextManager
from core.natural_language import NaturalLanguageInterpreter
from core.llm_interpreter import LLMInterpreter
from core.multi_command_parser import MAX_COMMANDS, TOO_MANY_COMMANDS_MESSAGE
from core.spell_checker import correct_tokens
from core.tokenizer import tokenize
from utils.logger import get_logger

logger = get_logger(__name__)
_llm_interpreter = LLMInterpreter()
_natural_language = NaturalLanguageInterpreter()

# Safe message when planning a command fails unexpectedly.
PLAN_FAILURE_MESSAGE = "I couldn't process one of the commands."


@dataclass
class PlannedCommand:
    """
    One command segment after planning, before execution.

    Attributes:
        text: The original (trimmed) segment as the user typed it.
        tokens: Tokenizer output for this segment.
        corrected_tokens: Spell-corrected tokens for this segment.
        intent: The detected intent value (lowercase).
        command: The structured command produced by the parser.
        validation: The validation result for the command.
    """

    text: str
    tokens: list[str]
    corrected_tokens: list[str]
    intent: str
    command: dict
    validation: dict

    @property
    def valid(self) -> bool:
        """Whether this command passed validation."""
        return bool(self.validation.get("valid"))


@dataclass
class MultiCommandOutcome:
    """
    The result of planning and executing a command chain.

    Attributes:
        plans: Every planned command (all were pre-validated unless
            planning itself failed part-way).
        results: Per-command result entries in execution order:
            {"command": text, "success": bool, "message": str,
             "data": dict}
        confirmation: The structured dangerous command awaiting
            confirmation, or None. The caller parks it in
            SessionState — this layer never executes it.
        final_plan: The plan the turn ended on (executed, parked, or
            failed) — used by the caller to update session state.
        stop_reason: None when the chain ran to completion, else
            "validation", "failure", or "confirmation".
        executed_count: How many commands actually reached the
            Executor.
    """

    plans: list[PlannedCommand] = field(default_factory=list)
    results: list[dict] = field(default_factory=list)
    confirmation: dict | None = None
    final_plan: PlannedCommand | None = None
    stop_reason: str | None = None
    executed_count: int = 0

    @property
    def success(self) -> bool:
        """True only when every command ran and every one succeeded."""
        return (
            self.stop_reason is None
            and bool(self.results)
            and all(entry.get("success") for entry in self.results)
        )


def plan_command(
    text: str, context_manager: ContextManager | None = None, *,
    use_llm: bool = True, use_natural_language: bool = False,
) -> PlannedCommand:
    """
    Run one command segment through the ordinary planning pipeline.

    Reuses the existing normalize → tokenize → spell check → intent →
    parse → validate components unchanged.

    Args:
        text: One command segment.

    Returns:
        PlannedCommand: The fully planned (not executed) command.
    """
    normalized = normalize_input(text)
    tokens = tokenize(normalized)
    corrected = correct_tokens(tokens)
    context_resolution = (
        context_manager.resolve_reference(text)
        if context_manager is not None else None
    )
    natural_result = (
        _natural_language.interpret(text, corrected)
        if use_natural_language else None
    )
    if natural_result is not None and natural_result.success:
        corrected = tokenize(natural_result.normalized_text)
    llm_result = None
    if natural_result is not None and natural_result.success:
        intent = Intent(natural_result.intent.lower())
    elif context_resolution is not None:
        corrected = tokenize(context_resolution.canonical_text)
        intent = Intent(context_resolution.intent.lower())
    else:
        hybrid_result = detect_hybrid_intent(
            tokens=corrected,
            text=" ".join(corrected),
        )
        intent = hybrid_result.intent
        llm_result = None
        if intent == Intent.UNKNOWN and use_llm:
            context = None
            if context_manager is not None and any(marker in text.casefold() for marker in (
                "just opened", "application i opened", "app i opened",
            )):
                context = context_manager.llm_context()
            llm_result = _llm_interpreter.interpret(text, context=context)
            if llm_result.success:
                intent = Intent(llm_result.intent.lower())
                if llm_result.target is None and context_manager is not None:
                    context_resolution = context_manager.resolve_for_intent(
                        llm_result.intent
                    )
    command = parse_command(corrected, intent)
    if natural_result is not None and natural_result.success:
        if natural_result.target is not None:
            command["target"] = natural_result.target
        command["arguments"].update(natural_result.arguments)
    elif context_resolution is not None:
        command["target"] = context_resolution.target
        command["arguments"].update(context_resolution.arguments)
    elif llm_result is not None and llm_result.success:
        command["target"] = llm_result.target
        command["arguments"].update(llm_result.arguments)
    validation = validate_command(command)
    return PlannedCommand(
        text=text,
        tokens=tokens,
        corrected_tokens=corrected,
        intent=intent.value,
        command=command,
        validation=validation,
    )


def run_multi_commands(
    texts: list[str],
    history: CommandHistory | None = None,
    input_text: str | None = None,
    context_manager: ContextManager | None = None,
) -> MultiCommandOutcome:
    """
    Plan every command, then execute them sequentially in order.

    Every processed command is recorded in `history` as it is
    processed (never after the whole chain): successes, failures,
    planning failures, and a dangerous command parked for
    confirmation (recorded with success=None — its outcome is not
    known yet). Exempt meta-commands (HISTORY / CLEAR_HISTORY) are
    executed but never recorded themselves.

    Args:
        texts: Command segments from the Multi-Command Parser.
        history: The session's CommandHistory to record into, or
            None to record nothing (observation is optional and can
            never change execution).
        input_text: The full original user input — stored on every
            entry so each command keeps its turn context. Defaults
            to the segments joined with " and ".

    Returns:
        MultiCommandOutcome: Plans, ordered result entries, the
        parked-confirmation command (if the chain stopped at one),
        and the stop reason.

    Raises:
        MultiCommandError: If more than MAX_COMMANDS texts are
            passed (defense in depth — the parser already checks).
    """
    if not isinstance(texts, list) or not texts:
        return MultiCommandOutcome(stop_reason="validation")

    # The original turn text recorded on every history entry.
    full_input = (
        input_text
        if isinstance(input_text, str) and input_text.strip()
        else " and ".join(str(text) for text in texts)
    )

    if len(texts) > MAX_COMMANDS:
        logger.warning(
            "Multi-command input rejected: %d commands (max %d)",
            len(texts),
            MAX_COMMANDS,
        )
        raise MultiCommandError(TOO_MANY_COMMANDS_MESSAGE)

    # -- Phase 1: plan and validate EVERY command first --------------
    plans: list[PlannedCommand] = []
    preview_context = context_manager.fork() if context_manager is not None else None
    for index, text in enumerate(texts, start=1):
        logger.debug("Processing command %d of %d", index, len(texts))
        try:
            if preview_context is not None:
                preview_context.advance()
            plan = plan_command(text, preview_context)
        except Exception:
            # Unexpected planning error: log it, stop the chain, and
            # return a safe message — never expose the traceback.
            logger.exception("Unexpected error while planning command %d", index)
            outcome = MultiCommandOutcome(
                plans=plans, stop_reason="validation"
            )
            if plans:
                outcome.final_plan = plans[-1]
            outcome.results.append(
                _entry(text, False, PLAN_FAILURE_MESSAGE, {})
            )
            # A failed attempt is part of what the user attempted
            # (intent is None — planning never produced one).
            record_entry(
                history,
                input_text=full_input,
                command_text=text,
                intent=None,
                success=False,
                message=PLAN_FAILURE_MESSAGE,
            )
            return outcome

        plans.append(plan)
        if not plan.valid:
            # A malformed later command means NOTHING executes.
            logger.warning("Command chain stopped due to failure")
            validation_message = plan.validation.get(
                "message", "Invalid command."
            )
            if plan.intent not in HISTORY_EXEMPT_INTENTS:
                record_entry(
                    history,
                    input_text=full_input,
                    command_text=plan.text,
                    intent=plan.intent,
                    success=False,
                    message=validation_message,
                )
            return MultiCommandOutcome(
                plans=plans,
                results=[
                    _entry(
                        plan.text,
                        False,
                        validation_message,
                        {"validation_errors": plan.validation.get(
                            "errors", []
                        )},
                    )
                ],
                final_plan=plan,
                stop_reason="validation",
            )

        if (
            preview_context is not None
            and plan.intent not in CONFIRMATION_INTENTS
        ):
            # Preview only: a later segment can reference this entity
            # while the real session context changes only after execution.
            preview_context.record_success(
                plan.command, {"success": True}
            )

    return _execute_planned_commands(plans, history, full_input, context_manager)


def run_structured_commands(
    steps: list[dict],
    history: CommandHistory | None = None,
    input_text: str | None = None,
    context_manager: ContextManager | None = None,
) -> MultiCommandOutcome:
    """Validate structured plan commands, then use the same executor path.

    ``steps`` contains only ``{"text": ..., "command": ...}`` objects. The
    planner never invokes this function directly; main delegates here after
    the complete plan has passed local schema validation.
    """
    if not isinstance(steps, list) or not steps or len(steps) > MAX_COMMANDS:
        raise MultiCommandError(TOO_MANY_COMMANDS_MESSAGE)
    plans: list[PlannedCommand] = []
    for index, item in enumerate(steps, start=1):
        if not isinstance(item, dict) or set(item) != {"text", "command"}:
            return MultiCommandOutcome(stop_reason="validation")
        text, command = item["text"], item["command"]
        if not isinstance(text, str) or not isinstance(command, dict):
            return MultiCommandOutcome(stop_reason="validation")
        validation = validate_command(command)
        plan = PlannedCommand(
            text=text, tokens=[], corrected_tokens=[],
            intent=str(command.get("intent", "UNKNOWN")).lower(),
            command=dict(command), validation=validation,
        )
        plans.append(plan)
        if not validation.get("valid"):
            return MultiCommandOutcome(
                plans=plans,
                results=[_entry(text, False, validation.get("message", "Invalid command."),
                                {"validation_errors": validation.get("errors", [])})],
                final_plan=plan,
                stop_reason="validation",
            )
    full_input = input_text if isinstance(input_text, str) and input_text.strip() else " and ".join(
        step["text"] for step in steps
    )
    return _execute_planned_commands(plans, history, full_input, context_manager)


def _execute_planned_commands(
    plans: list[PlannedCommand], history: CommandHistory | None,
    full_input: str, context_manager: ContextManager | None,
) -> MultiCommandOutcome:
    """Shared sequential Executor boundary for parsed and structured plans."""
    outcome = MultiCommandOutcome(plans=plans)
    for plan in plans:
        if context_manager is not None:
            context_manager.advance()
        if plan.intent in CONFIRMATION_INTENTS:
            prompt = confirmation_prompt(plan.command)
            outcome.results.append(_entry(
                plan.text, False, prompt, {"pending_confirmation": True}
            ))
            record_entry(history, input_text=full_input, command_text=plan.text,
                         intent=plan.intent, success=None, message=prompt)
            outcome.confirmation = plan.command
            outcome.final_plan = plan
            outcome.stop_reason = "confirmation"
            return outcome
        result = execute_command(plan.command)
        if context_manager is not None:
            context_manager.record_success(plan.command, result)
        outcome.executed_count += 1
        message = result.get("message")
        data = result.get("data")
        outcome.results.append(_entry(
            plan.text, bool(result.get("success")),
            message if isinstance(message, str) else "",
            data if isinstance(data, dict) else {},
        ))
        if plan.intent not in HISTORY_EXEMPT_INTENTS:
            record_entry(history, input_text=full_input, command_text=plan.text,
                         intent=plan.intent, success=bool(result.get("success")),
                         message=message if isinstance(message, str) else None)
        outcome.final_plan = plan
        if not result.get("success"):
            outcome.stop_reason = "failure"
            return outcome
    outcome.final_plan = plans[-1]
    return outcome


def _entry(
    command_text: str, success: bool, message: str, data: dict
) -> dict:
    """Build one ordered per-command result entry (PART 13 shape)."""
    return {
        "command": command_text,
        "success": success,
        "message": message,
        "data": data,
    }
