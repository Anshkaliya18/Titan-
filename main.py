"""
Jarvis - Main Entry Point

Coordinates the complete input processing pipeline:
    get_input() → split_commands() → [per command:]
    normalize_input() → tokenize() → correct_tokens() →
    detect_intent() → parse_command() → validate_command() →
    execute_command() → format_response() → display

Multi-command handling (Step 19):
- One input may contain several commands separated by deterministic
  separators ("and", "then", "after that", "next", ";", "," — but
  only when the text after the separator starts a Jarvis command).
- A single command produces one segment and is processed exactly as
  before — the multi-command layer is transparent for normal input.
- Segments are pre-validated as a whole chain, then executed
  sequentially in order. The chain STOPS at the first failure and
  STOPS at a dangerous command (which parks as a confirmation — the
  rest of the chain is never resumed afterwards).

Session state (Step 18):
- One SessionState object records the latest state of the current
  runtime session (in-memory only, gone when Jarvis exits).
- Dangerous commands (shutdown/restart/delete) are NOT executed on
  the turn they are typed. They are parked as a pending confirmation
  in session state and the user answers on the next turn:
      "yes"  → the EXACT stored command is revalidated and executed
      "no"   → the confirmation is cancelled, nothing executes
      other  → the pending confirmation is cancelled safely and the
               new input is processed as a normal command
- "yes" with no pending confirmation never executes anything.

Command history (Step 20):
- One CommandHistory object records the commands processed during
  this session (bounded, memory-only, gone when Jarvis exits).
- Every processed command — success, failure, or parked awaiting
  confirmation — is recorded as it happens; the parked entry is
  resolved to its real outcome when the user answers.
- The history/clear history commands themselves are never recorded.
- History only observes: it never modifies, retries, or replays a
  command, and reading it never executes anything.

Command-line flags (Step 23):
- ``--version`` and ``--help`` are handled by core.cli BEFORE logging
  and the managers initialize, so they print and exit without starting
  the pipeline. They are unrelated to the conversational ``help``
  command, which is processed inside the loop below.

Error handling:
- Logging is configured before anything else so early startup messages
  (including configuration loading) are captured in the log file.
- An unexpected exception inside the loop is logged with its traceback
  and the assistant keeps running — the user sees a safe message only.
"""

import re

# Lightweight command-line flags run first, before logging and every
# manager is initialized (Step 23, Phase 21). When a flag is handled
# the process exits here; normal startup continues below untouched.
if __name__ == "__main__":
    from core.cli import handle_cli_flags  # noqa: E402

    _cli_exit_code = handle_cli_flags()
    if _cli_exit_code is not None:
        raise SystemExit(_cli_exit_code)

# Configure file logging first so even configuration loading is logged.
# Imports follow configuration deliberately (module-level side effect).
from utils.logger import configure_logging, get_logger  # noqa: E402

configure_logging()

from core.config_manager import get_value  # noqa: E402
from core.command_history import (  # noqa: E402
    HISTORY_EXEMPT_INTENTS,
    get_command_history,
)
from core.input import get_input, normalize_input  # noqa: E402
from core.tokenizer import tokenize  # noqa: E402
from core.spell_checker import correct_tokens  # noqa: E402
from core.intent import detect_intent as detect_intent_rules  # noqa: E402
from core.hybrid_intent import (  # noqa: E402
    detect_hybrid_intent,
    initialize_hybrid_intent,
    HybridIntentResult,
)
from core.natural_language import NaturalLanguageInterpreter  # noqa: E402
from core.llm_interpreter import LLMInterpreter  # noqa: E402
from core.intent import Intent  # noqa: E402
from core.command_parser import parse_command  # noqa: E402
from core.command_validator import validate_command  # noqa: E402
from core.exceptions import MultiCommandError  # noqa: E402
from core.executor import (  # noqa: E402
    CONFIRMATION_INTENTS,
    confirmation_prompt,
    execute_command,
)
from core.multi_command_parser import split_commands  # noqa: E402
from core.multi_command_executor import (  # noqa: E402
    run_multi_commands,
    run_structured_commands,
)
from core.task_planner import (  # noqa: E402
    PlanStatus,
    StepStatus,
    TaskPlan,
    TaskPlanner,
)
from core.workflow_manager import WorkflowError, WorkflowManager  # noqa: E402
from core.response_manager import format_response  # noqa: E402
from core.tts_manager import TTSManager  # noqa: E402
from core.session_state import (  # noqa: E402
    SessionState,
    classify_confirmation_input,
)

# The configuration is loaded now; apply logging again so debug-mode
# console diagnostics are enabled when general.debug is true.
configure_logging()

# Configuration via the Configuration Manager (single source of truth)
DEBUG = get_value("general", "debug", False)
ASSISTANT_NAME = get_value("general", "assistant_name", "Jarvis")

logger = get_logger("main")
_natural_language = NaturalLanguageInterpreter()
_llm_interpreter = LLMInterpreter()
_tts_manager = TTSManager()

# Intents that must go through a session-level confirmation round-trip
# before anything executes come from the Executor (single source of
# truth, shared with the multi-command execution layer). Everything
# else runs immediately after validation.

# User-facing messages for a declined confirmation. Falls back to a
# generic message for any other intent.
_CANCEL_MESSAGES = {
    "shutdown_computer": "Shutdown cancelled.",
    "restart_computer": "Restart cancelled.",
    "delete_file": "Deletion cancelled.",
    "enable_startup": "Windows startup change cancelled.",
    "disable_startup": "Windows startup change cancelled.",
}

# One session state object for the whole runtime. Memory only — it is
# never written to disk and disappears when Jarvis exits.
session = SessionState()

# One CommandHistory object for the whole session (Step 20). Memory
# only — it disappears when Jarvis exits. Pipeline code reads it
# through get_command_history() so the Executor's history routing and
# every recording site always see the same instance (tests swap it
# with set_command_history()).
history = get_command_history()
task_planner = TaskPlanner(context_manager=session.context)
workflow_manager = WorkflowManager(task_planner=task_planner)
_pending_task_confirmation: tuple[str, int] | None = None


def main():
    """Main loop for the Jarvis assistant."""
    from core.version import VERSION
    from utils.paths import is_frozen

    # Initialize hybrid intent detection (ML + rules)
    initialize_hybrid_intent()

    logger.info(
        "%s session started (id: %s, version: %s, mode: %s)",
        ASSISTANT_NAME,
        session.session_id,
        VERSION,
        "packaged" if is_frozen() else "source",
    )
    logger.debug("Command history ready (max entries: %d)", history.max_entries)
    print(f"{ASSISTANT_NAME} initialized. Type your command below.\n")
    wake_status = (
        "OFF. Type 'wake' to start."
        if get_value("wake_word", "enabled", False) is True
        else "OFF (disabled in config)."
    )
    print(f"Wake-word mode: {wake_status}\n")

    try:
        while True:
            try:
                _process_once()
            except KeyboardInterrupt:
                # Ctrl+C outside the prompt (for example mid-processing)
                logger.info("Interrupted by user")
                print("\nGoodbye!")
                return
            except Exception:
                # Never crash the assistant or expose a traceback to the
                # user — the full traceback goes to the log file only.
                logger.exception("Unexpected error in main loop")
                print("Something went wrong. Please try again.")
    finally:
        # Reached on normal return and on SystemExit (EOF) alike.
        logger.info(f"{ASSISTANT_NAME} session ended")


def _process_once():
    """
    Process a single user turn through the complete pipeline.

    When a confirmation is pending, the turn is consumed by the
    confirmation flow (accept / cancel / expire / unrelated-input)
    instead of the normal pipeline. Otherwise the input is split into
    command segments and dispatched: one segment follows the ordinary
    single-command pipeline, several follow the multi-command flow.

    Raises:
        KeyboardInterrupt: If interrupted between pipeline stages.
    """
    global _pending_task_confirmation
    # Step 1: Get raw input from user
    raw_input = get_input()

    # Record the turn's input (memory only — never persisted)
    session.set_last_input(raw_input)

    # Handle empty input (an empty turn is neither a command nor an
    # answer, so a pending confirmation survives it untouched)
    if not raw_input:
        print("No command entered.\n")
        return

    # Step 1b: Session-level confirmation round-trip. Checked BEFORE
    # splitting — an answer like "yes" is never a command chain.
    status, pending = session.resolve_pending_confirmation()
    decision = classify_confirmation_input(raw_input)

    if decision == "positive":
        if status == "active":
            _accept_pending_confirmation(pending)
            return
        if status == "expired":
            _cancel_task_plan_confirmation()
            _resolve_awaiting_history(False, "That confirmation has expired.")
            _respond("That confirmation has expired.")
            return
        _respond("There is nothing to confirm.")
        return

    if status == "active" and decision == "negative":
        session.cancel_pending_confirmation()
        _cancel_task_plan_confirmation()
        logger.info("Confirmation cancelled")
        cancel_message = _CANCEL_MESSAGES.get(pending.intent, "Cancelled.")
        _resolve_awaiting_history(False, cancel_message)
        _respond(cancel_message)
        return

    if status == "active":
        # Unrelated input while a confirmation is pending: cancel the
        # pending confirmation safely and process the new command
        # normally. The input is never treated as an answer.
        session.cancel_pending_confirmation()
        _cancel_task_plan_confirmation()
        logger.info("Pending confirmation cancelled by unrelated input")
        _resolve_awaiting_history(False, "Confirmation cancelled.")

    # Workflow management is explicit and opt-in. A run is converted back
    # into a fresh TaskPlan and enters the same structured executor path.
    workflow_request = _parse_workflow_request(raw_input)
    if workflow_request is not None:
        _handle_workflow_request(workflow_request, raw_input)
        return

    # Step 1c: Split the input into command segments (Step 19). A
    # normal single command produces exactly one segment and flows
    # through the ordinary pipeline unchanged.
    try:
        segments = split_commands(raw_input)
    except MultiCommandError as exc:
        logger.warning("Multi-command input rejected: %s", exc)
        _respond(str(exc))
        return

    if not segments:
        # The input contained only separator noise — let the ordinary
        # pipeline reject it exactly as it did before Step 19.
        segments = [raw_input]

    # Step 34 is opt-in. Existing single and multi-command turns keep their
    # established path unless structured task planning is enabled.
    if get_value("task_planning", "enabled", False) is True:
        preview_request = _task_preview_request(raw_input)
        if preview_request is not None or len(segments) > 1:
            request = preview_request if preview_request is not None else raw_input
            try:
                plan = task_planner.create_plan(
                    request, preview=preview_request is not None
                )
            except ValueError as exc:
                _respond(str(exc))
                return
            if preview_request is not None:
                _show_task_plan(plan)
                return
            _process_multi(segments, raw_input, task_plan=plan)
            return

    if len(segments) == 1:
        _process_single(segments[0])
    else:
        _process_multi(segments, raw_input)


def _process_single(command_text: str) -> None:
    """
    Process one command through the ordinary single-command pipeline.

    Args:
        command_text: A single command (already split, trimmed and
            whitespace-normalized when the input contained
            separators).
    """
    # Step 2: Normalize the input
    normalized = normalize_input(command_text)

    # Step 3: Tokenize the normalized input
    tokens = tokenize(normalized)
    session.set_last_tokens(tokens)

    # Step 4: Spell-check the tokens
    corrected_tokens = correct_tokens(tokens)
    session.set_last_corrected_tokens(corrected_tokens)
    session.context.advance()

    # Deterministic natural-language interpretation runs on each already
    # split command segment. Successful interpretations are rewritten to a
    # canonical phrase, then still pass through hybrid intent, parser,
    # validator, executor, and confirmation handling below.
    natural_result = _natural_language.interpret(command_text, corrected_tokens)
    context_resolution = None
    if natural_result.success:
        corrected_tokens = tokenize(natural_result.normalized_text)
        session.set_last_corrected_tokens(corrected_tokens)
    else:
        context_resolution = session.context.resolve_reference(command_text)
        if context_resolution is not None:
            corrected_tokens = tokenize(context_resolution.canonical_text)
            session.set_last_corrected_tokens(corrected_tokens)

    # Step 5: Detect intent (hybrid: rules + ML). The ML layer
    # receives the spell-corrected text (not the raw normalized text)
    # so typos don't degrade classification — the model was trained on
    # correctly spelled commands.
    hybrid_result = detect_hybrid_intent(
        tokens=corrected_tokens,
        text=" ".join(corrected_tokens),
    )
    llm_result = None
    if (
        not natural_result.success
        and context_resolution is None
        and hybrid_result.intent == Intent.UNKNOWN
    ):
        # Keep the remote payload minimal: include a single app label only
        # for explicitly contextual wording, never the command history.
        context = (
            session.context.llm_context()
            if any(marker in command_text.casefold() for marker in (
                "just opened", "application i opened", "app i opened",
                "application I just opened".casefold(),
            ))
            else None
        )
        llm_result = _llm_interpreter.interpret(command_text, context=context)
        if llm_result.success and llm_result.target is None:
            context_resolution = session.context.resolve_for_intent(
                llm_result.intent
            )

    intent = (
        Intent(natural_result.intent.lower())
        if natural_result.success
        else Intent(context_resolution.intent.lower())
        if context_resolution is not None
        else Intent(llm_result.intent.lower())
        if llm_result is not None and llm_result.success
        else hybrid_result.intent
    )
    session.set_last_intent(intent.value)

    # Log hybrid intent source for debugging
    logger.debug("Hybrid intent: %s", hybrid_result.to_dict())

    # Step 6: Parse the command
    command = parse_command(corrected_tokens, intent)
    if natural_result.success:
        if natural_result.target is not None:
            command["target"] = natural_result.target
        if natural_result.arguments:
            command["arguments"].update(natural_result.arguments)
    elif context_resolution is not None:
        command["target"] = context_resolution.target
        command["arguments"].update(context_resolution.arguments)
    elif llm_result is not None and llm_result.success:
        command["target"] = llm_result.target
        command["arguments"].update(llm_result.arguments)
    session.set_last_command(command)
    session.set_last_commands([command])

    # Step 7: Validate the command. A failed attempt is still what
    # the user attempted, so it is recorded in history (Step 20).
    validation = validate_command(command)
    if not validation["valid"]:
        if intent.value not in HISTORY_EXEMPT_INTENTS:
            get_command_history().add_entry(
                input_text=command_text,
                command_text=command_text,
                intent=intent.value,
                success=False,
                message=validation["message"],
            )
        _respond(
            validation["message"],
            {"validation_errors": validation["errors"]},
        )
        return

    # Step 7b: Dangerous commands park in session state and wait for
    # the user's explicit answer on the next turn. Nothing executes.
    # The history entry records success=None — the outcome is unknown
    # until the user answers (Step 20 PART 28).
    if str(command.get("intent", "")).lower() in CONFIRMATION_INTENTS:
        prompt = confirmation_prompt(command)
        if intent.value not in HISTORY_EXEMPT_INTENTS:
            get_command_history().add_entry(
                input_text=command_text,
                command_text=command_text,
                intent=intent.value,
                success=None,
                message=prompt,
            )
        session.set_pending_confirmation(command)
        prompt_result = {
            "success": False,
            "message": prompt,
            "data": {"pending_confirmation": True},
        }
        session.set_last_result(prompt_result)
        session.set_last_results([prompt_result])
        session.set_last_response(prompt)
        _respond_with_result(prompt_result)
        return

    # Step 8: Execute the command
    result = execute_command(command)
    session.context.record_success(command, result)

    # Step 8b: Record the outcome (history observes — it never
    # modified or re-executed the command). The history commands
    # themselves are exempt: querying/clearing history is not a
    # command the user wants listed back (Step 20 PART 34).
    if intent.value not in HISTORY_EXEMPT_INTENTS:
        get_command_history().add_entry(
            input_text=command_text,
            command_text=command_text,
            intent=intent.value,
            success=(
                bool(result.get("success"))
                if isinstance(result, dict) else False
            ),
            message=(
                result.get("message")
                if isinstance(result, dict) else None
            ),
        )

    # Step 9: Format and display the user-facing response
    if DEBUG:
        print(f"Tokens: {tokens}")
        print(f"Corrected Tokens: {corrected_tokens}")
        print(f"Intent: {intent.value}")
        print(f"Command: {command}")
        print(f"Result: {result}")

    _respond_with_result(result)


def _process_multi(
    segments: list[str], input_text: str, task_plan: TaskPlan | None = None
) -> None:
    """
    Plan and execute several commands from one input (Step 19).

    Every segment runs through the ordinary pipeline via the
    Multi-Command Execution Layer: the whole chain is pre-validated,
    then executed sequentially in order. The chain stops at the first
    failure, and stops at a dangerous command — which parks exactly
    like a single dangerous command (the remaining chain commands are
    deliberately NOT resumed after the confirmation answer).

    The session's CommandHistory and the full original input are
    passed along so the layer records each command as it is
    processed (Step 20 PART 32).

    Args:
        segments: Two or more command segments, in input order.
        input_text: The full original user input (stored as the
            context of every recorded entry).
    """
    logger.info("Multi-command input received")
    logger.debug("Number of commands detected: %d", len(segments))

    if task_plan is not None and not task_planner.start_plan(task_plan.plan_id):
        _respond("That task plan is no longer available.")
        return
    try:
        if task_plan is not None:
            outcome = run_structured_commands(
                [
                    {"text": step.source_text, "command": step.command}
                    for step in task_plan.steps
                ],
                history=get_command_history(),
                input_text=input_text,
                context_manager=session.context,
            )
        else:
            outcome = run_multi_commands(
                segments,
                history=get_command_history(),
                input_text=input_text,
                context_manager=session.context,
            )
    except MultiCommandError as exc:
        logger.warning("Multi-command input rejected: %s", exc)
        if task_plan is not None:
            _mark_task_plan_failed(task_plan, str(exc))
        _respond(str(exc))
        return
    except Exception:
        # Never expose a traceback — log it and answer safely, and
        # the chain stays stopped.
        logger.exception("Unexpected error in multi-command processing")
        if task_plan is not None:
            _mark_task_plan_failed(task_plan, "Task execution was interrupted.")
        _respond("I couldn't complete the next command.")
        return

    # Park a dangerous command exactly like the single-command flow.
    # No remaining-chain commands are stored: after the answer only
    # the stored command can run.
    if outcome.confirmation is not None:
        session.set_pending_confirmation(outcome.confirmation)

    if task_plan is not None:
        _apply_task_plan_outcome(task_plan, outcome)

    # Session state for the turn: the pipeline fields reflect the
    # command the turn ended on (executed, parked, or failed), and
    # the full ordered chain/result lists record this turn only.
    final = outcome.final_plan
    if final is not None:
        session.set_last_tokens(final.tokens)
        session.set_last_corrected_tokens(final.corrected_tokens)
        session.set_last_intent(final.intent)
        session.set_last_command(final.command)
    session.set_last_commands([plan.command for plan in outcome.plans])
    session.set_last_results(list(outcome.results))

    combined = {
        "success": outcome.success,
        "results": list(outcome.results),
        "data": {
            "command_count": len(outcome.plans),
            "executed_count": outcome.executed_count,
            "stop_reason": outcome.stop_reason,
        },
    }
    session.set_last_result(combined)

    response = format_response(combined)
    session.set_last_response(response)

    if DEBUG:
        for plan in outcome.plans:
            print(f"Command: {plan.text}")
            print(f"  Intent: {plan.intent}")
            print(f"  Validation: {plan.validation['message']}")

    logger.info(
        "Multi-command execution completed (%d of %d executed%s)",
        outcome.executed_count,
        len(outcome.plans),
        f", stopped: {outcome.stop_reason}" if outcome.stop_reason else "",
    )
    _display_response(response)


def _task_preview_request(text: str) -> str | None:
    """Recognize the small, explicit preview phrase without LLM guessing."""
    leading = re.fullmatch(
        r"\s*(?:plan|preview)(?:\s+this\s+task)?\s+but\s+(?:don't|do not)\s+"
        r"(?:execute|run)\s+it\s*:\s*(.+?)\s*",
        text,
        flags=re.IGNORECASE,
    )
    if leading:
        return leading.group(1).strip()
    match = re.fullmatch(
        r"\s*(?:plan(?:\s+this\s+task)?|preview(?:\s+this\s+task)?)\s*:?\s*"
        r"(.+?)\s+(?:but\s+)?(?:don't|do not)\s+(?:execute|run)\s+it\s*[.!]?\s*",
        text,
        flags=re.IGNORECASE,
    )
    return match.group(1).strip() if match else None


def _parse_workflow_request(text: str) -> tuple[str, str | None, str | None] | None:
    """Recognize only explicit, bounded workflow management phrases."""
    value = " ".join(text.strip().split())
    create = re.fullmatch(
        r"create\s+(?:a\s+)?workflow\s+(?:called|named)\s+(.+?)\s+(?:that|to)\s+(.+?)[.!]?",
        value, flags=re.IGNORECASE,
    )
    if create:
        return ("create", _clean_workflow_name(create.group(1)), create.group(2).strip())
    if re.fullmatch(r"(?:list|show|view)\s+(?:my\s+)?workflows[.!]?", value, re.IGNORECASE):
        return ("list", None, None)
    detail = re.fullmatch(
        r"(?:show|inspect|view)\s+(?:my\s+)?workflow\s+(.+?)[.!]?",
        value, re.IGNORECASE,
    )
    if detail:
        return ("show", _clean_workflow_name(detail.group(1)), None)
    for verb in ("run", "execute", "start"):
        match = re.fullmatch(rf"{verb}\s+(?:my\s+)?workflow\s+(.+?)[.!]?", value, re.IGNORECASE)
        if match:
            return ("run", _clean_workflow_name(match.group(1)), None)
        match = re.fullmatch(rf"{verb}\s+(?:my\s+)?(.+?)\s+workflow[.!]?", value, re.IGNORECASE)
        if match:
            return ("run", _clean_workflow_name(match.group(1)), None)
    for verb, operation in (("enable", "enable"), ("disable", "disable"),
                            ("delete", "delete"), ("remove", "delete")):
        match = re.fullmatch(rf"{verb}\s+(?:my\s+)?workflow\s+(.+?)[.!]?", value, re.IGNORECASE)
        if match:
            return (operation, _clean_workflow_name(match.group(1)), None)
    return None


def _clean_workflow_name(value: str) -> str:
    value = value.strip().strip("\"'")
    value = re.sub(r"\s+workflow$", "", value, flags=re.IGNORECASE).strip()
    return value


def _handle_workflow_request(request: tuple[str, str | None, str | None], input_text: str) -> None:
    """Manage saved workflow data or delegate a run to TaskPlanner/executor."""
    operation, name, actions = request
    if get_value("automation", "enabled", False) is not True:
        _respond("Automation is disabled. Set automation.enabled to true in config/settings.json to use workflows.")
        return
    try:
        if operation == "create":
            action_text = _expand_workflow_action(actions or "")
            plan = task_planner.create_workflow_plan(action_text)
            workflow = workflow_manager.create_workflow(
                name or "", [step.command for step in plan.steps],
                description=action_text,
            )
            logger.info("Workflow created from CLI request (id=%s)", workflow["id"])
            _display_response(_format_workflow(workflow, heading=f"Saved workflow: {workflow['name']}"))
            return
        if operation == "list":
            workflows = workflow_manager.list_workflows()
            if not workflows:
                _respond("No workflows are saved yet.")
            else:
                _display_response("Workflows:\n" + "\n".join(
                    f"- {item['name']} ({'enabled' if item['enabled'] else 'disabled'})"
                    for item in workflows
                ))
            return
        if operation == "show":
            workflow = workflow_manager.get_workflow(name or "")
            if workflow is None:
                raise WorkflowError("Workflow not found.")
            _display_response(_format_workflow(workflow))
            return
        if operation == "run":
            result = workflow_manager.run_workflow(name or "")
            plan = result["plan"]
            logger.info("Workflow execution started (id=%s, steps=%d)", result["workflow"]["id"], len(plan.steps))
            _process_multi([step.source_text for step in plan.steps], input_text, task_plan=plan)
            status = task_planner.get_plan_status(plan.plan_id)
            if status is not None:
                logger.info(
                    "Workflow execution finished (id=%s, status=%s, completed_steps=%d)",
                    result["workflow"]["id"], status["status"], status["completed_steps"],
                )
            return
        if operation == "enable":
            workflow = workflow_manager.enable_workflow(name or "")
            _respond(f"Workflow '{workflow['name']}' enabled.")
            return
        if operation == "disable":
            workflow = workflow_manager.disable_workflow(name or "")
            _respond(f"Workflow '{workflow['name']}' disabled.")
            return
        if operation == "delete":
            if not workflow_manager.delete_workflow(name or ""):
                raise WorkflowError("Workflow not found.")
            _respond(f"Workflow '{name}' deleted.")
            return
    except (WorkflowError, ValueError) as exc:
        _respond(str(exc))


def _expand_workflow_action(text: str) -> str:
    """Expand the simple coordinated-action form used in workflow requests."""
    action = " ".join(text.split()).rstrip(".!?")
    action = re.sub(r"\b(?:opens|launches|starts)\b", "open", action, flags=re.IGNORECASE)
    coordinated = re.fullmatch(
        r"(open|launch|start)\s+(.+?)\s+and\s+(.+)", action,
        flags=re.IGNORECASE,
    )
    if coordinated:
        from core.application_manager import is_application
        verb, first, second = coordinated.groups()
        if is_application(first.strip()):
            action = f"{verb} {first.strip()} then {verb} {second.strip()}"
    return action


def _format_workflow(workflow: dict, heading: str | None = None) -> str:
    lines = [heading or f"Workflow: {workflow['name']}"]
    for index, step in enumerate(workflow["steps"], start=1):
        intent = step["intent"].replace("_", " ").title()
        target = step.get("target")
        lines.append(f"{index}. {intent}" + (f" → {target}" if target else ""))
    if not workflow["enabled"]:
        lines.append("Status: disabled")
    return "\n".join(lines)


def _show_task_plan(plan: TaskPlan) -> None:
    """Display a read-only preview of a validated plan."""
    lines = ["Task plan:"]
    for step in plan.steps:
        command = step.command
        target = command.get("target")
        intent = command.get("intent", "").replace("_", " ").title()
        lines.append(f"{step.order}. {intent}" + (f" → {target}" if target else ""))
    _display_response("\n".join(lines))


def _apply_task_plan_outcome(task_plan: TaskPlan, outcome) -> None:
    """Mirror outcomes reported by the existing multi-command pipeline."""
    global _pending_task_confirmation
    stored = task_planner.get_plan(task_plan.plan_id)
    if stored is None:
        return
    by_text = {step.source_text: step for step in stored.steps}
    touched: set[int] = set()
    for item in outcome.results:
        step = by_text.get(item.get("command"))
        if step is None:
            continue
        touched.add(step.step_id)
        data = item.get("data") if isinstance(item.get("data"), dict) else {}
        if data.get("pending_confirmation"):
            task_planner.update_step(
                task_plan.plan_id, step.step_id,
                StepStatus.REQUIRES_CONFIRMATION,
                result={"message": item.get("message", "Confirmation required.")},
            )
            _pending_task_confirmation = (task_plan.plan_id, step.step_id)
        elif item.get("success") is True:
            task_planner.update_step(
                task_plan.plan_id, step.step_id, StepStatus.COMPLETED,
                result=item,
            )
        else:
            task_planner.update_step(
                task_plan.plan_id, step.step_id, StepStatus.FAILED,
                result=item, error=str(item.get("message", "Step failed.")),
            )
    task_planner.mark_unexecuted(
        task_plan.plan_id,
        {step.step_id for step in stored.steps} - touched,
    )
    if not touched:
        _mark_task_plan_failed(task_plan, "Plan execution was rejected before any step ran.")


def _mark_task_plan_failed(task_plan: TaskPlan, message: str) -> None:
    stored = task_planner.get_plan(task_plan.plan_id)
    if stored is None or not stored.steps:
        return
    task_planner.update_step(
        task_plan.plan_id, stored.steps[0].step_id, StepStatus.FAILED,
        error=message,
    )


def _cancel_task_plan_confirmation() -> None:
    """Cancel the parked plan step; never start any later step."""
    global _pending_task_confirmation
    pending = _pending_task_confirmation
    _pending_task_confirmation = None
    if pending is not None:
        task_planner.resolve_confirmation(*pending, accepted=False)


def cancel_task_plan(plan_id: str) -> bool:
    """Cancel a plan safely, including clearing its exact parked command."""
    global _pending_task_confirmation
    pending = _pending_task_confirmation
    if pending is not None and pending[0] == plan_id:
        session.cancel_pending_confirmation()
        _pending_task_confirmation = None
        cancelled = task_planner.resolve_confirmation(*pending, accepted=False)
        _resolve_awaiting_history(False, "Task plan cancelled.")
        return cancelled
    return task_planner.cancel_plan(plan_id)


def _accept_pending_confirmation(pending) -> None:
    """
    Execute the exact stored command behind a pending confirmation.

    The pending confirmation is cleared first (one-shot), the stored
    command is revalidated, and only then does it execute with
    confirmed=True — arbitrary user input is never executed here.

    Args:
        pending: The PendingConfirmation retrieved this turn.
    """
    global _pending_task_confirmation
    plan_confirmation = _pending_task_confirmation
    _pending_task_confirmation = None
    session.cancel_pending_confirmation()

    validation = validate_command(pending.command)
    if not validation["valid"]:
        logger.warning(
            "Pending command failed revalidation: %s", validation["message"]
        )
        _resolve_awaiting_history(False, "That command is no longer valid.")
        _respond(
            "That command is no longer valid.",
            {"validation_errors": validation["errors"]},
        )
        if plan_confirmation is not None:
            task_planner.resolve_confirmation(
                *plan_confirmation,
                accepted=True,
                result={"success": False, "message": "Command failed validation."},
            )
        return

    logger.info("Confirmation accepted: %s", pending.intent)
    session.set_last_commands([pending.command])
    result = execute_command(pending.command, confirmed=True)
    session.context.record_success(pending.command, result)
    if plan_confirmation is not None:
        task_planner.resolve_confirmation(
            *plan_confirmation, accepted=True, result=result
        )
    # Only NOW is the outcome known: the parked entry is resolved to
    # the real result — success is never recorded before execution
    # (Step 20 PART 28).
    _resolve_awaiting_history(
        bool(result.get("success")) if isinstance(result, dict) else False,
        result.get("message") if isinstance(result, dict) else None,
    )
    if plan_confirmation is not None:
        result = dict(result)
        result["message"] = (
            f"{result.get('message', 'The confirmed step finished.')} "
            "The task plan stopped here; remaining steps were not run."
        )
    _respond_with_result(result)


def _resolve_awaiting_history(success: bool, message: str | None) -> None:
    """
    Resolve the parked command's history entry to its real outcome.

    Step 18 guarantees at most one confirmation is pending at a time,
    and its entry is the newest one still marked success=None, so a
    reverse scan finds it deterministically. If it was already
    trimmed from the bounded history (or never recorded) there is
    simply nothing to update — this never raises and never touches
    any other entry.

    Args:
        success: The resolved outcome.
        message: The outcome message; None keeps the stored prompt.
    """
    history = get_command_history()
    for entry in reversed(history.get_history()):
        if entry.success is None:
            history.update_outcome(entry.id, success, message)
            logger.debug(
                "History entry %d resolved (success=%s)", entry.id, success
            )
            return
    logger.debug("No awaiting history entry to resolve")


def _respond(message: str, data: dict | None = None) -> None:
    """Record and display a session-level response."""
    _respond_with_result({
        "success": False,
        "message": message,
        "data": data if data is not None else {},
    })


def _respond_with_result(result: dict) -> None:
    """
    Record a structured result, format it, and display it.

    Single-result turns record the result both as last_result (the
    turn's combined result) and as a one-element last_results list,
    matching the multi-command turn representation.
    """
    session.set_last_result(result)
    session.set_last_results([result])
    response = format_response(result)
    session.set_last_response(response)
    _display_response(response)


def _display_response(response: str) -> None:
    """Keep printed output authoritative and optionally speak that text."""
    print(f"\n{response}")
    try:
        _tts_manager.speak(response)
    except Exception as exc:
        # Speech is an optional output device and must never affect command flow.
        logger.warning("TTS response failed (%s)", type(exc).__name__)


if __name__ == "__main__":
    main()
