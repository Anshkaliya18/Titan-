"""
Response Manager Module

Converts the structured result returned by the Executor into a
clean message that can be displayed to the user.

This module does NOT execute commands and does NOT modify results.
It only formats/interprets execution results.

It also supports multi-command results (Step 19): a result carrying
an ordered "results" list is formatted by joining each per-command
message in execution order, one line per command.

And history results (Step 20): a result whose data.kind == "history"
is rendered as a simple numbered list ("1. open chrome"), with no
internal fields and no replay of anything.

Architecture:
    executor -> specialized manager -> structured result
      -> response_manager.format_response() -> user-facing text

    multi_command_executor -> combined {"results": [...]}
      -> response_manager.format_response() -> one line per command
"""

from utils.logger import get_logger

logger = get_logger(__name__)

# Safe fallback when a result cannot be interpreted
FALLBACK_MESSAGE = "Unable to process the command result."

# Shown when history is requested but no entries exist yet (never "[]")
EMPTY_HISTORY_MESSAGE = "There is no command history yet."

# Suffix for an entry whose outcome is not resolved yet (parked
# confirmation) — history must never claim a pending command ran.
AWAITING_SUFFIX = " — awaiting confirmation"


def _format_history(data: dict, fallback_message) -> str:
    """
    Render history entries as a simple numbered list (Step 20).

    Args:
        data: The history result data: {"kind": "history",
            "entries": [ {...}, ... ]}.
        fallback_message: The result's own message, used when there
            is nothing to list.

    Returns:
        str: One "N. command" line per entry in chronological order,
        the empty-history message when there are no entries, or the
        fallback for unusable data. No internal fields (timestamps,
        session ids, class names) are ever displayed.

    Notes:
        - Formatting only — this never executes, replays, or
          approves anything, and the full list is never logged.
    """
    entries = data.get("entries") if isinstance(data, dict) else None

    if not isinstance(entries, list) or not entries:
        if isinstance(fallback_message, str) and fallback_message.strip():
            return fallback_message.strip()
        return EMPTY_HISTORY_MESSAGE

    lines: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            logger.warning(
                f"Skipping malformed history entry: "
                f"{type(entry).__name__}"
            )
            continue
        display_id = entry.get("id")
        command = entry.get("command")
        if (
            isinstance(display_id, bool)
            or not isinstance(display_id, int)
            or not isinstance(command, str)
            or not command.strip()
        ):
            logger.warning("History entry has no usable display fields")
            continue
        line = f"{display_id}. {command.strip()}"
        if entry.get("success") is None:
            # Resolved outcomes never lie: a parked command reads as
            # still awaiting confirmation (PART 29).
            line += AWAITING_SUFFIX
        lines.append(line)

    if not lines:
        if isinstance(fallback_message, str) and fallback_message.strip():
            return fallback_message.strip()
        return EMPTY_HISTORY_MESSAGE

    logger.info("Response: command history (%d entries)", len(lines))
    return "\n".join(lines)


def _format_results(results: list) -> str:
    """
    Join per-command messages in execution order (multi-command).

    Args:
        results: Ordered result entries:
            [{"command": str, "success": bool, "message": str,
              "data": dict}, ...]

    Returns:
        str: One message per command, newline-separated, in the
        original order. Malformed entries are skipped; when no entry
        yields a usable message, FALLBACK_MESSAGE is returned.

    Notes:
        - Formatting only — this never executes or approves anything.
        - Order is preserved exactly as given (execution order).
    """
    messages: list[str] = []
    for entry in results:
        if not isinstance(entry, dict):
            logger.warning(
                f"Skipping malformed multi-command entry: "
                f"{type(entry).__name__}"
            )
            continue
        message = entry.get("message")
        data = entry.get("data")
        is_history = isinstance(data, dict) and data.get("kind") == "history"
        # A history command inside a chain renders as its numbered
        # list instead of its plain message (Step 20).
        if is_history:
            message = _format_history(
                data, message if isinstance(message, str) else None
            )
        if not isinstance(message, str) or not message.strip():
            logger.warning("Multi-command entry has no usable message")
            continue
        message = message.strip()
        messages.append(message)
        if is_history:
            # _format_history already logged the entry count — the
            # full history is never logged (Step 20 PART 37).
            continue
        if entry.get("success"):
            logger.info(f"Response: {message}")
        else:
            logger.warning(f"Response: {message}")

    if not messages:
        return FALLBACK_MESSAGE
    return "\n".join(messages)


def format_response(result: dict | None) -> str:
    """
    Convert a structured execution result into a user-facing message.

    Args:
        result: The structured result returned by the Executor:
            {
                "success": bool,
                "message": str,
                "data": dict,
            }
        Or a combined multi-command result (Step 19):
            {
                "success": bool,
                "results": [ {command, success, message, data}, ... ],
                "data": dict,
            }

    Returns:
        str: The message to display to the user. Malformed results
        produce a safe fallback message instead of raising. For
        multi-command results, each command's message appears on its
        own line, in execution order.

    Notes:
        - The message is returned as-is (no SUCCESS:/ERROR: prefixes).
        - Success and failure are distinguished only in the logs.
        - Confirmation-related results are displayed verbatim; this
          function never approves or executes anything.
    """
    try:
        if not isinstance(result, dict):
            logger.warning(f"Invalid result type: {type(result).__name__}")
            return FALLBACK_MESSAGE

        # Multi-command result: join the ordered per-command messages.
        results = result.get("results")
        if isinstance(results, list) and results:
            return _format_results(results)

        # History result (Step 20): render the stored entries as a
        # numbered list — never internal fields, never "[]".
        data = result.get("data")
        if isinstance(data, dict) and data.get("kind") == "history":
            return _format_history(data, result.get("message"))

        message = result.get("message")

        if not isinstance(message, str) or not message.strip():
            logger.warning("Result has no usable message")
            return FALLBACK_MESSAGE

        message = message.strip()

        if result.get("success"):
            logger.info(f"Response: {message}")
        else:
            logger.warning(f"Response: {message}")

        return message

    except Exception:
        logger.exception("Failed to format response")
        return FALLBACK_MESSAGE
