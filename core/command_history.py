"""
Command History Module
----------------------
Bounded, in-memory, session-only record of the commands TITAN has
processed (Step 20).

Architecture:
    Command History (this module)
        ↓ observes command + result, NEVER executes anything
    main.py / multi_command_executor.py record entries
        ↓
    HISTORY / CLEAR_HISTORY commands read or clear them

Key rules:
- Memory only: nothing here is ever written to disk. History
  disappears when TITAN exits (no history.json, no database).
- Bounded: the oldest entry is dropped once the limit is reached;
  sequential IDs are never renumbered, so references stay stable
  for the lifetime of the session.
- Immutable entries: HistoryEntry is a frozen dataclass and all
  getters return copies of the containing list, so callers cannot
  corrupt the stored history.
- Sanitized storage: obvious secrets (password/token/key/...) are
  redacted BEFORE an entry is stored. The raw text is still what
  the Executor receives — sanitization applies to history only.
- Observation only: recording a history entry never modifies,
  retries, or bypasses the pipeline, and reading history (the
  HISTORY command) never replays a stored command.

Session State (core/session_state.py) remains separate:

    Session State   = current runtime context (one turn)
    Command History = bounded list of past commands (this module)
    Long-Term Memory = future feature (not implemented)
"""

import re
from dataclasses import dataclass
from datetime import datetime, timezone

from core.config_manager import get_value
from utils.logger import get_logger

logger = get_logger(__name__)

# Safe default and hard cap for the history size. The configured
# value comes from history.max_entries; anything malformed falls
# back to the default so a bad setting can never cause unlimited
# growth (or a zero-sized history).
DEFAULT_MAX_ENTRIES = 100
HARD_MAX_ENTRIES = 1000

# Value shown in place of a redacted secret.
REDACTED = "[REDACTED]"

# Intents whose own commands are never recorded in history: querying
# or clearing history is a management operation, not a meaningful
# command the user wants listed back (PART 34). Shared by main.py
# and the multi-command execution layer.
HISTORY_EXEMPT_INTENTS = frozenset({"history", "clear_history"})

# Deterministic redaction of obvious secrets in stored text.
# A sensitive keyword followed by an explicit separator (=, :, is,
# was) redacts everything after it. Conservative on purpose:
# commands without a separator after the keyword are left alone
# ("delete file token.txt" stays untouched), and no attempt is made
# at clever secret detection.
_SENSITIVE_PATTERN = re.compile(
    r"(?i)(\b(?:"
    r"passwords?|passwd|pwd|"
    r"tokens?|api[_\s-]?keys?|apikey|secrets?|"
    r"credentials?|cookies?|authorization|bearer|"
    r"private[_\s-]?keys?"
    r")\b\s*(?:=|:|is|was)\s*).+",
    re.DOTALL,
)


def sanitize_history_text(text: str) -> str:
    """
    Redact obvious secrets from text destined for history storage.

    Deterministic regex-based replacement — no ML, no heuristics
    beyond the keyword+separator pattern above.

    Examples:
        "password=abc123"            -> "password=[REDACTED]"
        "search web my password is abc123"
                                      -> "search web my password is [REDACTED]"
        "open chrome"                -> unchanged

    Args:
        text: The text to sanitize.

    Returns:
        str: The sanitized text. Non-string input is returned
        unchanged (callers coerce before storing).
    """
    if not isinstance(text, str) or not text:
        return text
    return _SENSITIVE_PATTERN.sub(r"\1" + REDACTED, text)


def _safe_text(value) -> str:
    """Coerce a value to a storable string (never raises)."""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)


def _safe_success(value) -> bool | None:
    """Coerce a value to bool|None (None means 'awaiting outcome')."""
    if value is None or isinstance(value, bool):
        return value
    return bool(value)


@dataclass(frozen=True)
class HistoryEntry:
    """
    One recorded command from the current session (immutable).

    Attributes:
        id: Sequential session-unique ID (never renumbered, never
            reused — not even after clear_history()).
        timestamp: Timezone-aware time the entry was recorded.
        input_text: The full raw input of the turn (for chains this
            is the whole original input, shared by every entry).
        command_text: The single command this entry represents.
        intent: The detected intent value, or None when planning
            failed before an intent could be detected.
        success: True (completed), False (failed/cancelled), or
            None (awaiting confirmation — the outcome is not known
            yet and must not be fabricated).
        message: The user-facing result/prompt message, if any.
    """

    id: int
    timestamp: datetime
    input_text: str
    command_text: str
    intent: str | None
    success: bool | None
    message: str | None


def entry_to_dict(entry: HistoryEntry) -> dict:
    """
    Serialize an entry for structured result data.

    Returns a fresh dict per call, so callers cannot mutate stored
    state through it.
    """
    return {
        "id": entry.id,
        "timestamp": entry.timestamp.isoformat(),
        "input": entry.input_text,
        "command": entry.command_text,
        "intent": entry.intent,
        "success": entry.success,
        "message": entry.message,
    }


def _validated_limit(value) -> int | None:
    """
    Return value as a usable history limit, or None if malformed.

    Accepts only a genuine positive int (bools excluded) and clamps
    anything above HARD_MAX_ENTRIES.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    if value > HARD_MAX_ENTRIES:
        logger.warning(
            "History limit %d exceeds the hard maximum; using %d",
            value,
            HARD_MAX_ENTRIES,
        )
        return HARD_MAX_ENTRIES
    return value


class CommandHistory:
    """
    Bounded, in-memory command history for one TITAN session.

    The internal list is only reachable through explicit methods;
    getters return copies of a list of frozen entries, so callers
    cannot corrupt the history. Nothing is ever persisted.
    """

    def __init__(self, max_entries: int | None = None) -> None:
        """
        Create an empty history.

        Args:
            max_entries: Optional explicit limit. When omitted the
                configured history.max_entries is read through the
                Configuration Manager. A malformed limit (0,
                negative, non-int, bool) falls back to the safe
                default; an excessively large value is clamped.
        """
        if max_entries is None:
            configured = get_value(
                "history", "max_entries", DEFAULT_MAX_ENTRIES
            )
            limit = _validated_limit(configured)
            if limit is None:
                logger.warning(
                    "History limit must be a positive integer; "
                    "using default %d",
                    DEFAULT_MAX_ENTRIES,
                )
                limit = DEFAULT_MAX_ENTRIES
        else:
            limit = _validated_limit(max_entries)
            if limit is None:
                logger.warning(
                    "Invalid history limit %r; using default %d",
                    max_entries,
                    DEFAULT_MAX_ENTRIES,
                )
                limit = DEFAULT_MAX_ENTRIES
        self._max_entries = limit
        self._entries: list[HistoryEntry] = []
        self._next_id = 1

    @property
    def max_entries(self) -> int:
        """The configured maximum number of retained entries."""
        return self._max_entries

    # -- Recording --------------------------------------------------------

    def add_entry(
        self,
        input_text: str,
        command_text: str,
        intent: str | None = None,
        success: bool | None = None,
        message: str | None = None,
    ) -> HistoryEntry:
        """
        Append one entry and enforce the size limit.

        The stored texts and message are sanitized (obvious secrets
        redacted). This method never raises — callers record history
        as a side effect of processing, and a history problem must
        never break a command.

        Args:
            input_text: Full raw input of the turn.
            command_text: The single command this entry represents.
            intent: Detected intent value, or None.
            success: True/False for a resolved outcome, None while
                awaiting confirmation.
            message: Result or prompt message to store.

        Returns:
            HistoryEntry: The stored (already sanitized) entry.
        """
        if isinstance(intent, str) and not intent.strip():
            intent = None
        elif intent is not None and not isinstance(intent, str):
            intent = str(intent)

        entry = HistoryEntry(
            id=self._next_id,
            timestamp=datetime.now(timezone.utc),
            input_text=sanitize_history_text(_safe_text(input_text)),
            command_text=sanitize_history_text(_safe_text(command_text)),
            intent=intent,
            success=_safe_success(success),
            message=(
                None if message is None
                else sanitize_history_text(_safe_text(message))
            ),
        )
        self._next_id += 1
        self._entries.append(entry)

        while len(self._entries) > self._max_entries:
            removed = self._entries.pop(0)
            logger.warning("History limit reached; oldest entry removed")
            logger.debug("History entry dropped (id: %d)", removed.id)

        logger.debug("History entry added (id: %d)", entry.id)
        return entry

    def update_outcome(
        self,
        entry_id: int,
        success: bool,
        message: str | None = None,
    ) -> HistoryEntry | None:
        """
        Resolve an entry's outcome (used by the confirmation flow).

        A parked dangerous command is recorded with success=None;
        once the user answers, its entry is updated to the REAL
        outcome — success is never recorded before it happens.

        Args:
            entry_id: The entry's sequential ID.
            success: The resolved outcome.
            message: The outcome message; None keeps the stored one.

        Returns:
            The updated entry, or None if the ID is unknown (for
            example when the entry was already trimmed). Never raises.
        """
        if isinstance(entry_id, bool) or not isinstance(entry_id, int):
            return None
        for index, entry in enumerate(self._entries):
            if entry.id == entry_id:
                resolved = HistoryEntry(
                    id=entry.id,
                    timestamp=entry.timestamp,
                    input_text=entry.input_text,
                    command_text=entry.command_text,
                    intent=entry.intent,
                    success=_safe_success(success),
                    message=(
                        entry.message if message is None
                        else sanitize_history_text(_safe_text(message))
                    ),
                )
                self._entries[index] = resolved
                logger.debug("History entry outcome updated (id: %d)", entry_id)
                return resolved
        return None

    # -- Reading ----------------------------------------------------------

    def get_history(self) -> list[HistoryEntry]:
        """
        Return every entry in chronological order (oldest first).

        Returns a new list of immutable entries: clearing the
        returned list can never affect the stored history.
        """
        logger.debug("History entries retrieved: %d", len(self._entries))
        return list(self._entries)

    def get_recent(self, count: int) -> list[HistoryEntry]:
        """
        Return the last `count` entries (newest last).

        Fewer entries are returned when the history is smaller;
        get_recent(0) and negative or non-int counts return [].
        """
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            return []
        if count >= len(self._entries):
            return list(self._entries)
        return list(self._entries[-count:])

    def get_entry(self, entry_id: int) -> HistoryEntry | None:
        """
        Look up one entry by its sequential ID.

        Returns:
            The entry, or None when the ID is unknown.
        """
        if isinstance(entry_id, bool) or not isinstance(entry_id, int):
            return None
        for entry in self._entries:
            if entry.id == entry_id:
                return entry
        return None

    def count(self) -> int:
        """Number of entries currently stored."""
        return len(self._entries)

    # -- Clearing ---------------------------------------------------------

    def clear_history(self) -> int:
        """
        Remove every entry (in-memory only — no files are touched).

        Returns:
            int: How many entries were removed. IDs continue with
            the next number afterwards, so they stay unique for the
            whole session.
        """
        cleared = len(self._entries)
        self._entries.clear()
        logger.info("Command history cleared (removed %d entries)", cleared)
        return cleared

    # -- Structured results ------------------------------------------------

    def get_history_result(self) -> dict:
        """
        Build the structured result for the HISTORY command.

        Returns:
            dict: {success, message, data} following the project's
            result conventions. data.kind == "history" tells the
            Response Manager to render the numbered list.
        """
        entries = self.get_history()
        if not entries:
            return {
                "success": True,
                "message": "There is no command history yet.",
                "data": {"kind": "history", "entries": []},
            }
        return {
            "success": True,
            "message": "Command history retrieved.",
            "data": {
                "kind": "history",
                "entries": [entry_to_dict(entry) for entry in entries],
            },
        }


def record_entry(
    history: "CommandHistory | None",
    *,
    input_text: str,
    command_text: str,
    intent: str | None = None,
    success: bool | None = None,
    message: str | None = None,
) -> None:
    """
    Best-effort history recording that never raises (PART 33).

    Recording is a side effect of processing a command; a history
    problem must never break the command it observes.

    Args:
        history: The history instance, or None to record nothing.
        input_text: Full raw input of the turn.
        command_text: The single command being recorded.
        intent: Detected intent value, or None.
        success: Resolved outcome, or None while awaiting.
        message: Result or prompt message.
    """
    if history is None:
        return
    try:
        history.add_entry(
            input_text=input_text,
            command_text=command_text,
            intent=intent,
            success=success,
            message=message,
        )
    except Exception:
        logger.exception("Failed to record a history entry")


# ---------------------------------------------------------------------------
# Session-wide instance
#
# main.py creates ONE CommandHistory for the running session and every
# component reads it through this accessor, so the Executor's history
# routing and the pipeline always see the same object. Tests swap the
# instance with set_command_history() instead of patching references.
# ---------------------------------------------------------------------------

_HISTORY: CommandHistory | None = None


def get_command_history() -> CommandHistory:
    """
    Return the session-wide history instance, creating it on first use.

    Returns:
        CommandHistory: The single in-memory instance for this run.
    """
    global _HISTORY
    if _HISTORY is None:
        _HISTORY = CommandHistory()
    return _HISTORY


def set_command_history(history: CommandHistory | None) -> None:
    """
    Install (or remove) the session-wide history instance.

    Args:
        history: The instance to install; None resets it so the
            next get_command_history() creates a fresh one.
    """
    global _HISTORY
    _HISTORY = history
