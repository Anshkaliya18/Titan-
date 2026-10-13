"""
Session State Module

In-memory, deterministic short-term state for the current TITAN
running session. Cleared automatically when TITAN exits — nothing
is ever persisted to disk.

This is NOT long-term memory:

    Session State   = temporary state of the current process
                      (current command / session context)
    Command History = bounded list of past commands from this
                      runtime (core/command_history.py, Step 20)
    Long-Term Memory = future advanced feature (not implemented here)

Session State and Command History are deliberately separate: this
module tracks the CURRENT turn's context and the pending
confirmation, while CommandHistory owns the list of past commands.
Neither one duplicates the other.

Architecture:
    main.py owns one SessionState instance and updates it at each
    pipeline stage. Dangerous commands (shutdown, restart, delete)
    are parked here as a pending confirmation until the user answers
    the NEXT turn — "yes" executes only the exact stored command
    after revalidation, never an arbitrary new one.
"""

import copy
import time
import uuid
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from typing import Any

from core.config_manager import get_value
from core.context_manager import ContextManager
from utils.logger import get_logger

logger = get_logger(__name__)

# Default confirmation lifetime when configuration is missing/invalid.
# A stale confirmation must never sit around waiting to be triggered.
DEFAULT_CONFIRMATION_TIMEOUT_SECONDS = 60

# Exact confirmation phrases (deterministic, no fuzzy matching).
# Matching is done on the whole normalized input, so complex input
# like "yes please open chrome" is NOT treated as a confirmation.
POSITIVE_CONFIRMATION_PHRASES = frozenset({
    "yes", "y", "confirm", "confirmed", "do it", "proceed",
})
NEGATIVE_CONFIRMATION_PHRASES = frozenset({
    "no", "n", "cancel", "stop", "don't", "do not", "cancel it",
})

# State keys that are immutable for the lifetime of a session, plus
# the pending confirmation which must go through set_pending_*
# (direct writes would bypass the expiry bookkeeping).
_PROTECTED_KEYS = {
    "session_id": "immutable; use reset_session()",
    "started_at": "immutable; use reset_session()",
    "pending_confirmation": (
        "use set_pending_confirmation()/cancel_pending_confirmation()"
    ),
}


@dataclass(frozen=True)
class PendingConfirmation:
    """
    A dangerous command waiting for the user's explicit answer.

    Attributes:
        intent: The command's intent (lowercase), e.g. "shutdown_computer".
        command: A deep copy of the exact structured command that will
                 be revalidated and executed if the user confirms.
        created_at: Timezone-aware timestamp of when it was created.
    """

    intent: str
    command: dict
    created_at: datetime


@dataclass
class SessionData:
    """
    The raw state of the current session.

    session_id and started_at are fixed for the lifetime of the
    session; everything else is command-specific state that is
    updated (or cleared) per interaction.
    """

    session_id: str
    started_at: datetime
    last_input: str | None = None
    last_tokens: list[str] | None = None
    last_corrected_tokens: list[str] | None = None
    last_intent: str | None = None
    last_command: dict | None = None
    last_commands: list[dict] | None = None
    last_result: dict | None = None
    last_results: list[dict] | None = None
    last_response: str | None = None
    pending_confirmation: PendingConfirmation | None = field(
        default=None, repr=False
    )


def _confirmation_timeout_seconds() -> float:
    """
    Return the configured confirmation lifetime in seconds.

    Reads safety.confirmation_timeout_seconds through the
    Configuration Manager. Any malformed or non-positive value falls
    back to the safe default — a bad setting can never make
    confirmations immortal or instantly expiring.
    """
    value = get_value(
        "safety", "confirmation_timeout_seconds",
        DEFAULT_CONFIRMATION_TIMEOUT_SECONDS,
    )
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        return float(DEFAULT_CONFIRMATION_TIMEOUT_SECONDS)
    return float(value)


def classify_confirmation_input(text: str) -> str | None:
    """
    Classify raw user input as a confirmation answer.

    Matching is an exact, normalized phrase comparison — no fuzzy
    matching, no partial matches, no LLM interpretation.

    Args:
        text: Raw user input for the current turn.

    Returns:
        "positive", "negative", or None (not a confirmation answer).
    """
    if not isinstance(text, str):
        return None
    normalized = " ".join(text.strip().lower().split())
    if normalized in POSITIVE_CONFIRMATION_PHRASES:
        return "positive"
    if normalized in NEGATIVE_CONFIRMATION_PHRASES:
        return "negative"
    return None


class SessionState:
    """
    In-memory state for the current TITAN session.

    The internal state is only reachable through explicit methods;
    snapshot() returns copies so callers cannot accidentally mutate
    the live state. TITAN is a single-threaded CLI, so no locking
    is used — and none is needed while access stays on one thread.
    """

    def __init__(self) -> None:
        self._data = self._new_data()
        self._context = ContextManager()
        # Expiry deadline for the pending confirmation, measured on
        # the monotonic clock so wall-clock changes cannot extend it.
        self._pending_expires_at: float | None = None
        logger.debug(
            "Session state created (id: %s)", self._data.session_id
        )

    @staticmethod
    def _new_data() -> SessionData:
        return SessionData(
            session_id=uuid.uuid4().hex,
            started_at=datetime.now(timezone.utc),
        )

    # -- Identity -------------------------------------------------------

    @property
    def session_id(self) -> str:
        """Unique identifier of the current runtime session."""
        return self._data.session_id

    @property
    def started_at(self) -> datetime:
        """Timezone-aware timestamp of when the session began."""
        return self._data.started_at

    @property
    def context(self) -> ContextManager:
        """Bounded, in-memory context owned by this session."""
        return self._context

    # -- Explicit setters (command-specific state) ----------------------

    def set_last_input(self, value: str | None) -> None:
        """Record the raw input of the current turn (kept in memory only)."""
        self._data.last_input = value

    def set_last_tokens(self, tokens: list[str] | None) -> None:
        """Record the tokenizer output of the current turn."""
        self._data.last_tokens = list(tokens) if tokens is not None else None

    def set_last_corrected_tokens(
        self, tokens: list[str] | None
    ) -> None:
        """Record the spell-corrected tokens of the current turn."""
        self._data.last_corrected_tokens = (
            list(tokens) if tokens is not None else None
        )

    def set_last_intent(self, intent: str | None) -> None:
        """Record the detected intent of the current turn."""
        self._data.last_intent = intent

    def set_last_command(self, command: dict | None) -> None:
        """Record the parsed command (stored as a copy)."""
        self._data.last_command = (
            copy.deepcopy(command) if command is not None else None
        )

    def set_last_commands(self, commands: list[dict] | None) -> None:
        """
        Record every command parsed during the current turn.

        Single-command turns store a one-element list; multi-command
        turns store the whole chain in original order. This is still
        only the CURRENT turn — never a running history.
        """
        self._data.last_commands = (
            copy.deepcopy(commands) if commands is not None else None
        )

    def set_last_result(self, result: dict | None) -> None:
        """Record the structured execution result (stored as a copy)."""
        self._data.last_result = (
            copy.deepcopy(result) if result is not None else None
        )

    def set_last_results(self, results: list[dict] | None) -> None:
        """
        Record the per-command results of the current turn.

        Single-command turns store a one-element list; multi-command
        turns store every result entry in execution order. Cleared
        like every other last_* field — no history is kept.
        """
        self._data.last_results = (
            copy.deepcopy(results) if results is not None else None
        )

    def set_last_response(self, response: str | None) -> None:
        """Record the user-facing response of the current turn."""
        self._data.last_response = response

    # -- Pending confirmation ------------------------------------------

    def set_pending_confirmation(self, command: dict) -> PendingConfirmation:
        """
        Park a dangerous command until the user answers next turn.

        The command is deep-copied, so later mutations of the caller's
        dict can never alter what would be executed on "yes".

        Args:
            command: The structured command awaiting confirmation.

        Returns:
            The stored PendingConfirmation.
        """
        intent = str(command.get("intent", "")).lower()
        pending = PendingConfirmation(
            intent=intent,
            command=copy.deepcopy(dict(command)),
            created_at=datetime.now(timezone.utc),
        )
        self._data.pending_confirmation = pending
        self._pending_expires_at = (
            time.monotonic() + _confirmation_timeout_seconds()
        )
        logger.debug("Pending confirmation created: %s", intent)
        return pending

    def get_pending_confirmation(self) -> PendingConfirmation | None:
        """
        Return the pending confirmation, honouring expiration.

        An expired confirmation is cleared immediately so it can
        never be triggered afterwards.

        Returns:
            The PendingConfirmation, or None when absent or expired.
        """
        status, pending = self.resolve_pending_confirmation()
        if status == "active":
            return pending
        return None

    def resolve_pending_confirmation(
        self,
    ) -> tuple[str, PendingConfirmation | None]:
        """
        Check the pending confirmation once per turn (main-loop API).

        Distinguishes "never existed" from "expired" so the caller can
        answer the user safely (e.g. "That confirmation has expired.").
        An expired confirmation is cleared as part of this call.

        Returns:
            tuple: (status, pending) where status is one of:
                "none"    — nothing pending,
                "active"  — pending and still within its timeout,
                "expired" — pending but past its deadline (cleared;
                            pending is None in this case).
        """
        pending = self._data.pending_confirmation
        if pending is None:
            return "none", None
        if (
            self._pending_expires_at is not None
            and time.monotonic() >= self._pending_expires_at
        ):
            logger.info("Pending confirmation expired: %s", pending.intent)
            self.cancel_pending_confirmation()
            return "expired", None
        # Always hand out a copy — callers can never mutate the stored
        # command that would later be revalidated and executed.
        return "active", self._copy_pending(pending)

    @staticmethod
    def _copy_pending(
        pending: PendingConfirmation,
    ) -> PendingConfirmation:
        """Return a private copy of a pending confirmation."""
        return PendingConfirmation(
            intent=pending.intent,
            command=copy.deepcopy(pending.command),
            created_at=pending.created_at,
        )

    def cancel_pending_confirmation(self) -> bool:
        """
        Drop the pending confirmation, if any.

        Returns:
            bool: True when a confirmation was actually pending.
        """
        had_pending = self._data.pending_confirmation is not None
        self._data.pending_confirmation = None
        self._pending_expires_at = None
        return had_pending

    # -- Clearing / resetting ------------------------------------------

    def clear(self) -> None:
        """
        Clear command-specific state (PART 7 semantics).

        Resets every last_* field and any pending confirmation, while
        preserving session_id and started_at.
        """
        self._data.last_input = None
        self._data.last_tokens = None
        self._data.last_corrected_tokens = None
        self._data.last_intent = None
        self._data.last_command = None
        self._data.last_commands = None
        self._data.last_result = None
        self._data.last_results = None
        self._data.last_response = None
        self.cancel_pending_confirmation()
        self._context.clear()
        logger.debug("Session command state cleared")

    def reset_session(self) -> None:
        """
        Start a complete new session in place.

        Generates a fresh session id and start time, then clears all
        command-specific state.
        """
        self._data = self._new_data()
        self._pending_expires_at = None
        self._context.clear()
        logger.info("Session reset (new id: %s)", self._data.session_id)

    # -- Generic access -------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """
        Read a state value by key.

        Args:
            key: A SessionData field name (e.g. "last_intent").
            default: Returned when the key is unknown.

        Returns:
            The current value, the default for unknown keys, or a
            copy for mutable values (lists/dicts) so the internal
            state cannot be mutated through the return value.
        """
        if not hasattr(self._data, key):
            return default
        value = getattr(self._data, key)
        if isinstance(value, PendingConfirmation):
            # Hand out a copy — the stored command dict must never be
            # reachable for mutation from outside.
            return self._copy_pending(value)
        if isinstance(value, (list, dict)):
            return copy.deepcopy(value)
        return value

    def set(self, key: str, value: Any) -> None:
        """
        Write a state value by key.

        session_id, started_at and pending_confirmation cannot be
        written directly — set() cannot be used to smuggle arbitrary
        attributes onto the session or bypass expiry bookkeeping.

        Raises:
            KeyError: If the key is not a SessionData field.
            ValueError: If the key is protected for this session.
        """
        if not hasattr(self._data, key):
            raise KeyError(f"Unknown session state key: {key!r}")
        if key in _PROTECTED_KEYS:
            raise ValueError(
                f"Session key {key!r} is protected ({_PROTECTED_KEYS[key]})."
            )
        setattr(self._data, key, value)

    # -- Inspection -----------------------------------------------------

    def snapshot(self) -> dict:
        """
        Return a safe copy of the whole state.

        Mutable values are deep-copied and timestamps are rendered as
        ISO strings, so mutating the snapshot can never affect the
        live session.
        """
        data = self._data
        pending = data.pending_confirmation
        return {
            "session_id": data.session_id,
            "started_at": data.started_at.isoformat(),
            "last_input": data.last_input,
            "last_tokens": (
                list(data.last_tokens)
                if data.last_tokens is not None else None
            ),
            "last_corrected_tokens": (
                list(data.last_corrected_tokens)
                if data.last_corrected_tokens is not None else None
            ),
            "last_intent": data.last_intent,
            "last_command": (
                copy.deepcopy(data.last_command)
                if data.last_command is not None else None
            ),
            "last_commands": (
                copy.deepcopy(data.last_commands)
                if data.last_commands is not None else None
            ),
            "last_result": (
                copy.deepcopy(data.last_result)
                if data.last_result is not None else None
            ),
            "last_results": (
                copy.deepcopy(data.last_results)
                if data.last_results is not None else None
            ),
            "last_response": data.last_response,
            "pending_confirmation": (
                {
                    "intent": pending.intent,
                    "command": copy.deepcopy(pending.command),
                    "created_at": pending.created_at.isoformat(),
                }
                if pending is not None else None
            ),
            "context": self._context.snapshot(),
        }

    @staticmethod
    def state_keys() -> "set[str]":
        """Return the set of valid state keys."""
        return {f.name for f in fields(SessionData)}
