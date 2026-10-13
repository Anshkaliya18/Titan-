"""
Multi-Command Parser Module
----------------------------
Splits a single user input into individual TITAN command segments.

    single user input → list of command segments

This module ONLY splits text. It does not execute anything, does not
call any manager, and does not interpret commands beyond checking
whether text after a separator looks like the start of a TITAN
command. Command-start knowledge is derived from the Command
Registry — there is no second command list here.

Splitting is deterministic and command-aware:

    INPUT
     ↓
    candidate separator ("and" / "then" / "after that" / "next" / , / ;)
     ↓
    does the text AFTER it look like the beginning of a TITAN
    command? (registry aliases, with typo tolerance)
     ↓
    YES → split there     NO → keep it inside the current command

so search queries ("search web cats and dogs"), file names
("open rock and roll.mp3") and paths ("C:\\Users\\Test\\and\\data.txt")
stay intact. Spell correction is NOT applied to the input here —
each returned segment is spell-corrected later, inside the ordinary
per-command pipeline.
"""

import re

from core.command_registry import get_all_commands
from core.exceptions import MultiCommandError
from core.spell_checker import correct_tokens
from utils.logger import get_logger

logger = get_logger(__name__)

# Safety limits: one input can never fan out into an unbounded chain,
# and no single segment may be arbitrarily large.
MAX_COMMANDS = 10
MAX_COMMAND_LENGTH = 500

# User-safe messages for limit violations (displayed verbatim).
TOO_MANY_COMMANDS_MESSAGE = (
    f"I can process up to {MAX_COMMANDS} commands at a time."
)
COMMAND_TOO_LONG_MESSAGE = "Command is too long to process."

# Word separators: matched only at word boundaries and only when they
# are acting as separators — never inside arguments or file names.
_WORD_SEPARATORS = frozenset({"and", "then", "next"})
_SEPARATOR_RE = re.compile(
    r"(?<!\S)after that(?=\s|$|[,;])"
    r"|(?<!\S)and(?=\s|$|[,;])"
    r"|(?<!\S)then(?=\s|$|[,;])"
    r"|(?<!\S)next(?=\s|$|[,;])"
    r"|[,;]",
    re.IGNORECASE,
)

# Dangling separators at the very END of an input — dropped before
# splitting so "open chrome and then" never leaks "and" into a
# command segment. Leading separators are dropped by the main loop.
_TRAILING_SEPARATOR_RE = re.compile(
    r"\s*(?:after that|and|then|next)\s*[,;]?\s*$"
    r"|[,;]\s*$",
    re.IGNORECASE,
)

# Punctuation stripped from a candidate word before comparing it to
# command-start words (e.g. ", open calculator" → "open").
_BOUNDARY_PUNCTUATION = ",;:"


def _derive_command_start_words() -> frozenset[str]:
    """
    Derive the set of words that can begin a TITAN command.

    Built from the FIRST word of every alias in the Command
    Registry (the single source of truth for commands), so adding a
    new command automatically teaches the splitter about it —
    no duplicated command list lives here.

    Returns:
        frozenset[str]: Lowercase first words of all registered
        aliases (e.g. "open", "search", "shutdown", "help").
    """
    words = set()
    for definition in get_all_commands().values():
        for alias in definition.aliases:
            parts = alias.split()
            if parts:
                words.add(parts[0].lower())
    return frozenset(words)


# Derived once at import — the registry is static for the process.
COMMAND_START_WORDS = _derive_command_start_words()


def split_commands(text: str) -> list[str]:
    """
    Split user input into individual command segments.

    Args:
        text: Raw user input for one turn.

    Returns:
        list[str]: Trimmed, whitespace-normalized command segments
        in their original order. Empty or blank input returns [].
        A single command returns a one-element list, so normal
        single-command input is completely unaffected.

    Raises:
        MultiCommandError: If the input would produce more than
            MAX_COMMANDS segments, or any segment exceeds
            MAX_COMMAND_LENGTH. The message is user-safe.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    # Collapse repeated whitespace once so every segment is clean
    # (" open   chrome " → "open chrome"). Segment content is
    # otherwise preserved exactly as typed.
    normalized = " ".join(text.split())

    # Drop dangling separators at the very end of the input so they
    # can never leak into a command segment ("open chrome and then"
    # → "open chrome"). Leading separators are handled below.
    while True:
        trimmed = _TRAILING_SEPARATOR_RE.sub("", normalized)
        if trimmed == normalized:
            break
        normalized = trimmed

    segments: list[str] = []
    current = 0
    scan = 0
    length = len(normalized)

    while scan < length:
        match = _SEPARATOR_RE.search(normalized, scan)
        if match is None:
            break

        left = normalized[current:match.start()]
        right = normalized[match.end():]

        if not left.strip():
            # Leading or repeated separator — pure noise, drop it
            # ("and open chrome" → "open chrome").
            current = match.end()
            scan = current
            continue

        if not right.strip():
            # Trailing separator with nothing after it — drop it
            # ("open chrome and" → "open chrome").
            segments.append(left.strip())
            current = length
            break

        if _starts_command(right):
            segments.append(left.strip())
            current = match.end()
            scan = current
            continue

        # Not a command boundary ("search web cats and dogs") —
        # keep scanning, this separator stays inside the segment.
        scan = match.end()

    if current < length:
        remainder = normalized[current:].strip()
        if remainder:
            segments.append(remainder)

    _enforce_limits(segments)
    return segments


def _starts_command(text: str) -> bool:
    """
    Decide whether text begins with a TITAN command.

    Leading separators are skipped ("then open calculator" starts a
    command), then the first meaningful word is checked against the
    registry-derived start words. A typo-tolerant check follows
    ("serch python" starts a command) — only this single candidate
    word is spell-corrected; the segment text itself is never
    modified (spell correction happens per command in the pipeline).

    Args:
        text: The text AFTER a candidate separator.

    Returns:
        bool: True when the text looks like another TITAN command.
    """
    words = text.split()
    index = 0

    while index < len(words):
        word = words[index].lower().strip(_BOUNDARY_PUNCTUATION)
        if not word:
            index += 1
            continue

        # Skip leading separator words/phrases.
        if word in _WORD_SEPARATORS:
            index += 1
            continue
        if (
            word == "after"
            and index + 1 < len(words)
            and words[index + 1].lower().strip(_BOUNDARY_PUNCTUATION)
            == "that"
        ):
            index += 2
            continue

        # First meaningful word decides — registry first, then the
        # corrected spelling of the same word.
        if word in COMMAND_START_WORDS:
            return True
        corrected = correct_tokens([word])[0].lower()
        return corrected in COMMAND_START_WORDS

    # Only separators (or nothing) followed — not a command.
    return False


def _enforce_limits(segments: list[str]) -> None:
    """
    Enforce the chain-size and segment-length safety limits.

    Args:
        segments: The candidate segments produced by splitting.

    Raises:
        MultiCommandError: When there are too many segments or a
            segment is too long. Logged as a warning first.
    """
    if len(segments) > MAX_COMMANDS:
        logger.warning(
            "Multi-command input rejected: %d commands (max %d)",
            len(segments),
            MAX_COMMANDS,
        )
        raise MultiCommandError(TOO_MANY_COMMANDS_MESSAGE)

    for segment in segments:
        if len(segment) > MAX_COMMAND_LENGTH:
            logger.warning(
                "Command segment too long: %d characters (max %d)",
                len(segment),
                MAX_COMMAND_LENGTH,
            )
            raise MultiCommandError(COMMAND_TOO_LONG_MESSAGE)
