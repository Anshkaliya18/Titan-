"""
Spell Checker Module

Receives a list of tokens and returns corrected tokens.
Uses a lightweight edit-distance model with protected vocabulary.

Protected words (known commands, applications, etc.) are never corrected.

Step 26: Integrates the advanced context-aware spell checker when enabled.
"""

from core.application_manager import APP_NAMES
from core.spell_model import SpellModel
from core.vocabulary import COMMAND_TYPOS, PROTECTED_WORDS
from core.config_manager import get_value
from utils.logger import get_logger

logger = get_logger(__name__)

# Common punctuation tokens that should not be spell-corrected
# Includes all ASCII punctuation characters
PUNCTUATION = {
    ".", ",", "!", "?", ":", ";", "(", ")", "[", "]", "{", "}", '"', "'",
    "@", "#", "$", "%", "^", "&", "*", "_", "+", "-", "=", "|", "<", ">",
    "/", "\\", "`", "~",
}

# Module-level model cache (loaded once, reused)
_model = None

# Step 26: advanced context-aware spell checker (lazy singleton)
_advanced = None


def _get_advanced():
    """Load the advanced context-aware spell checker once and cache it.

    Falls back to the base spell model when the advanced module cannot
    be imported (e.g. a deployment trimmed to the base model). Failure
    to load is logged; correction then continues with the base model.
    """
    global _advanced
    if _advanced is None:
        try:
            from core.advanced_spell_checker import AdvancedSpellChecker

            _advanced = AdvancedSpellChecker()
            logger.debug("Advanced context-aware spell checker loaded")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Advanced spell checker unavailable (%s); "
                "using base spell model",
                exc,
            )
            _advanced = False
    return _advanced or None


def _should_use_advanced() -> bool:
    """Decide whether the advanced (context-aware) path may run."""
    if not get_value("spell_correction", "enabled", True):
        return False
    if not get_value("spell_correction", "context_aware", True):
        return False
    return True


def load_model() -> SpellModel:
    """
    Load the spell-correction model once and cache it.

    The model's vocabulary is augmented with application names
    so that misspellings like "notrpad" can be corrected to
    "notepad".

    Returns:
        SpellModel: The loaded model instance.

    Raises:
        FileNotFoundError: If the model file does not exist.
        ValueError: If the model file is invalid.
    """
    global _model

    if _model is None:
        _model = SpellModel.load()
        _augment_with_applications(_model)

    return _model


def _augment_with_applications(model: SpellModel) -> None:
    """
    Add application names to the model's vocabulary.

    This allows the spell checker to correct misspellings of
    application names like "notrpad" -> "notepad".

    Also adds individual words from multi-word application names
    (e.g., "google" from "google chrome", "mozilla" from "mozilla firefox")
    so that misspellings of those words can be corrected.

    Args:
        model: The spell model to augment.
    """
    # Common Windows/application terms that should be in vocabulary
    extra_terms = {"cmd", "google", "mozilla"}

    # Compute base frequency once to avoid exponential growth
    base_freq = max(model.vocab.values(), default=1) * 10

    for app_name in APP_NAMES:
        if app_name not in model.vocab_set:
            model.vocab_set.add(app_name)
            model.vocab[app_name] = base_freq

        # Also add individual words from multi-word application names
        # so that misspellings like "gogle" -> "google" work
        for word in app_name.split():
            if word not in model.vocab_set:
                model.vocab_set.add(word)
                model.vocab[word] = base_freq

    # Add extra common terms
    for term in extra_terms:
        if term not in model.vocab_set:
            model.vocab_set.add(term)
            model.vocab[term] = base_freq


def correct_token(token: str) -> str:
    """
    Correct a single token using the ML model.

    Protected words (known commands, applications, etc.) are
    returned unchanged to prevent incorrect corrections like
    "calc" -> "call" or "notepad" -> "notehead".
    Known command typos (COMMAND_TYPOS) get an explicit fix.

    Step 26: When enabled, the advanced context-aware spell checker
    is used first. It considers surrounding tokens, multi-word
    application phrases, and confidence thresholds. If the advanced
    checker is unavailable or disabled, the base model is used.

    Args:
        token (str): A single word or punctuation token.

    Returns:
        str: The corrected token. Protected words, punctuation,
             and empty strings are returned unchanged.
    """
    # Don't attempt to spell-correct punctuation or empty strings
    if not token or token in PUNCTUATION:
        return token

    enabled = get_value("spell_correction", "enabled", True)
    if enabled is False:
        return token

    # Don't correct protected words (known commands, apps, etc.)
    if token.lower() in PROTECTED_WORDS:
        return token

    # Deterministic corrections for known command-word typos
    # (explicit fixes, not ML — e.g. "lok" -> "lock")
    typo_fix = COMMAND_TYPOS.get(token.lower())
    if typo_fix is not None:
        return typo_fix

    # Step 26: try the advanced context-aware spell checker first
    if _should_use_advanced():
        advanced = _get_advanced()
        if advanced is not None:
            try:
                # The advanced checker needs context, so we pass the token
                # as a single-element list. It will handle classification,
                # phrase detection, and confidence checking internally.
                result = advanced.correct_tokens([token])
                if result:
                    return result[0]
            except Exception as exc:
                logger.warning(
                    "Advanced spell correction failed (%s); "
                    "falling back to base model",
                    exc,
                )

    try:
        model = load_model()
        return model.correct(token)
    except (FileNotFoundError, ValueError) as e:
        # If the model fails to load, return the original token. The
        # warning goes to the log — never to the user's console.
        logger.warning(
            "Spell model unavailable (%s); returning original token", e
        )
        return token


def correct_tokens(tokens: list[str]) -> list[str]:
    """
    Correct a list of tokens, preserving order and count.

    Step 26: When enabled, the advanced context-aware spell checker
    processes the entire token list together, allowing it to detect
    multi-word application phrases and use context for better
    corrections. If the advanced checker is unavailable or disabled,
    each token is corrected independently using the base model.

    Args:
        tokens (list[str]): List of tokens from the tokenizer.

    Returns:
        list[str]: List of corrected tokens (same length and order).
    """
    if not tokens:
        return []

    enabled = get_value("spell_correction", "enabled", True)
    if enabled is False:
        return list(tokens)

    # Step 26: try the advanced context-aware spell checker first
    if _should_use_advanced():
        advanced = _get_advanced()
        if advanced is not None:
            try:
                result = advanced.correct_tokens(tokens)
                if result is not None and len(result) == len(tokens):
                    logger.debug(
                        "Advanced spell correction completed. "
                        "Original token count: %d, corrected token count: %d",
                        len(tokens),
                        len(result),
                    )
                    changes = [
                        (original, fixed)
                        for original, fixed in zip(tokens, result)
                        if original != fixed
                    ]
                    if changes:
                        logger.debug(f"Corrections applied: {changes}")
                    return result
            except Exception as exc:
                logger.warning(
                    "Advanced spell correction failed (%s); "
                    "falling back to base model",
                    exc,
                )

    # Base model: correct each token independently
    corrected = [correct_token(token) for token in tokens]
    logger.debug(
        "Spell correction completed. Original token count: %d, "
        "corrected token count: %d",
        len(tokens),
        len(corrected),
    )
    changes = [
        (original, fixed)
        for original, fixed in zip(tokens, corrected)
        if original != fixed
    ]
    if changes:
        logger.debug(f"Corrections applied: {changes}")
    return corrected
