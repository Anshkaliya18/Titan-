"""
Hybrid Intent Detection (Step 25 — Hybrid Intent Detection)

Combines deterministic rule-based intent detection with ML-based
classification for improved natural language understanding while
preserving the deterministic safety architecture.
"""

from __future__ import annotations

from typing import Any

from core.intent import Intent, detect_intent as detect_intent_rules
from core.ml_intent_classifier import (
    MLIntentClassifier,
    get_classifier,
    initialize_classifier,
)
from core.config_manager import get_value
from utils.logger import get_logger

logger = get_logger(__name__)


class HybridIntentResult:
    """Structured result from hybrid intent detection."""

    def __init__(
        self,
        intent: Intent,
        source: str,  # "rule" or "ml"
        confidence: float = 0.0,
        rule_intent: Intent | None = None,
        ml_intent: str | None = None,
        ml_confidence: float = 0.0,
    ) -> None:
        self.intent = intent
        self.source = source
        self.confidence = confidence
        self.rule_intent = rule_intent
        self.ml_intent = ml_intent
        self.ml_confidence = ml_confidence

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/debugging."""
        return {
            "intent": self.intent.value,
            "source": self.source,
            "confidence": self.confidence,
            "rule_intent": self.rule_intent.value if self.rule_intent else None,
            "ml_intent": self.ml_intent,
            "ml_confidence": self.ml_confidence,
        }


def _get_ml_config() -> dict[str, Any]:
    """Get ML intent detection configuration from ConfigManager."""
    return {
        "enabled": get_value("intent_detection", "ml_enabled", True),
        "confidence_threshold": get_value("intent_detection", "ml_confidence_threshold", 0.50),
        "dangerous_confidence_threshold": get_value("intent_detection", "ml_dangerous_confidence_threshold", 0.85),
    }


def _get_classifier() -> tuple[bool, object]:
    """
    Get the ML classifier instance.

    Returns:
        (available, classifier_or_none): available is True if ML is enabled
        and classifier is trained/loaded.
    """
    config = _get_ml_config()
    if not config["enabled"]:
        return False, None

    try:
        classifier = get_classifier()
        if not classifier.is_trained:
            logger.debug("ML classifier not yet trained")
            return False, None
        return True, classifier
    except Exception as e:
        logger.warning("ML classifier unavailable: %s", e)
        return False, None


def detect_hybrid_intent(
    tokens: list[str],
    text: str | None = None,
    deterministic_intent: Intent | None = None,
) -> HybridIntentResult:
    """
    Detect intent using hybrid approach: deterministic rules first, then ML fallback.

    Decision logic:
    1. Run deterministic rule-based detection first
    2. If deterministic result is not UNKNOWN, use it (Rule 1)
    3. If deterministic returns UNKNOWN, try ML classification (Rule 2)
    4. Apply confidence thresholds (Rule 3)
    4. Dangerous intents require higher confidence (Rule 4)

    Args:
        tokens: Corrected tokens from spell checker
        text: Original text (for ML classification, optional)
        deterministic_intent: Pre-computed deterministic intent (optional)

    Returns:
        HybridIntentResult with the final intent and metadata
    """
    # Get deterministic intent (or use provided one)
    if deterministic_intent is not None:
        rule_intent = deterministic_intent
    else:
        rule_intent = detect_intent_rules(tokens)

    # Rule 1: Deterministic high-confidence rules win
    if rule_intent != Intent.UNKNOWN:
        logger.debug("Hybrid intent: rule matched %s", rule_intent.value)
        return HybridIntentResult(
            intent=rule_intent,
            source="rule",
            confidence=1.0,
            rule_intent=rule_intent,
        )

    # Rule 2: ML fills gaps when deterministic returns UNKNOWN
    # Use provided text or reconstruct from tokens
    ml_text = text if text is not None else " ".join(tokens)

    if not ml_text.strip():
        logger.debug("Hybrid intent: empty text, returning UNKNOWN")
        return HybridIntentResult(
            intent=Intent.UNKNOWN,
            source="ml",
            confidence=0.0,
            rule_intent=rule_intent,
        )

    # Check if ML is available
    ml_available, classifier = _get_classifier()
    if not ml_available or classifier is None:
        logger.debug("Hybrid intent: ML unavailable, returning UNKNOWN")
        return HybridIntentResult(
            intent=Intent.UNKNOWN,
            source="ml",
            confidence=0.0,
            rule_intent=rule_intent,
        )

    # Get ML config
    config = _get_ml_config()
    confidence_threshold = config["confidence_threshold"]
    dangerous_threshold = config["dangerous_confidence_threshold"]

    # Rule 2: ML classification. Guarded: any prediction failure
    # (raising classifier, broken vectorizer, missing dependency at
    # call time) falls back to UNKNOWN instead of crashing the turn.
    try:
        ml_intent, ml_confidence = classifier.predict_with_confidence(ml_text)
    except Exception as exc:
        logger.warning("ML prediction failed: %s", exc)
        return HybridIntentResult(
            intent=Intent.UNKNOWN,
            source="ml",
            confidence=0.0,
            rule_intent=rule_intent,
        )

    if ml_intent is None:
        logger.debug("Hybrid intent: ML prediction returned None")
        return HybridIntentResult(
            intent=Intent.UNKNOWN,
            source="ml",
            confidence=0.0,
            rule_intent=rule_intent,
        )

    # Rule 3: Low confidence becomes UNKNOWN
    if ml_confidence < confidence_threshold:
        logger.debug(
            "Hybrid intent: ML confidence %.3f below threshold %.3f, returning UNKNOWN",
            ml_confidence, confidence_threshold
        )
        return HybridIntentResult(
            intent=Intent.UNKNOWN,
            source="ml",
            confidence=ml_confidence,
            rule_intent=rule_intent,
            ml_intent=ml_intent,
            ml_confidence=ml_confidence,
        )

    # Convert ML intent string to Intent enum
    try:
        ml_intent_enum = Intent(ml_intent.lower())
    except ValueError:
        logger.warning("ML predicted invalid intent: %s", ml_intent)
        return HybridIntentResult(
            intent=Intent.UNKNOWN,
            source="ml",
            confidence=ml_confidence,
            rule_intent=rule_intent,
            ml_intent=ml_intent,
            ml_confidence=ml_confidence,
        )

    # Rule 4: Dangerous intents require higher confidence
    dangerous_intents = frozenset({
        Intent.SHUTDOWN_COMPUTER,
        Intent.RESTART_COMPUTER,
        Intent.DELETE_FILE,
        Intent.ENABLE_STARTUP,
        Intent.DISABLE_STARTUP,
    })

    if ml_intent_enum in dangerous_intents:
        if ml_confidence < config["dangerous_confidence_threshold"]:
            logger.debug(
                "Hybrid intent: dangerous intent %s confidence %.3f below dangerous threshold %.3f",
                ml_intent_enum.value, ml_confidence, config["dangerous_confidence_threshold"]
            )
            return HybridIntentResult(
                intent=Intent.UNKNOWN,
                source="ml",
                confidence=ml_confidence,
                rule_intent=rule_intent,
                ml_intent=ml_intent,
                ml_confidence=ml_confidence,
            )

    # Success: ML filled the gap
    logger.debug(
        "Hybrid intent: ML matched %s (confidence %.3f)",
        ml_intent_enum.value, ml_confidence
    )
    return HybridIntentResult(
        intent=ml_intent_enum,
        source="ml",
        confidence=ml_confidence,
        rule_intent=rule_intent,
        ml_intent=ml_intent,
        ml_confidence=ml_confidence,
    )


def initialize_hybrid_intent() -> None:
    """Initialize the hybrid intent system at startup."""
    try:
        initialize_classifier()
        logger.info("Hybrid intent detection initialized")
    except Exception as e:
        logger.warning("Failed to initialize hybrid intent system: %s", e)