"""
ML Intent Classifier (Step 25 — Hybrid Intent Detection)

A lightweight TF-IDF + Logistic Regression classifier for intent detection.
Runs entirely locally with no external dependencies beyond scikit-learn.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from typing import TYPE_CHECKING

from core.command_registry import is_valid_intent, get_all_commands
from utils import paths

# sklearn / joblib / numpy are imported lazily inside methods (Step 25
# debugging): importing them at module top adds ~1.2 s to `import main`
# and would wipe out the Step 22 startup optimization. The hybrid layer
# treats an unavailable ML stack as "ML unavailable" and falls back to
# deterministic detection, so lazy imports are also the safe failure
# mode. Type checking only — never evaluated at runtime.
if TYPE_CHECKING:  # pragma: no cover
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# Default model storage location (relative to project root or AppData)
MODEL_DIR_NAME = "models"
MODEL_FILENAME = "intent_classifier.joblib"
# Dataset is a read-only application resource, so it is anchored to the
# bundle directory (utils.paths.bundle_dir) — never to the current working
# directory. This keeps training working when Jarvis is launched from any
# directory, in both source and packaged builds.
DATASET_PATH = paths.bundle_dir() / "data" / "intent_dataset.csv"

# Confidence thresholds (configurable via config)
DEFAULT_CONFIDENCE_THRESHOLD = 0.50
DEFAULT_DANGEROUS_CONFIDENCE_THRESHOLD = 0.85

# Dangerous intents that require higher confidence
DANGEROUS_INTENTS = frozenset({
    "SHUTDOWN_COMPUTER",
    "RESTART_COMPUTER",
    "DELETE_FILE",
    "ENABLE_STARTUP",
    "DISABLE_STARTUP",
})


def _ensure_real_ml_stack() -> None:
    """Remove NLTK import shims so the real scipy/sklearn can load.

    Step 22 installs stand-in modules for scipy/sklearn so NLTK takes
    its fast guarded fallback branch at import time. Those stand-ins
    would break scikit-learn (which genuinely needs scipy), so before
    any sklearn import this removes ONLY objects whose type name is
    MissingOptionalDependency. Real modules — e.g. when something else
    already imported the real scipy — are never touched. NLTK modules
    that already took their guarded fallbacks keep them; Jarvis never
    calls the NLTK helpers that would need the real packages.
    """
    try:
        from utils import nltk_loader
    except Exception:
        return
    for name in nltk_loader.OPTIONAL_DEPENDENCY_SHIMS:
        mod = sys.modules.get(name)
        if mod is not None and type(mod).__name__ == "MissingOptionalDependency":
            del sys.modules[name]


class MLIntentClassifier:
    """
    Lightweight TF-IDF + Logistic Regression intent classifier.

    Trained once at startup (or loaded from disk), then used for fast
    prediction on corrected token sequences.
    """

    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None
        self._label_encoder: LabelEncoder | None = None
        self._classes: list[str] = []
        self._is_trained = False
        self._load_time_ms: float = 0.0

    @property
    def is_trained(self) -> bool:
        """Return True if the classifier has been trained or loaded."""
        return self._is_trained

    @property
    def classes(self) -> list[str]:
        """Return the list of known intent classes."""
        return self._classes.copy()

    @property
    def load_time_ms(self) -> float:
        """Return the model load/train time in milliseconds."""
        return self._load_time_ms

    def _get_registry_intents(self) -> list[str]:
        """Get all valid intent names from the Command Registry."""
        commands = get_all_commands()
        # Exclude UNKNOWN from training — it's the fallback class
        return [name for name in commands.keys() if name != "UNKNOWN"]

    def _load_dataset(self) -> tuple[list[str], list[str]]:
        """Load and validate the training dataset from CSV."""
        path = DATASET_PATH
        if not path.exists():
            raise FileNotFoundError(f"Training dataset not found: {path}")

        texts = []
        labels = []
        seen = set()

        with open(path, encoding="utf-8") as f:
            # Simple CSV parsing (no quoting complications in our dataset)
            lines = f.readlines()

        if not lines:
            raise ValueError("Dataset is empty")

        # Skip header
        for i, line in enumerate(lines[1:], start=2):
            line = line.strip()
            if not line:
                continue
            # Split on first comma only (intent labels don't contain commas)
            parts = line.split(",", 1)
            if len(parts) != 2:
                raise ValueError(f"Line {i}: malformed CSV row: {line}")
            text, intent = parts[0].strip(), parts[1].strip()

            if not text:
                raise ValueError(f"Line {i}: empty text")
            if not intent:
                raise ValueError(f"Line {i}: empty intent")

            # Validate intent against registry
            if not is_valid_intent(intent):
                raise ValueError(f"Line {i}: invalid intent '{intent}' (not in registry)")

            # Deduplicate
            key = (text.lower(), intent)
            if key in seen:
                continue
            seen.add(key)

            texts.append(text)
            labels.append(intent)

        if not texts:
            raise ValueError("No valid training examples after validation")

        return texts, labels

    def train(self, texts: list[str] | None = None, labels: list[str] | None = None) -> None:
        """
        Train the classifier.

        If texts/labels are not provided, loads from the default dataset.
        """
        start = time.perf_counter()

        if texts is None or labels is None:
            texts, labels = self._load_dataset()

        if not texts or not labels:
            raise ValueError("Training data is empty")

        if len(texts) != len(labels):
            raise ValueError("Texts and labels length mismatch")

        # Validate all labels against registry (UNKNOWN is allowed as an
        # explicit training class so the model learns what "not a
        # command" looks like; the validator still rejects UNKNOWN for
        # execution, so this can never authorize anything).
        valid_intents = set(self._get_registry_intents()) | {"UNKNOWN"}
        for label in labels:
            if label not in valid_intents:
                raise ValueError(f"Label '{label}' is not a valid registered intent")

        # Lazy sklearn imports (see module docstring): keeps `import main`
        # fast and makes a missing ML stack a train-time error, not an
        # import-time crash. Purge NLTK shims first so the real packages
        # load even though NLTK was imported with stand-ins.
        _ensure_real_ml_stack()
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import LabelEncoder

        # Build pipeline: TF-IDF + Logistic Regression
        self._pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),  # unigrams + bigrams
                min_df=1,
                max_df=0.95,
                sublinear_tf=True,
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                # C is intentionally high (low regularization) so the linear
                # TF-IDF model can express confident predictions on the
                # short command phrases it is trained on. With the previous
                # C=1.0 the softmax probabilities were systematically diffuse
                # (median ~0.24, max ~0.46 on held-out data) so they never
                # cleared the 0.50 confidence threshold and the ML layer was
                # effectively dead. C=100 calibrates confidence so that
                # correct predictions sit well above the threshold while
                # out-of-distribution text stays below it (verified by
                # 5-fold cross-validation, accuracy ~0.94). See the ML
                # debugging report for the calibration analysis.
                C=100.0,
                class_weight="balanced",
                solver="lbfgs",
                random_state=42,
            )),
        ])

        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(labels)
        self._classes = list(self._label_encoder.classes_)

        # Train
        self._pipeline.fit(texts, y_encoded)
        self._is_trained = True
        self._load_time_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "ML intent classifier trained: %d samples, %d classes, %.1f ms",
            len(texts), len(self._classes), self._load_time_ms
        )

    def load(self, path: Path | None = None) -> bool:
        """
        Load a trained model from disk.

        Returns True if loaded successfully, False otherwise.
        """
        start = time.perf_counter()

        if path is None:
            path = paths.data_root() / MODEL_DIR_NAME / MODEL_FILENAME

        if not path.exists():
            logger.debug("No saved model found at %s", path)
            return False

        try:
            import joblib

            # Purge NLTK shims before unpickling: the saved model holds
            # references to real sklearn classes, and unpickling imports
            # those modules by name.
            _ensure_real_ml_stack()
            data = joblib.load(path)
            self._pipeline = data["pipeline"]
            self._label_encoder = data["label_encoder"]
            self._classes = data["classes"]
            self._is_trained = True
            self._load_time_ms = (time.perf_counter() - start) * 1000
            logger.info("ML intent classifier loaded from %s (%.1f ms)", path, self._load_time_ms)
            return True
        except Exception as e:
            logger.warning("Failed to load ML model from %s: %s", path, e)
            return False

    def save(self, path: Path | None = None) -> None:
        """Save the trained model to disk."""
        if not self._is_trained or self._pipeline is None or self._label_encoder is None:
            raise RuntimeError("Cannot save untrained model")

        if path is None:
            path = paths.data_root() / MODEL_DIR_NAME / MODEL_FILENAME

        path.parent.mkdir(parents=True, exist_ok=True)
        import joblib

        data = {
            "pipeline": self._pipeline,
            "label_encoder": self._label_encoder,
            "classes": self._classes,
        }
        joblib.dump(data, path)
        logger.info("ML intent classifier saved to %s", path)

    def predict(self, text: str) -> str | None:
        """
        Predict the intent for a given text.

        Returns the predicted intent name, or None if not trained.
        """
        if not self._is_trained or self._pipeline is None:
            return None

        try:
            y_pred = self._pipeline.predict([text])
            return self._label_encoder.inverse_transform(y_pred)[0]
        except Exception as e:
            logger.warning("ML prediction failed: %s", e)
            return None

    def predict_with_confidence(self, text: str) -> tuple[str | None, float]:
        """
        Predict the intent with confidence score.

        Returns (intent_name, confidence) where confidence is 0.0-1.0.
        Returns (None, 0.0) if not trained or prediction fails.
        """
        if not self._is_trained or self._pipeline is None:
            return None, 0.0

        try:
            import numpy as np

            proba = self._pipeline.predict_proba([text])[0]
            max_idx = int(np.argmax(proba))
            confidence = float(proba[max_idx])
            intent = self._label_encoder.inverse_transform([max_idx])[0]
            return intent, confidence
        except Exception as e:
            logger.warning("ML prediction with confidence failed: %s", e)
            return None, 0.0


# Global instance for lazy initialization
_classifier: MLIntentClassifier | None = None


def get_classifier() -> MLIntentClassifier:
    """Get the global ML intent classifier instance (lazy initialization)."""
    global _classifier
    if _classifier is None:
        _classifier = MLIntentClassifier()
    return _classifier


def initialize_classifier(force_retrain: bool = False) -> MLIntentClassifier:
    """
    Initialize the ML intent classifier at startup.

    Attempts to load from disk; if not found or force_retrain=True,
    trains from the local dataset and saves.
    """
    classifier = get_classifier()

    if force_retrain or not classifier.is_trained:
        if not classifier.load():
            logger.info("No saved ML model found, training from dataset...")
            classifier.train()
            try:
                classifier.save()
            except Exception as e:
                logger.warning("Could not save trained model: %s", e)
    return classifier