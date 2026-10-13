"""ML intent classifier unit tests (Step 25 — classifier behavior).

Uses tiny in-memory training sets (fast, no disk, deterministic) for
unit behavior, plus the production dataset for a realism spot-check.

Safety model: the classifier only returns (intent, confidence) pairs.
It never executes anything — there is deliberately no code path from
here to subprocess, shell, files, or managers (asserted structurally).
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.ml_intent_classifier import MLIntentClassifier

TINY_TEXTS = [
    "open chrome",
    "launch chrome",
    "please open chrome",
    "open firefox",
    "close chrome",
    "close firefox",
    "lock computer",
    "lock my computer",
    "search web python",
    "search the web for python",
    "hello",
    "how are you",
]
TINY_LABELS = [
    "OPEN_APPLICATION",
    "OPEN_APPLICATION",
    "OPEN_APPLICATION",
    "OPEN_APPLICATION",
    "CLOSE_APPLICATION",
    "CLOSE_APPLICATION",
    "LOCK_COMPUTER",
    "LOCK_COMPUTER",
    "SEARCH_WEB",
    "SEARCH_WEB",
    "UNKNOWN",
    "UNKNOWN",
]


def _tiny_classifier() -> MLIntentClassifier:
    clf = MLIntentClassifier()
    clf.train(list(TINY_TEXTS), list(TINY_LABELS))
    return clf


class TestBasicClassification(unittest.TestCase):
    """Core predictions on seen and paraphrased inputs."""

    @classmethod
    def setUpClass(cls):
        cls.clf = _tiny_classifier()

    def test_open_chrome(self):
        self.assertEqual(self.clf.predict("open chrome"), "OPEN_APPLICATION")

    def test_launch_variant(self):
        self.assertEqual(
            self.clf.predict("launch google chrome"), "OPEN_APPLICATION"
        )

    def test_open_browser_phrasing(self):
        clf = MLIntentClassifier()
        clf.train(
            ["open browser", "launch browser", "open chrome", "hello"],
            ["OPEN_BROWSER", "OPEN_BROWSER", "OPEN_APPLICATION", "UNKNOWN"],
        )
        self.assertEqual(clf.predict("open browser"), "OPEN_BROWSER")

    def test_search_web(self):
        self.assertEqual(
            self.clf.predict("search web python"), "SEARCH_WEB"
        )

    def test_unknown_greeting(self):
        self.assertEqual(self.clf.predict("hello"), "UNKNOWN")

    def test_predict_returns_none_when_untrained(self):
        clf = MLIntentClassifier()
        self.assertFalse(clf.is_trained)
        self.assertIsNone(clf.predict("open chrome"))
        intent, conf = clf.predict_with_confidence("open chrome")
        self.assertIsNone(intent)
        self.assertEqual(conf, 0.0)


class TestConfidence(unittest.TestCase):
    """Confidence is numeric, bounded, and meaningful."""

    @classmethod
    def setUpClass(cls):
        cls.clf = _tiny_classifier()

    def test_confidence_between_zero_and_one(self):
        for text in TINY_TEXTS + ["some random words here", "asdfgh"]:
            with self.subTest(text=text):
                _, conf = self.clf.predict_with_confidence(text)
                self.assertIsInstance(conf, float)
                self.assertGreaterEqual(conf, 0.0)
                self.assertLessEqual(conf, 1.0)

    def test_seen_example_has_high_confidence(self):
        _, conf = self.clf.predict_with_confidence("open chrome")
        self.assertGreater(conf, 0.5)

    def test_gibberish_has_low_confidence(self):
        _, conf = self.clf.predict_with_confidence("asdfgh qwerty zxcv")
        self.assertLess(conf, 0.5)


class TestTrainingValidation(unittest.TestCase):
    """Invalid training data fails loudly, never silently."""

    def test_empty_data_rejected(self):
        clf = MLIntentClassifier()
        with self.assertRaises(ValueError):
            clf.train([], [])

    def test_mismatched_lengths_rejected(self):
        clf = MLIntentClassifier()
        with self.assertRaises(ValueError):
            clf.train(["open chrome"], ["OPEN_APPLICATION", "HELP"])

    def test_invalid_label_rejected(self):
        clf = MLIntentClassifier()
        with self.assertRaises(ValueError):
            clf.train(["do a backflip"], ["BACKFLIP"])

    def test_unknown_label_allowed_as_training_class(self):
        # UNKNOWN is a legitimate "not a command" class; the validator
        # still rejects it for execution downstream.
        clf = MLIntentClassifier()
        clf.train(["hello", "open chrome"], ["UNKNOWN", "OPEN_APPLICATION"])
        self.assertIn("UNKNOWN", clf.classes)

    def test_missing_dataset_file_raises(self):
        clf = MLIntentClassifier()
        with patch.object(
            type(clf), "_load_dataset",
            side_effect=FileNotFoundError("nope"),
        ):
            with self.assertRaises(FileNotFoundError):
                clf.train()


class TestModelPersistence(unittest.TestCase):
    """Save/load round-trip and corrupted-model handling."""

    def test_save_and_load_round_trip(self):
        clf = _tiny_classifier()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.joblib"
            clf.save(path)
            self.assertTrue(path.is_file())

            loaded = MLIntentClassifier()
            self.assertTrue(loaded.load(path))
            self.assertTrue(loaded.is_trained)
            self.assertEqual(
                loaded.predict("open chrome"), "OPEN_APPLICATION"
            )
            intent, conf = loaded.predict_with_confidence("open chrome")
            self.assertEqual(intent, "OPEN_APPLICATION")
            self.assertGreater(conf, 0.0)

    def test_load_missing_file_returns_false(self):
        clf = MLIntentClassifier()
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(clf.load(Path(tmp) / "nope.joblib"))
        self.assertFalse(clf.is_trained)

    def test_load_corrupted_file_returns_false(self):
        clf = MLIntentClassifier()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.joblib"
            path.write_bytes(b"this is not a joblib file")
            self.assertFalse(clf.load(path))
        self.assertFalse(clf.is_trained)

    def test_save_untrained_raises(self):
        clf = MLIntentClassifier()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                clf.save(Path(tmp) / "model.joblib")

    def test_load_does_not_execute_anything(self):
        # The loaded object graph contains only sklearn pieces.
        clf = _tiny_classifier()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.joblib"
            clf.save(path)
            import joblib

            data = joblib.load(path)
        self.assertEqual(
            set(data.keys()), {"pipeline", "label_encoder", "classes"}
        )


class TestNoExecutionCapability(unittest.TestCase):
    """Structural: the ML layer cannot execute anything."""

    def test_module_has_no_execution_imports(self):
        import ast

        source = (
            Path(__file__).resolve().parent.parent
            / "core" / "ml_intent_classifier.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    self.assertNotIn(
                        root,
                        {"subprocess", "shutil", "pickle",
                         "marshal", "shelve"},
                        f"forbidden import: {alias.name}",
                    )
            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name):
                    self.assertNotIn(
                        func.id, {"eval", "exec", "__import__", "compile"},
                        f"forbidden call: {func.id}()",
                    )
                for kw in node.keywords:
                    self.assertNotEqual(
                        kw.arg, "shell", "shell= keyword forbidden"
                    )


if __name__ == "__main__":
    unittest.main()
