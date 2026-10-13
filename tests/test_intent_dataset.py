"""Dataset validation tests (Step 25 — dataset design and validation).

Verifies data/intent_dataset.csv:
- exists at the bundle-anchored location (never cwd-dependent)
- has the expected columns and no malformed rows
- has no empty text/labels
- has no exact duplicate examples
- every label is a registered intent (UNKNOWN allowed as an explicit
  "not a command" training class; the validator still rejects UNKNOWN
  for execution, so this can never authorize anything)
- every registered intent (except UNKNOWN) has a minimum number of
  examples
- UNKNOWN examples exist (the classifier must learn "not a command")
"""

import csv
import os
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.command_registry import get_all_commands, is_valid_intent
from utils import paths

MIN_EXAMPLES_PER_INTENT = 5


def _dataset_path() -> Path:
    return paths.bundle_dir() / "data" / "intent_dataset.csv"


def _load_rows() -> list[tuple[str, str]]:
    path = _dataset_path()
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["text", "intent"], (
            f"expected columns ['text', 'intent'], got {reader.fieldnames}"
        )
        return [(r["text"], r["intent"]) for r in reader]


class TestIntentDataset(unittest.TestCase):
    """Structural validation of the intent training dataset."""

    @classmethod
    def setUpClass(cls):
        cls.rows = _load_rows()

    def test_dataset_exists_and_is_not_empty(self):
        path = _dataset_path()
        self.assertTrue(path.is_file(), f"dataset missing: {path}")
        self.assertGreater(len(self.rows), 0)

    def test_no_empty_text_or_labels(self):
        for i, (text, intent) in enumerate(self.rows, start=2):
            with self.subTest(line=i):
                self.assertTrue(text.strip(), f"line {i}: empty text")
                self.assertTrue(intent.strip(), f"line {i}: empty intent")

    def test_every_label_is_registered(self):
        for i, (text, intent) in enumerate(self.rows, start=2):
            with self.subTest(line=i):
                self.assertTrue(
                    is_valid_intent(intent.strip()),
                    f"line {i}: '{intent}' is not a registered intent",
                )

    def test_no_exact_duplicates(self):
        seen: set[tuple[str, str]] = set()
        dupes = []
        for text, intent in self.rows:
            key = (text.strip().lower(), intent.strip())
            if key in seen:
                dupes.append(key)
            seen.add(key)
        self.assertEqual(dupes, [], f"duplicate examples: {dupes[:5]}")

    def test_minimum_examples_per_intent(self):
        counts = Counter(intent.strip() for _, intent in self.rows)
        registry = get_all_commands()
        for name in registry:
            if name == "UNKNOWN":
                continue
            with self.subTest(intent=name):
                self.assertGreaterEqual(
                    counts.get(name, 0),
                    MIN_EXAMPLES_PER_INTENT,
                    f"{name} has only {counts.get(name, 0)} examples",
                )

    def test_unknown_examples_exist(self):
        counts = Counter(intent.strip() for _, intent in self.rows)
        self.assertGreaterEqual(
            counts.get("UNKNOWN", 0),
            MIN_EXAMPLES_PER_INTENT,
            "dataset must teach the model what 'not a command' looks like",
        )

    def test_no_whitespace_label_problems(self):
        for i, (text, intent) in enumerate(self.rows, start=2):
            with self.subTest(line=i):
                self.assertEqual(intent, intent.strip())
                self.assertEqual(text, text.strip())

    def test_dataset_path_is_not_cwd_dependent(self):
        # bundle_dir() is anchored to the installation, never os.getcwd().
        import os as _os

        cwd = _os.getcwd()
        self.assertTrue(
            str(_dataset_path()).startswith(
                str(paths.bundle_dir())
            ),
            "dataset path must be anchored to bundle_dir(), not cwd",
        )
        self.assertNotIn(cwd.replace("\\", "/") + "/data", str(_dataset_path()))


if __name__ == "__main__":
    unittest.main()
