"""
Tests for the tokenizer module (Step 21 hardening).

The tokenizer is a pure function: it splits normalized text into
tokens, never modifies valid application names, and never executes
anything.
"""

import os
import sys
import unittest
from unittest.mock import patch

# Add parent directory to path so we can import core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.input import normalize_input
from core.tokenizer import tokenize


class TestTokenizeBasics(unittest.TestCase):
    """Basic tokenization behaviour."""

    def test_basic_command(self):
        self.assertEqual(tokenize("open chrome"), ["open", "chrome"])

    def test_empty_string(self):
        self.assertEqual(tokenize(""), [])

    def test_whitespace_only_string(self):
        self.assertEqual(tokenize("   "), [])

    def test_whitespace_only_after_normalize(self):
        self.assertEqual(tokenize(normalize_input("  \t  ")), [])

    def test_single_word(self):
        self.assertEqual(tokenize("help"), ["help"])


class TestApplicationNamesPreserved(unittest.TestCase):
    """Tokenization must never mangle valid application names."""

    def test_google_chrome_stays_three_tokens(self):
        self.assertEqual(
            tokenize("open google chrome"), ["open", "google", "chrome"]
        )

    def test_file_explorer_stays_three_tokens(self):
        self.assertEqual(
            tokenize("open file explorer"), ["open", "file", "explorer"]
        )

    def test_microsoft_edge_stays_three_tokens(self):
        self.assertEqual(
            tokenize("open microsoft edge"), ["open", "microsoft", "edge"]
        )

    def test_normalization_runs_before_tokenization(self):
        raw = "  OPEN    GOOGLE    CHROME  "
        normalized = normalize_input(raw)
        self.assertEqual(normalized, "open google chrome")
        self.assertEqual(tokenize(normalized), ["open", "google", "chrome"])

    def test_repeated_spaces_collapse_before_tokenize(self):
        normalized = normalize_input("open    google    chrome")
        self.assertEqual(tokenize(normalized), ["open", "google", "chrome"])


class TestPunctuationAndShapes(unittest.TestCase):
    """Punctuation and special shapes tokenize predictably."""

    def test_trailing_punctuation_is_own_token(self):
        self.assertEqual(
            tokenize("create folder test!"), ["create", "folder", "test", "!"]
        )

    def test_punctuation_not_silently_dropped(self):
        tokens = tokenize("open chrome.")
        self.assertIn(".", tokens)

    def test_url_fragments(self):
        tokens = tokenize("open https://github.com")
        self.assertIn("https", tokens)
        self.assertIn(":", tokens)

    def test_deterministic(self):
        first = tokenize("opn google chrpme")
        second = tokenize("opn google chrpme")
        self.assertEqual(first, second)

    def test_returns_a_list_never_none(self):
        self.assertIsInstance(tokenize("help"), list)
        self.assertIsInstance(tokenize(""), list)


class TestTokenizerSafety(unittest.TestCase):
    """The tokenizer must be a pure function with no side effects."""

    def setUp(self):
        # Warm up NLTK resources so the checks below cannot be
        # confused by first-run lazy loading.
        tokenize("warmup")

    def test_no_process_or_shell_calls(self):
        with patch("subprocess.Popen") as popen, \
                patch("subprocess.run") as run, \
                patch("os.system") as system:
            tokenize("open chrome & shutdown")
            popen.assert_not_called()
            run.assert_not_called()
            system.assert_not_called()

    def test_nothing_written_to_disk(self):
        with patch("builtins.open", side_effect=AssertionError("write!")):
            tokens = tokenize("create folder x")
        self.assertEqual(tokens, ["create", "folder", "x"])

    def test_normalization_does_not_execute(self):
        with patch("subprocess.Popen") as popen, \
                patch("os.system") as system:
            normalize_input("  SHUTDOWN   COMPUTER  ")
            popen.assert_not_called()
            system.assert_not_called()


if __name__ == "__main__":
    unittest.main()
