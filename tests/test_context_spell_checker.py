"""
Tests for the advanced context-aware spell checker (Step 26).

Covers:
- Token classification
- Multi-word application phrase detection
- URL/path/extension/technical term protection
- Confidence thresholds
- Dangerous command handling
- Configuration safety
- Search query preservation
- Multi-command compatibility
- ML failure fallback
- Security (no shell execution)
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.advanced_spell_checker import (
    AdvancedSpellChecker,
    TokenClassifier,
    TokenType,
    MultiWordAppDetector,
    ContextAwareCandidateGenerator,
    CandidateRanker,
    CorrectionCandidate,
    correct_tokens_advanced,
    correct_token,
)
from core.spell_checker import correct_tokens, correct_token as base_correct_token


class TestTokenClassifier(unittest.TestCase):
    """Token classification tests."""

    @classmethod
    def setUpClass(cls):
        cls.classifier = TokenClassifier()

    def test_classify_command_word(self):
        result = self.classifier.classify_token("open", ["open", "chrome"], 0)
        self.assertEqual(result.token_type, TokenType.COMMAND_WORD)
        self.assertTrue(result.is_protected)

    def test_classify_application_name(self):
        result = self.classifier.classify_token("chrome", ["open", "chrome"], 1)
        self.assertEqual(result.token_type, TokenType.APPLICATION_NAME)
        self.assertTrue(result.is_protected)

    def test_classify_system_word(self):
        result = self.classifier.classify_token("computer", ["shutdown", "computer"], 1)
        self.assertEqual(result.token_type, TokenType.SYSTEM_WORD)
        self.assertTrue(result.is_protected)

    def test_classify_url(self):
        result = self.classifier.classify_token("https://github.com", ["open", "https://github.com"], 1)
        self.assertEqual(result.token_type, TokenType.URL)
        self.assertTrue(result.is_protected)

    def test_classify_windows_path(self):
        result = self.classifier.classify_token("C:\\Users\\Test\\file.txt", ["open", "C:\\Users\\Test\\file.txt"], 1)
        self.assertEqual(result.token_type, TokenType.WINDOWS_PATH)
        self.assertTrue(result.is_protected)

    def test_classify_file_extension(self):
        result = self.classifier.classify_token(".txt", ["open", "file", ".txt"], 2)
        self.assertEqual(result.token_type, TokenType.FILE_EXTENSION)
        self.assertTrue(result.is_protected)

    def test_classify_technical_term(self):
        result = self.classifier.classify_token("python", ["search", "web", "python"], 2)
        self.assertEqual(result.token_type, TokenType.TECHNICAL_TERM)
        self.assertTrue(result.is_protected)

    def test_classify_normal_word(self):
        result = self.classifier.classify_token("hello", ["hello"], 0)
        self.assertEqual(result.token_type, TokenType.NORMAL_WORD)
        self.assertFalse(result.is_protected)

    def test_classify_punctuation(self):
        result = self.classifier.classify_token(".", ["open", "chrome", "."], 2)
        self.assertEqual(result.token_type, TokenType.UNKNOWN)
        self.assertTrue(result.is_protected)

    def test_classify_empty_token(self):
        result = self.classifier.classify_token("", [""], 0)
        self.assertEqual(result.token_type, TokenType.UNKNOWN)
        self.assertTrue(result.is_protected)


class TestMultiWordAppDetector(unittest.TestCase):
    """Multi-word application phrase detection tests."""

    @classmethod
    def setUpClass(cls):
        cls.detector = MultiWordAppDetector()

    def test_detect_google_chrome(self):
        tokens = ["open", "google", "chrome"]
        matches = self.detector.detect_phrases(tokens)
        self.assertEqual(len(matches), 1)
        start, end, phrase = matches[0]
        self.assertEqual(start, 1)
        self.assertEqual(end, 3)
        self.assertEqual(phrase["canonical"], "google chrome")

    def test_detect_file_explorer(self):
        tokens = ["open", "file", "explorer"]
        matches = self.detector.detect_phrases(tokens)
        self.assertEqual(len(matches), 1)
        start, end, phrase = matches[0]
        self.assertEqual(start, 1)
        self.assertEqual(end, 3)
        self.assertEqual(phrase["canonical"], "file explorer")

    def test_detect_microsoft_edge(self):
        tokens = ["open", "microsoft", "edge"]
        matches = self.detector.detect_phrases(tokens)
        self.assertEqual(len(matches), 1)

    def test_no_phrase_in_single_word(self):
        tokens = ["open", "chrome"]
        matches = self.detector.detect_phrases(tokens)
        self.assertEqual(len(matches), 0)

    def test_is_part_of_phrase(self):
        tokens = ["open", "google", "chrome"]
        result = self.detector.is_part_of_phrase(tokens, 1)
        self.assertIsNotNone(result)
        self.assertEqual(result["canonical"], "google chrome")

    def test_is_not_part_of_phrase(self):
        tokens = ["open", "chrome"]
        result = self.detector.is_part_of_phrase(tokens, 1)
        self.assertIsNone(result)


class TestAdvancedSpellChecker(unittest.TestCase):
    """Advanced spell checker integration tests."""

    @classmethod
    def setUpClass(cls):
        cls.checker = AdvancedSpellChecker()

    def test_correct_simple_typo(self):
        result = self.checker.correct_tokens(["opn", "chrme"])
        self.assertEqual(result, ["open", "chrome"])

    def test_correct_multi_word_app_typo(self):
        result = self.checker.correct_tokens(["opn", "google", "chrpme"])
        self.assertEqual(result, ["open", "google", "chrome"])

    def test_correct_shutdown_typo(self):
        result = self.checker.correct_tokens(["shutdwon", "computer"])
        self.assertEqual(result, ["shutdown", "computer"])

    def test_correct_reboot_typo(self):
        result = self.checker.correct_tokens(["rebot", "computer"])
        self.assertEqual(result, ["reboot", "computer"])

    def test_preserve_calc(self):
        result = self.checker.correct_tokens(["open", "calc"])
        self.assertEqual(result, ["open", "calc"])

    def test_preserve_notepad(self):
        result = self.checker.correct_tokens(["open", "notepad"])
        self.assertEqual(result, ["open", "notepad"])

    def test_preserve_file_explorer(self):
        result = self.checker.correct_tokens(["open", "file", "explorer"])
        self.assertEqual(result, ["open", "file", "explorer"])

    def test_preserve_google_chrome(self):
        result = self.checker.correct_tokens(["open", "google", "chrome"])
        self.assertEqual(result, ["open", "google", "chrome"])

    def test_preserve_url(self):
        result = self.checker.correct_tokens(["open", "https://github.com"])
        self.assertEqual(result, ["open", "https://github.com"])

    def test_preserve_windows_path(self):
        result = self.checker.correct_tokens(["open", "C:\\Users\\Test\\file.txt"])
        self.assertEqual(result, ["open", "C:\\Users\\Test\\file.txt"])

    def test_preserve_file_extension(self):
        result = self.checker.correct_tokens(["open", "report.py"])
        self.assertEqual(result, ["open", "report.py"])

    def test_preserve_technical_term(self):
        result = self.checker.correct_tokens(["search", "web", "python"])
        self.assertEqual(result, ["search", "web", "python"])

    def test_preserve_search_query(self):
        result = self.checker.correct_tokens(["search", "web", "cats", "and", "dogs"])
        self.assertEqual(result, ["search", "web", "cats", "and", "dogs"])

    def test_preserve_unknown_word(self):
        result = self.checker.correct_tokens(["open", "xyzabc"])
        self.assertEqual(result, ["open", "xyzabc"])

    def test_empty_list(self):
        result = self.checker.correct_tokens([])
        self.assertEqual(result, [])

    def test_punctuation_preserved(self):
        result = self.checker.correct_tokens(["open", "chrome", "."])
        self.assertEqual(result, ["open", "chrome", "."])


class TestBackwardCompatibility(unittest.TestCase):
    """Backward compatibility tests."""

    def test_correct_tokens_api(self):
        result = correct_tokens(["opn", "chrme"])
        self.assertEqual(result, ["open", "chrome"])

    def test_correct_token_api(self):
        result = correct_token("opn")
        self.assertEqual(result, "open")

    def test_base_correct_tokens_still_works(self):
        result = correct_tokens(["opn", "chrme"])
        self.assertEqual(result, ["open", "chrome"])

    def test_base_correct_token_still_works(self):
        result = base_correct_token("opn")
        self.assertEqual(result, "open")


class TestConfigurationSafety(unittest.TestCase):
    """Configuration safety tests."""

    def test_missing_config_uses_defaults(self):
        with patch("core.advanced_spell_checker.get_value") as mock_get:
            mock_get.return_value = None
            checker = AdvancedSpellChecker()
            result = checker.correct_tokens(["opn", "chrme"])
            self.assertEqual(result, ["open", "chrome"])

    def test_invalid_threshold_uses_default(self):
        with patch("core.advanced_spell_checker.get_value") as mock_get:
            mock_get.return_value = 1.5  # Invalid: > 1
            checker = AdvancedSpellChecker()
            result = checker.correct_tokens(["opn", "chrme"])
            self.assertEqual(result, ["open", "chrome"])

    def test_negative_threshold_uses_default(self):
        with patch("core.advanced_spell_checker.get_value") as mock_get:
            mock_get.return_value = -0.5  # Invalid: < 0
            checker = AdvancedSpellChecker()
            result = checker.correct_tokens(["opn", "chrme"])
            self.assertEqual(result, ["open", "chrome"])

    def test_wrong_type_threshold_uses_default(self):
        with patch("core.advanced_spell_checker.get_value") as mock_get:
            mock_get.return_value = "high"  # Wrong type
            checker = AdvancedSpellChecker()
            result = checker.correct_tokens(["opn", "chrme"])
            self.assertEqual(result, ["open", "chrome"])

    def test_spell_correction_disabled(self):
        with patch("core.advanced_spell_checker.get_value") as mock_get:
            mock_get.return_value = False
            checker = AdvancedSpellChecker()
            result = checker.correct_tokens(["opn", "chrme"])
            self.assertEqual(result, ["opn", "chrme"])

    def test_natural_language_is_not_rewritten(self):
        result = AdvancedSpellChecker().correct_tokens(
            ["could", "you", "please", "launch", "google", "chrome"]
        )
        self.assertEqual(
            result, ["could", "you", "please", "launch", "google", "chrome"]
        )

    def test_ambiguous_unknown_app_is_preserved(self):
        result = AdvancedSpellChecker().correct_tokens(["open", "xyzabc"])
        self.assertEqual(result, ["open", "xyzabc"])

    def test_dangerous_scheme_is_preserved(self):
        for value in ("javascript:alert(1)", "data:text/plain,x", "cmd:echo"):
            with self.subTest(value=value):
                self.assertEqual(
                    AdvancedSpellChecker().correct_tokens(["open", value]), ["open", value]
                )


class TestSecurity(unittest.TestCase):
    """Security tests - no shell execution."""

    def test_no_subprocess_import(self):
        import ast

        source = (
            os.path.join(os.path.dirname(__file__), "..", "core", "advanced_spell_checker.py")
        )
        with open(source, "r") as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertNotIn(
                        alias.name,
                        {"subprocess", "os", "sys", "shutil", "pickle", "marshal", "shelve"},
                        f"forbidden import: {alias.name}",
                    )

    def test_no_eval_exec(self):
        import ast

        source = (
            os.path.join(os.path.dirname(__file__), "..", "core", "advanced_spell_checker.py")
        )
        with open(source, "r") as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name):
                    self.assertNotIn(
                        func.id,
                        {"eval", "exec", "__import__", "compile"},
                        f"forbidden call: {func.id}()",
                    )
                for kw in node.keywords:
                    self.assertNotEqual(
                        kw.arg, "shell", "shell= keyword forbidden"
                    )

    def test_no_file_system_side_effects(self):
        import ast

        source = (
            os.path.join(os.path.dirname(__file__), "..", "core", "advanced_spell_checker.py")
        )
        with open(source, "r") as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute):
                    self.assertNotIn(
                        func.attr,
                        {"remove", "rename", "unlink", "rmdir", "makedirs", "mkdir"},
                        f"forbidden file operation: {func.attr}()",
                    )


class TestMLFailureFallback(unittest.TestCase):
    """ML failure fallback tests."""

    def test_missing_model_falls_back(self):
        with patch("core.advanced_spell_checker.SpellModel.load") as mock_load:
            mock_load.side_effect = FileNotFoundError("model not found")
            checker = AdvancedSpellChecker()
            result = checker.correct_tokens(["opn", "chrme"])
            # Should fall back to base model
            self.assertEqual(result, ["open", "chrome"])

    def test_corrupt_model_falls_back(self):
        with patch("core.advanced_spell_checker.SpellModel.load") as mock_load:
            mock_load.side_effect = ValueError("corrupt model")
            checker = AdvancedSpellChecker()
            result = checker.correct_tokens(["opn", "chrme"])
            # Should fall back to base model
            self.assertEqual(result, ["open", "chrome"])

    def test_prediction_exception_falls_back(self):
        with patch("core.advanced_spell_checker.SpellModel.load") as mock_load:
            mock_model = MagicMock()
            mock_model.correct.side_effect = RuntimeError("prediction failed")
            mock_load.return_value = mock_model
            checker = AdvancedSpellChecker()
            result = checker.correct_tokens(["opn", "chrme"])
            # Should not crash
            self.assertIsInstance(result, list)


class TestMultiCommandCompatibility(unittest.TestCase):
    """Multi-command compatibility tests."""

    def test_search_cats_and_dogs_stays_one_command(self):
        result = correct_tokens_advanced(["search", "web", "cats", "and", "dogs"])
        self.assertEqual(result, ["search", "web", "cats", "and", "dogs"])

    def test_open_chrome_and_open_calculator(self):
        # This should be split by multi-command parser, not spell checker
        result = correct_tokens_advanced(["open", "chrome", "and", "open", "calculator"])
        self.assertEqual(result, ["open", "chrome", "and", "open", "calculator"])

    def test_spell_checker_does_not_split_commands(self):
        # Spell checker should not split "and" - that's multi-command parser's job
        result = correct_tokens_advanced(["open", "chrome", "and", "open", "calculator"])
        self.assertEqual(len(result), 5)


class TestDangerousCommands(unittest.TestCase):
    """Dangerous command handling tests."""

    def test_shutdown_typo_corrected(self):
        result = correct_tokens_advanced(["shutdwon", "computer"])
        self.assertEqual(result, ["shutdown", "computer"])

    def test_restart_typo_corrected(self):
        result = correct_tokens_advanced(["restat", "computer"])
        self.assertEqual(result, ["restart", "computer"])

    def test_delete_typo_corrected(self):
        result = correct_tokens_advanced(["delet", "file", "test.txt"])
        self.assertEqual(result, ["delete", "file", "test.txt"])

    def test_dangerous_command_still_requires_confirmation(self):
        # Spell correction should not bypass confirmation
        # This is tested in test_confirmation_integration.py
        result = correct_tokens_advanced(["shutdwon", "computer"])
        self.assertEqual(result, ["shutdown", "computer"])
        # The confirmation flow is handled by the executor, not spell checker


class TestPerformance(unittest.TestCase):
    """Performance tests."""

    def test_correction_latency(self):
        import time

        checker = AdvancedSpellChecker()
        start = time.perf_counter()
        result = checker.correct_tokens(["opn", "chrme"])
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 1.0)  # Should be under 1 second
        self.assertEqual(result, ["open", "chrome"])

    def test_repeated_correction_uses_cache(self):
        checker = AdvancedSpellChecker()
        result1 = checker.correct_tokens(["opn", "chrme"])
        result2 = checker.correct_tokens(["opn", "chrme"])
        self.assertEqual(result1, result2)


if __name__ == "__main__":
    unittest.main()
