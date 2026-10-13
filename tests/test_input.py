"""
Tests for the input normalization and tokenization pipeline.
"""

import sys
import os

# Add parent directory to path so we can import core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.input import normalize_input
from core.tokenizer import tokenize


def test_normalize_basic():
    """Test basic normalization: extra spaces and mixed case."""
    assert normalize_input("  OPEN     CHROME  ") == "open chrome"


def test_normalize_empty():
    """Test that empty string is handled safely."""
    assert normalize_input("") == ""


def test_normalize_whitespace_only():
    """Test that whitespace-only string becomes empty."""
    assert normalize_input("   ") == ""


def test_normalize_mixed_case():
    """Test that mixed case is converted to lowercase."""
    assert normalize_input("Open    YouTube") == "open youtube"


def test_normalize_single_word():
    """Test single word normalization."""
    assert normalize_input("  HELLO  ") == "hello"


def test_tokenize_basic():
    """Test basic tokenization."""
    assert tokenize("open chrome") == ["open", "chrome"]


def test_tokenize_empty():
    """Test tokenization of empty string."""
    assert tokenize("") == []


def test_full_pipeline():
    """Test the complete pipeline: normalize then tokenize."""
    raw = "Open    YouTube"
    normalized = normalize_input(raw)
    tokens = tokenize(normalized)
    assert normalized == "open youtube"
    assert tokens == ["open", "youtube"]


def test_normalize_preserves_punctuation():
    """Normalization collapses spaces but keeps punctuation intact."""
    assert normalize_input("Open chrome!") == "open chrome!"


def test_normalize_collapses_tabs_and_newlines():
    """All whitespace variants collapse to single spaces."""
    assert normalize_input("open\tgoogle\tchrome") == "open google chrome"
    assert normalize_input("  open \n google  ") == "open google"


def test_normalize_keeps_single_spaces():
    """Already-normalized input passes through unchanged."""
    assert normalize_input("open google chrome") == "open google chrome"


def test_normalization_before_tokenization_order():
    """Uppercase + repeated spaces are fixed before tokens are made."""
    tokens = tokenize(normalize_input("  OPEN   GOOGLE   CHROME  "))
    assert tokens == ["open", "google", "chrome"]


if __name__ == "__main__":
    # Run all tests
    test_normalize_basic()
    test_normalize_empty()
    test_normalize_whitespace_only()
    test_normalize_mixed_case()
    test_normalize_single_word()
    test_tokenize_basic()
    test_tokenize_empty()
    test_full_pipeline()
    print("All tests passed!")
