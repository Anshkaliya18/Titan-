"""
Tests for the spell checker module.
"""

import sys
import os

# Add parent directory to path so we can import core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.spell_checker import correct_token, correct_tokens


def test_empty_list():
    """Test that an empty list returns an empty list."""
    assert correct_tokens([]) == []


def test_empty_token():
    """Test that an empty token remains empty."""
    assert correct_token("") == ""


def test_correct_words_unchanged():
    """Test that correctly spelled words remain unchanged."""
    assert correct_tokens(["open", "chrome"]) == ["open", "chrome"]


def test_misspelled_words_corrected():
    """Test that misspelled words are corrected by the model."""
    result = correct_tokens(["opn", "chrme"])
    assert len(result) == 2
    assert result[0] == "open"
    assert result[1] == "chrome"


def test_punctuation_preserved():
    """Test that punctuation tokens are not modified."""
    assert correct_tokens(["open", "chrome", "."]) == ["open", "chrome", "."]


def test_token_order_preserved():
    """Test that token order and count are preserved."""
    tokens = ["opn", "chrme", "calclator"]
    result = correct_tokens(tokens)
    assert len(result) == len(tokens)
    assert result == ["open", "chrome", "calculator"]


def test_single_token():
    """Test correcting a single token."""
    assert correct_token("opn") == "open"
    assert correct_token("open") == "open"


def test_single_punctuation():
    """Test that a single punctuation token is preserved."""
    assert correct_token(".") == "."
    assert correct_token(",") == ","


def test_mixed_tokens():
    """Test a mix of correct, incorrect, and punctuation tokens."""
    tokens = ["opn", "chrome", "calclator", "."]
    result = correct_tokens(tokens)
    assert result == ["open", "chrome", "calculator", "."]


# --- Regression tests: protected words must not be corrected ---


def test_protected_calc_unchanged():
    """Regression: 'calc' must not be corrected to 'call'."""
    assert correct_token("calc") == "calc"
    assert correct_tokens(["open", "calc"]) == ["open", "calc"]


def test_protected_notepad_unchanged():
    """Regression: 'notepad' must not be corrected to 'notehead'."""
    assert correct_token("notepad") == "notepad"
    assert correct_tokens(["open", "notepad"]) == ["open", "notepad"]


def test_protected_explorer_unchanged():
    """Regression: 'explorer' must remain 'explorer'."""
    assert correct_token("explorer") == "explorer"
    assert correct_tokens(["open", "explorer"]) == ["open", "explorer"]


def test_protected_chrome_unchanged():
    """Regression: 'chrome' must remain 'chrome'."""
    assert correct_token("chrome") == "chrome"


def test_protected_file_unchanged():
    """Regression: 'file' must remain 'file'."""
    assert correct_token("file") == "file"


def test_protected_file_explorer_unchanged():
    """Regression: 'file explorer' must remain unchanged."""
    assert correct_tokens(["file", "explorer"]) == ["file", "explorer"]


def test_protected_firefox_unchanged():
    """Regression: 'firefox' must remain 'firefox'."""
    assert correct_token("firefox") == "firefox"


def test_protected_edge_unchanged():
    """Regression: 'edge' must remain 'edge'."""
    assert correct_token("edge") == "edge"


def test_misspelled_notrpad_corrected():
    """Regression: 'notrpad' should still be corrected to 'notepad'."""
    assert correct_token("notrpad") == "notepad"


# --- System command words (Step 10) ---


def test_lock_unchanged():
    """'lock' must remain 'lock' (system command keyword)."""
    assert correct_token("lock") == "lock"


def test_lok_corrected_to_lock():
    """'lok' must be corrected to 'lock', not 'loka'."""
    assert correct_token("lok") == "lock"


def test_shutdown_unchanged():
    """'shutdown' must remain 'shutdown' (system command keyword)."""
    assert correct_token("shutdown") == "shutdown"


def test_shutdwon_corrected_to_shutdown():
    """'shutdwon' must be corrected to 'shutdown'."""
    assert correct_token("shutdwon") == "shutdown"


def test_restart_unchanged():
    """'restart' must remain 'restart' (system command keyword)."""
    assert correct_token("restart") == "restart"


def test_reboot_unchanged():
    """'reboot' must remain 'reboot', not be corrected to 'rebook'."""
    assert correct_token("reboot") == "reboot"


# --- Browser/URL words (Step 11) ---


def test_https_unchanged():
    """'https' must remain 'https', not be corrected to 'heaps'."""
    assert correct_token("https") == "https"


def test_http_unchanged():
    """'http' must remain 'http', not be corrected to 'hatty'."""
    assert correct_token("http") == "http"


def test_website_unchanged():
    """'website' must remain 'website', not be corrected to 'weissite'."""
    assert correct_token("website") == "website"


def test_web_unchanged():
    """'web' must remain 'web'."""
    assert correct_token("web") == "web"


def test_url_unchanged():
    """'url' must remain 'url', not be corrected to 'ural'."""
    assert correct_token("url") == "url"


def test_url_tokens_unchanged():
    """Tokenized URL fragments must survive correction intact."""
    tokens = ["open", "https", ":", "//github.com"]
    assert correct_tokens(tokens) == tokens


# --- Step 21 hardening: explicit regression phrases (Phase 5) ---


def test_opn_google_chrpme_full_phrase():
    """'opn google chrpme' must correct to 'open google chrome'."""
    assert correct_tokens(["opn", "google", "chrpme"]) == [
        "open", "google", "chrome",
    ]


def test_opn_chrokme_full_phrase():
    """'opn chrokme' must correct to 'open chrome'."""
    assert correct_tokens(["opn", "chrokme"]) == ["open", "chrome"]


def test_open_calc_never_becomes_call():
    """Regression: 'open calc' must never correct 'calc' to 'call'."""
    result = correct_tokens(["open", "calc"])
    assert result == ["open", "calc"]
    assert "call" not in result


def test_open_notepad_never_becomes_notehead():
    """Regression: 'open notepad' must never produce 'notehead'."""
    result = correct_tokens(["open", "notepad"])
    assert result == ["open", "notepad"]
    assert "notehead" not in result


def test_valid_application_names_preserved_together():
    """Every supported application name survives correction untouched."""
    words = [
        "chrome", "google", "firefox", "edge", "notepad",
        "calculator", "calc", "explorer",
    ]
    assert correct_tokens(words) == words


def test_multiword_application_names_preserved():
    """Multi-word names stay word-for-word intact."""
    assert correct_tokens(["google", "chrome"]) == ["google", "chrome"]
    assert correct_tokens(["file", "explorer"]) == ["file", "explorer"]
    assert correct_tokens(["microsoft", "edge"]) == ["microsoft", "edge"]


def test_punctuation_and_numbers_untouched():
    """Punctuation and numeric tokens are never spell-corrected."""
    assert correct_tokens(["!", "?", "123", "test.txt"]) == [
        "!", "?", "123", "test.txt",
    ]


def test_unknown_word_correction_is_deterministic():
    """Unknown words never crash and always correct the same way."""
    first = correct_token("zzqqxx")
    assert isinstance(first, str) and first
    assert correct_token("zzqqxx") == first


if __name__ == "__main__":
    # Run all tests
    test_empty_list()
    test_empty_token()
    test_correct_words_unchanged()
    test_misspelled_words_corrected()
    test_punctuation_preserved()
    test_token_order_preserved()
    test_single_token()
    test_single_punctuation()
    test_mixed_tokens()
    test_protected_calc_unchanged()
    test_protected_notepad_unchanged()
    test_protected_explorer_unchanged()
    test_protected_chrome_unchanged()
    test_protected_file_unchanged()
    test_protected_file_explorer_unchanged()
    test_protected_firefox_unchanged()
    test_protected_edge_unchanged()
    test_misspelled_notrpad_corrected()
    test_lock_unchanged()
    test_lok_corrected_to_lock()
    test_shutdown_unchanged()
    test_shutdwon_corrected_to_shutdown()
    test_restart_unchanged()
    test_reboot_unchanged()
    test_https_unchanged()
    test_http_unchanged()
    test_website_unchanged()
    test_web_unchanged()
    test_url_unchanged()
    test_url_tokens_unchanged()
    print("All tests passed!")
