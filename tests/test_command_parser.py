"""
Tests for the command parser module.
"""

import sys
import os

# Add parent directory to path so we can import core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.intent import Intent
from core.command_parser import parse_command


def test_open_application():
    """Test parsing open application command."""
    result = parse_command(["open", "chrome"], Intent.OPEN_APPLICATION)
    assert result["intent"] == "open_application"
    assert result["target"] == "chrome"
    assert result["arguments"] == {}


def test_multi_word_application():
    """Test parsing multi-word application name."""
    result = parse_command(["open", "google", "chrome"], Intent.OPEN_APPLICATION)
    assert result["intent"] == "open_application"
    assert result["target"] == "google chrome"
    assert result["arguments"] == {}


def test_close_application():
    """Test parsing close application command."""
    result = parse_command(["close", "chrome"], Intent.CLOSE_APPLICATION)
    assert result["intent"] == "close_application"
    assert result["target"] == "chrome"
    assert result["arguments"] == {}


def test_create_folder():
    """Test parsing create folder command."""
    result = parse_command(["create", "folder", "projects"], Intent.CREATE_FOLDER)
    assert result["intent"] == "create_folder"
    assert result["target"] == "projects"
    assert result["arguments"] == {}


def test_delete_file():
    """Test parsing delete file command."""
    result = parse_command(["delete", "file", "test.txt"], Intent.DELETE_FILE)
    assert result["intent"] == "delete_file"
    assert result["target"] == "test.txt"
    assert result["arguments"] == {}


def test_open_file():
    """Test parsing open file command."""
    result = parse_command(["open", "file", "notes.txt"], Intent.OPEN_FILE)
    assert result["intent"] == "open_file"
    assert result["target"] == "notes.txt"
    assert result["arguments"] == {}


def test_search_file():
    """Test parsing search file command."""
    result = parse_command(["search", "file", "report.pdf"], Intent.SEARCH_FILE)
    assert result["intent"] == "search_file"
    assert result["target"] == "report.pdf"
    assert result["arguments"] == {}


def test_lock_computer():
    """Test parsing lock computer command (system commands have no target)."""
    result = parse_command(["lock", "computer"], Intent.LOCK_COMPUTER)
    assert result["intent"] == "lock_computer"
    assert result["target"] is None
    assert result["arguments"] == {}


def test_lock_my_computer():
    """Test that filler words do not become a target for system commands."""
    result = parse_command(["lock", "my", "computer"], Intent.LOCK_COMPUTER)
    assert result["intent"] == "lock_computer"
    assert result["target"] is None


def test_shutdown_computer():
    """Test parsing shutdown computer command (system commands have no target)."""
    result = parse_command(["shutdown", "computer"], Intent.SHUTDOWN_COMPUTER)
    assert result["intent"] == "shutdown_computer"
    assert result["target"] is None
    assert result["arguments"] == {}


def test_shut_down_computer():
    """Test parsing the two-word 'shut down' variant."""
    result = parse_command(["shut", "down", "computer"], Intent.SHUTDOWN_COMPUTER)
    assert result["intent"] == "shutdown_computer"
    assert result["target"] is None


def test_restart_computer():
    """Test parsing restart computer command (system commands have no target)."""
    result = parse_command(["restart", "computer"], Intent.RESTART_COMPUTER)
    assert result["intent"] == "restart_computer"
    assert result["target"] is None
    assert result["arguments"] == {}


def test_open_browser():
    """Test parsing open browser command (no target)."""
    result = parse_command(["open", "browser"], Intent.OPEN_BROWSER)
    assert result["intent"] == "open_browser"
    assert result["target"] is None
    assert result["arguments"] == {}


def test_open_default_browser():
    """Test 'open default browser' also has no target."""
    result = parse_command(
        ["open", "default", "browser"], Intent.OPEN_BROWSER
    )
    assert result["intent"] == "open_browser"
    assert result["target"] is None


def test_open_url():
    """Test parsing 'open https://github.com' (URL reassembled)."""
    result = parse_command(
        ["open", "https", ":", "//github.com"], Intent.OPEN_URL
    )
    assert result["intent"] == "open_url"
    assert result["target"] == "https://github.com"
    assert result["arguments"] == {}


def test_open_url_with_path():
    """Test a URL with a path survives reassembly."""
    result = parse_command(
        ["open", "https", ":", "//github.com/user/repo"], Intent.OPEN_URL
    )
    assert result["target"] == "https://github.com/user/repo"


def test_search_web():
    """Test parsing 'search web python tutorials'."""
    result = parse_command(
        ["search", "web", "python", "tutorials"], Intent.SEARCH_WEB
    )
    assert result["intent"] == "search_web"
    assert result["target"] == "python tutorials"
    assert result["arguments"] == {}


def test_search_the_web_for():
    """Test filler words are stripped from the search query."""
    result = parse_command(
        ["search", "the", "web", "for", "python", "tutorials"],
        Intent.SEARCH_WEB,
    )
    assert result["target"] == "python tutorials"


def test_google_search():
    """Test parsing 'google python tutorials'."""
    result = parse_command(
        ["google", "python", "tutorials"], Intent.SEARCH_WEB
    )
    assert result["intent"] == "search_web"
    assert result["target"] == "python tutorials"


def test_missing_target():
    """Test handling of missing target."""
    result = parse_command(["open"], Intent.OPEN_APPLICATION)
    assert result["intent"] == "open_application"
    assert result["target"] is None
    assert result["arguments"] == {}


def test_unknown_intent():
    """Test parsing with unknown intent."""
    result = parse_command(["hello", "TITAN"], Intent.UNKNOWN)
    assert result["intent"] == "unknown"
    assert result["target"] is None
    assert result["arguments"] == {}


def test_empty_tokens():
    """Test parsing with empty tokens."""
    result = parse_command([], Intent.UNKNOWN)
    assert result["intent"] == "unknown"
    assert result["target"] is None
    assert result["arguments"] == {}


# --- Regression tests: application targets ---


def test_open_calc_target():
    """Regression: 'open calc' -> target = 'calc'."""
    result = parse_command(["open", "calc"], Intent.OPEN_APPLICATION)
    assert result["intent"] == "open_application"
    assert result["target"] == "calc"


def test_open_notepad_target():
    """Regression: 'open notepad' -> target = 'notepad'."""
    result = parse_command(["open", "notepad"], Intent.OPEN_APPLICATION)
    assert result["intent"] == "open_application"
    assert result["target"] == "notepad"


def test_open_explorer_target():
    """Regression: 'open explorer' -> target = 'explorer'."""
    result = parse_command(["open", "explorer"], Intent.OPEN_APPLICATION)
    assert result["intent"] == "open_application"
    assert result["target"] == "explorer"


def test_open_file_explorer_target():
    """Regression: 'open file explorer' -> target = 'file explorer'."""
    result = parse_command(["open", "file", "explorer"], Intent.OPEN_APPLICATION)
    assert result["intent"] == "open_application"
    assert result["target"] == "file explorer"


def test_open_google_chrome_target():
    """Regression: 'open google chrome' -> target = 'google chrome'."""
    result = parse_command(["open", "google", "chrome"], Intent.OPEN_APPLICATION)
    assert result["intent"] == "open_application"
    assert result["target"] == "google chrome"


def test_open_microsoft_edge_target():
    """Test 'open microsoft edge' -> target = 'microsoft edge'."""
    result = parse_command(["open", "microsoft", "edge"], Intent.OPEN_APPLICATION)
    assert result["intent"] == "open_application"
    assert result["target"] == "microsoft edge"


# --- Command history parsing (Step 20) ---


def test_history():
    """Test parsing the history command (targetless)."""
    result = parse_command(["history"], Intent.HISTORY)
    assert result["intent"] == "history"
    assert result["target"] is None
    assert result["arguments"] == {}


def test_show_history():
    """Test parsing 'show history' (targetless)."""
    result = parse_command(["show", "history"], Intent.HISTORY)
    assert result["intent"] == "history"
    assert result["target"] is None
    assert result["arguments"] == {}


def test_recent_commands():
    """Test parsing 'recent commands' (targetless)."""
    result = parse_command(["recent", "commands"], Intent.HISTORY)
    assert result["intent"] == "history"
    assert result["target"] is None
    assert result["arguments"] == {}


def test_clear_history():
    """Test parsing 'clear history' (targetless)."""
    result = parse_command(["clear", "history"], Intent.CLEAR_HISTORY)
    assert result["intent"] == "clear_history"
    assert result["target"] is None
    assert result["arguments"] == {}


def test_delete_history():
    """Test parsing 'delete history' (targetless)."""
    result = parse_command(["delete", "history"], Intent.CLEAR_HISTORY)
    assert result["intent"] == "clear_history"
    assert result["target"] is None
    assert result["arguments"] == {}


if __name__ == "__main__":
    # Run all tests
    test_open_application()
    test_multi_word_application()
    test_close_application()
    test_create_folder()
    test_delete_file()
    test_open_file()
    test_search_file()
    test_lock_computer()
    test_lock_my_computer()
    test_shutdown_computer()
    test_shut_down_computer()
    test_restart_computer()
    test_open_browser()
    test_open_default_browser()
    test_open_url()
    test_open_url_with_path()
    test_search_web()
    test_search_the_web_for()
    test_google_search()
    test_missing_target()
    test_unknown_intent()
    test_empty_tokens()
    test_open_calc_target()
    test_open_notepad_target()
    test_open_explorer_target()
    test_open_file_explorer_target()
    test_open_google_chrome_target()
    test_open_microsoft_edge_target()
    test_history()
    test_show_history()
    test_recent_commands()
    test_clear_history()
    test_delete_history()
    print("All tests passed!")
