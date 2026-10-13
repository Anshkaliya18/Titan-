"""
Tests for the intent detection module.
"""

import sys
import os

# Add parent directory to path so we can import core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.intent import detect_intent, Intent


def test_open_application():
    """Test open application intent."""
    assert detect_intent(["open", "chrome"]) == Intent.OPEN_APPLICATION


def test_close_application():
    """Test close application intent."""
    assert detect_intent(["close", "chrome"]) == Intent.CLOSE_APPLICATION


def test_lock_computer():
    """Test lock computer intent."""
    assert detect_intent(["lock", "computer"]) == Intent.LOCK_COMPUTER


def test_restart_computer():
    """Test restart computer intent."""
    assert detect_intent(["restart", "computer"]) == Intent.RESTART_COMPUTER


def test_shutdown_computer():
    """Test shutdown computer intent."""
    assert detect_intent(["shutdown", "computer"]) == Intent.SHUTDOWN_COMPUTER


def test_lock_my_computer():
    """Test that filler words don't block the lock intent."""
    assert detect_intent(["lock", "my", "computer"]) == Intent.LOCK_COMPUTER


def test_shut_down_computer():
    """Test the two-word 'shut down' variant."""
    assert detect_intent(["shut", "down", "computer"]) == Intent.SHUTDOWN_COMPUTER


def test_shut_down_my_computer():
    """Test 'shut down my computer' with a filler word."""
    assert detect_intent(["shut", "down", "my", "computer"]) == Intent.SHUTDOWN_COMPUTER


def test_reboot_computer():
    """Test that 'reboot' is recognized as restart."""
    assert detect_intent(["reboot", "computer"]) == Intent.RESTART_COMPUTER


def test_computer_alone_is_unknown():
    """'computer' alone must not be classified as any system command."""
    assert detect_intent(["computer"]) == Intent.UNKNOWN


def test_restart_alone_is_recognized():
    """'restart' is a registered alias and must be recognized.

    The bare action word maps to RESTART_COMPUTER; the dangerous
    command still parks as a confirmation downstream, so recognizing
    it here is safe.
    """
    assert detect_intent(["restart"]) == Intent.RESTART_COMPUTER


# --- Browser intents (Step 11) ---


def test_open_browser():
    """Test 'open browser' is OPEN_BROWSER."""
    assert detect_intent(["open", "browser"]) == Intent.OPEN_BROWSER


def test_open_default_browser():
    """Test 'open default browser' is OPEN_BROWSER."""
    assert detect_intent(["open", "default", "browser"]) == Intent.OPEN_BROWSER


def test_launch_browser():
    """Test 'launch browser' is OPEN_BROWSER."""
    assert detect_intent(["launch", "browser"]) == Intent.OPEN_BROWSER


def test_open_url():
    """Test 'open https://github.com' (tokenized) is OPEN_URL."""
    assert (
        detect_intent(["open", "https", ":", "//github.com"])
        == Intent.OPEN_URL
    )


def test_open_website_url():
    """Test 'open website https://github.com' is OPEN_URL."""
    assert (
        detect_intent(["open", "website", "https", ":", "//github.com"])
        == Intent.OPEN_URL
    )


def test_open_http_url():
    """Test 'http' URLs are recognized too."""
    assert (
        detect_intent(["open", "http", ":", "//example.com"])
        == Intent.OPEN_URL
    )


def test_open_url_not_open_file():
    """Regression: a URL path ending in .pdf must stay OPEN_URL."""
    assert (
        detect_intent(["open", "https", ":", "//example.com/report.pdf"])
        == Intent.OPEN_URL
    )


def test_search_web():
    """Test 'search web python tutorials' is SEARCH_WEB."""
    assert (
        detect_intent(["search", "web", "python", "tutorials"])
        == Intent.SEARCH_WEB
    )


def test_search_the_web_for():
    """Test 'search the web for python tutorials' is SEARCH_WEB."""
    assert (
        detect_intent(["search", "the", "web", "for", "python", "tutorials"])
        == Intent.SEARCH_WEB
    )


def test_bare_search_query_is_search():
    """Test 'search python tutorials' (without 'web') is SEARCH_WEB."""
    assert (
        detect_intent(["search", "python", "tutorials"])
        == Intent.SEARCH_WEB
    )


def test_lone_search_is_unknown():
    """A bare 'search' with no query is not a command."""
    assert detect_intent(["search"]) == Intent.UNKNOWN


def test_google_query_is_search():
    """Test 'google python tutorials' is SEARCH_WEB."""
    assert (
        detect_intent(["google", "python", "tutorials"]) == Intent.SEARCH_WEB
    )


# --- Browser regressions: context must be preserved ---


def test_open_google_chrome_stays_application():
    """Regression: 'open google chrome' must remain OPEN_APPLICATION."""
    assert (
        detect_intent(["open", "google", "chrome"])
        == Intent.OPEN_APPLICATION
    )


def test_open_chrome_stays_application():
    """Regression: 'open chrome' must remain OPEN_APPLICATION."""
    assert detect_intent(["open", "chrome"]) == Intent.OPEN_APPLICATION


def test_open_firefox_stays_application():
    """Regression: 'open firefox' must remain OPEN_APPLICATION."""
    assert detect_intent(["open", "firefox"]) == Intent.OPEN_APPLICATION


def test_google_chrome_alone_is_not_search():
    """'google chrome' (no action) must not become SEARCH_WEB."""
    assert detect_intent(["google", "chrome"]) == Intent.UNKNOWN


def test_bare_google_is_not_search():
    """A bare 'google' with no query is not a search."""
    assert detect_intent(["google"]) == Intent.UNKNOWN


def test_unknown():
    """Test unknown intent."""
    assert detect_intent(["hello", "jarvis"]) == Intent.UNKNOWN


def test_empty():
    """Test empty input returns UNKNOWN."""
    assert detect_intent([]) == Intent.UNKNOWN


def test_create_folder():
    """Test create folder intent."""
    assert detect_intent(["create", "folder"]) == Intent.CREATE_FOLDER


def test_delete_file():
    """Test delete file intent."""
    assert detect_intent(["delete", "file"]) == Intent.DELETE_FILE


def test_search_file():
    """Test search file intent."""
    assert detect_intent(["search", "file"]) == Intent.SEARCH_FILE


def test_open_browser():
    """Test open browser intent."""
    assert detect_intent(["open", "browser"]) == Intent.OPEN_BROWSER


def test_open_file():
    """Test open file intent."""
    assert detect_intent(["open", "file"]) == Intent.OPEN_FILE


def test_launch_application():
    """Test launch keyword for open application."""
    assert detect_intent(["launch", "firefox"]) == Intent.OPEN_APPLICATION


def test_make_folder():
    """Test make keyword for create folder."""
    assert detect_intent(["make", "folder"]) == Intent.CREATE_FOLDER


def test_remove_file():
    """Test remove keyword for delete file."""
    assert detect_intent(["remove", "file"]) == Intent.DELETE_FILE


def test_find_file():
    """Test find keyword for search file."""
    assert detect_intent(["find", "file"]) == Intent.SEARCH_FILE


# --- Regression tests: application priority over file commands ---


def test_open_calc_is_application():
    """Regression: 'open calc' must be OPEN_APPLICATION."""
    assert detect_intent(["open", "calc"]) == Intent.OPEN_APPLICATION


def test_open_notepad_is_application():
    """Regression: 'open notepad' must be OPEN_APPLICATION."""
    assert detect_intent(["open", "notepad"]) == Intent.OPEN_APPLICATION


def test_open_explorer_is_application():
    """Regression: 'open explorer' must be OPEN_APPLICATION."""
    assert detect_intent(["open", "explorer"]) == Intent.OPEN_APPLICATION


def test_open_file_explorer_is_application():
    """Regression: 'open file explorer' must be OPEN_APPLICATION, not OPEN_FILE."""
    assert detect_intent(["open", "file", "explorer"]) == Intent.OPEN_APPLICATION


def test_open_google_chrome_is_application():
    """Regression: 'open google chrome' must be OPEN_APPLICATION."""
    assert detect_intent(["open", "google", "chrome"]) == Intent.OPEN_APPLICATION


def test_open_report_pdf_is_file():
    """Regression: 'open report.pdf' must be OPEN_FILE."""
    assert detect_intent(["open", "report.pdf"]) == Intent.OPEN_FILE


def test_open_notes_txt_is_file():
    """Test that 'open notes.txt' is OPEN_FILE."""
    assert detect_intent(["open", "notes.txt"]) == Intent.OPEN_FILE


def test_open_microsoft_edge_is_application():
    """Test that 'open microsoft edge' is OPEN_APPLICATION."""
    assert detect_intent(["open", "microsoft", "edge"]) == Intent.OPEN_APPLICATION


def test_open_google_is_application():
    """Test that 'open google' is OPEN_APPLICATION."""
    assert detect_intent(["open", "google"]) == Intent.OPEN_APPLICATION


# --- Help / command-discovery intents (Step 15) ---


def test_help():
    """Test 'help' is HELP."""
    assert detect_intent(["help"]) == Intent.HELP


def test_what_can_you_do():
    """Test 'what can you do' is HELP."""
    assert detect_intent(["what", "can", "you", "do"]) == Intent.HELP


def test_how_can_you_help():
    """Test 'how can you help' is HELP."""
    assert detect_intent(["how", "can", "you", "help"]) == Intent.HELP


def test_show_help():
    """Test 'show help' is HELP, not LIST_COMMANDS."""
    assert detect_intent(["show", "help"]) == Intent.HELP


def test_commands():
    """Test 'commands' is LIST_COMMANDS."""
    assert detect_intent(["commands"]) == Intent.LIST_COMMANDS


def test_list_commands():
    """Test 'list commands' is LIST_COMMANDS."""
    assert detect_intent(["list", "commands"]) == Intent.LIST_COMMANDS


def test_show_commands():
    """Test 'show commands' is LIST_COMMANDS."""
    assert detect_intent(["show", "commands"]) == Intent.LIST_COMMANDS


def test_available_commands():
    """Test 'available commands' is LIST_COMMANDS."""
    assert detect_intent(["available", "commands"]) == Intent.LIST_COMMANDS


# --- Command history intents (Step 20) ---


def test_history():
    """Test 'history' is HISTORY."""
    assert detect_intent(["history"]) == Intent.HISTORY


def test_show_history():
    """Test 'show history' is HISTORY."""
    assert detect_intent(["show", "history"]) == Intent.HISTORY


def test_command_history():
    """Test 'command history' is HISTORY."""
    assert detect_intent(["command", "history"]) == Intent.HISTORY


def test_recent_commands():
    """Test 'recent commands' is HISTORY, not LIST_COMMANDS."""
    assert detect_intent(["recent", "commands"]) == Intent.HISTORY


def test_clear_history():
    """Test 'clear history' is CLEAR_HISTORY."""
    assert detect_intent(["clear", "history"]) == Intent.CLEAR_HISTORY


def test_delete_history():
    """Test 'delete history' is CLEAR_HISTORY."""
    assert detect_intent(["delete", "history"]) == Intent.CLEAR_HISTORY


def test_remove_history():
    """Test 'remove history' is CLEAR_HISTORY."""
    assert detect_intent(["remove", "history"]) == Intent.CLEAR_HISTORY


def test_search_history_is_not_history():
    """'search history' must not be confused with HISTORY — it stays
    a search query (no such history feature exists)."""
    assert detect_intent(["search", "history"]) == Intent.SEARCH_WEB


def test_search_web_history_is_search():
    """'search web history' remains a web search."""
    assert detect_intent(["search", "web", "history"]) == Intent.SEARCH_WEB


def test_history_txt_file_stays_delete_file():
    """'delete file history.txt' is a file command, not CLEAR_HISTORY."""
    assert (
        detect_intent(["delete", "file", "history.txt"])
        == Intent.DELETE_FILE
    )


def test_open_history_is_not_history():
    """'open history' falls through to OPEN_APPLICATION (rejected by
    the allowlist later) — it is not a history command."""
    assert detect_intent(["open", "history"]) == Intent.OPEN_APPLICATION


def test_show_commands_still_list_commands():
    """Regression: 'show commands' must not become HISTORY."""
    assert detect_intent(["show", "commands"]) == Intent.LIST_COMMANDS


def test_commands_still_list_commands():
    """Regression: 'commands' must not become HISTORY."""
    assert detect_intent(["commands"]) == Intent.LIST_COMMANDS


if __name__ == "__main__":
    # Run all tests
    test_open_application()
    test_close_application()
    test_lock_computer()
    test_restart_computer()
    test_shutdown_computer()
    test_lock_my_computer()
    test_shut_down_computer()
    test_shut_down_my_computer()
    test_reboot_computer()
    test_computer_alone_is_unknown()
    test_restart_alone_is_unknown()
    test_open_browser()
    test_open_default_browser()
    test_launch_browser()
    test_open_url()
    test_open_website_url()
    test_open_http_url()
    test_open_url_not_open_file()
    test_search_web()
    test_search_the_web_for()
    test_bare_search_query_is_search()
    test_lone_search_is_unknown()
    test_google_query_is_search()
    test_open_google_chrome_stays_application()
    test_open_chrome_stays_application()
    test_open_firefox_stays_application()
    test_google_chrome_alone_is_not_search()
    test_bare_google_is_not_search()
    test_unknown()
    test_empty()
    test_create_folder()
    test_delete_file()
    test_search_file()
    test_open_browser()
    test_open_file()
    test_launch_application()
    test_make_folder()
    test_remove_file()
    test_find_file()
    test_open_calc_is_application()
    test_open_notepad_is_application()
    test_open_explorer_is_application()
    test_open_file_explorer_is_application()
    test_open_google_chrome_is_application()
    test_open_report_pdf_is_file()
    test_open_notes_txt_is_file()
    test_open_microsoft_edge_is_application()
    test_open_google_is_application()
    test_help()
    test_what_can_you_do()
    test_how_can_you_help()
    test_show_help()
    test_commands()
    test_list_commands()
    test_show_commands()
    test_available_commands()
    test_history()
    test_show_history()
    test_command_history()
    test_recent_commands()
    test_clear_history()
    test_delete_history()
    test_remove_history()
    test_search_history_is_not_history()
    test_search_web_history_is_search()
    test_history_txt_file_stays_delete_file()
    test_open_history_is_not_history()
    test_show_commands_still_list_commands()
    test_commands_still_list_commands()
    print("All tests passed!")
