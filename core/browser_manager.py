"""
Browser Manager Module

Browser operations: open the default browser, open URLs, web search.
Receives structured calls — never natural-language commands:

    open_browser()
    open_url("https://example.com")
    search_web("python tutorials")
    open_url_in_browser("chrome", "https://example.com")

Safe execution rules:
- Only http/https URLs are accepted (validated with urllib.parse).
- Search queries are URL-encoded with urllib.parse.quote_plus().
- URL opening uses the standard-library webbrowser module, which
  hands the URL to the Windows default handler — no shell involved.
- The default browser is resolved from the Windows registry, never
  hardcoded to a specific browser.
- Browser names/aliases and executable paths are NOT duplicated
  here: specific browsers are resolved through the Application
  Manager registry (single source of truth).
"""

import re
import subprocess
import webbrowser
from typing import Any
from urllib.parse import quote_plus, urlparse

from core.application_manager import get_application, resolve_application
from core.command_history import sanitize_history_text
from core.config_manager import get_value
from utils.logger import get_logger

logger = get_logger(__name__)

# The Windows registry module is Windows-only. Import it defensively so
# the module can still be imported (and tested) on other platforms.
try:
    import winreg
except ImportError:  # pragma: no cover - non-Windows only
    winreg = None  # type: ignore[assignment]


# Search engine fallback — used only if the configuration value is
# missing or invalid. The Configuration Manager is the source of truth.
DEFAULT_SEARCH_ENGINE = "https://www.google.com/search?q={query}"


def _get_search_engine() -> str:
    """
    Return the configured default search engine.

    Reads from the Configuration Manager. Falls back to the built-in
    default when the configured value is missing or not a string, so
    a malformed configuration can never introduce an unsafe engine.
    """
    engine = get_value("browser", "default_search_engine")
    if not isinstance(engine, str) or "{query}" not in engine:
        logger.warning("Invalid search engine config, using default")
        return DEFAULT_SEARCH_ENGINE
    return engine


def _safe_url_for_log(url: str) -> str:
    """
    Return a log-safe rendering of a URL.

    Query strings and fragments may carry credentials, session tokens,
    or other secrets, so they are stripped before a URL is written to
    the log file.
    """
    try:
        parts = urlparse(url)
    except ValueError:
        return "[unparseable URL]"
    # Embedded credentials (user:pass@host) must never be logged.
    netloc = parts.netloc
    if "@" in netloc:
        netloc = netloc.rsplit("@", 1)[1]
    safe = f"{parts.scheme}://{netloc}{parts.path}"
    if parts.query or parts.fragment:
        safe += " [query omitted]"
    return safe


# Only these URL schemes may ever be opened.
ALLOWED_URL_SCHEMES = {"http", "https"}

# Registry location of the per-user default handler for https:// URLs
_USER_CHOICE_KEY = (
    r"Software\Microsoft\Windows\Shell\Associations"
    r"\UrlAssociations\https\UserChoice"
)


def is_valid_url(url: str) -> bool:
    """
    Check whether a string is a safe, openable URL.

    Uses Python's standard URL parser — no hand-rolled RFC parsing.
    Only http and https are allowed, and a network location (host)
    is required. Dangerous schemes such as file:, javascript:,
    data:, vbscript:, cmd:, and powershell: are all rejected.

    Args:
        url: The candidate URL.

    Returns:
        True when the URL is safe to open.
    """
    if not url or not url.strip():
        return False

    parsed = urlparse(url.strip())

    if parsed.scheme.lower() not in ALLOWED_URL_SCHEMES:
        return False
    if not parsed.netloc:
        return False
    return True


def open_browser() -> dict[str, Any]:
    """
    Open the user's default web browser.

    Asks Windows which program handles https:// URLs (the per-user
    UserChoice association) and launches that program — the browser
    then opens to its own start page. No browser is hardcoded, so
    this works whether the default is Edge, Chrome, Firefox, etc.

    Returns:
        dict: {success, message, data}
    """
    logger.info("Browser operation requested: open_browser")

    executable = _resolve_default_browser()
    if executable is None:
        logger.warning("Default browser could not be resolved")
        return _result(False, "Unable to determine the default browser.", {})

    try:
        # Launch only the resolved executable path — never raw user input
        subprocess.Popen([executable], shell=False)
        logger.info(f"Default browser opened: {executable}")
        return _result(True, "Browser opened successfully.", {"browser": executable})
    except Exception as e:
        logger.error(f"Failed to open browser: {e}")
        return _result(False, "Unable to open the browser.", {})


def open_url(url: str) -> dict[str, Any]:
    """
    Open a URL in the user's default browser.

    The URL must be a validated http/https URL. It is handed to the
    Windows default handler via the standard webbrowser module —
    URLs are never passed to a shell.

    Args:
        url: A complete URL (e.g. "https://github.com").

    Returns:
        dict: {success, message, data}
    """
    logger.info("Browser operation requested: open_url")

    if not url or not url.strip():
        return _result(False, "No URL provided.", {})

    url = url.strip()

    if not is_valid_url(url):
        logger.warning(
            f"Invalid URL rejected: {_safe_url_for_log(url)}"
        )
        return _result(False, "Invalid URL.", {})

    try:
        opened = webbrowser.open(url)
    except Exception:
        logger.exception(
            f"Failed to open URL: {_safe_url_for_log(url)}"
        )
        return _result(False, "Failed to open the URL.", {})

    if not opened:
        logger.warning(
            f"Browser did not open the URL: {_safe_url_for_log(url)}"
        )
        return _result(False, "Failed to open the URL.", {})

    logger.info(f"Opening URL: {_safe_url_for_log(url)}")
    return _result(True, "Opened URL successfully.", {"url": url})


def search_web(query: str) -> dict[str, Any]:
    """
    Open a web search for the given query.

    The query is URL-encoded (spaces become '+', special characters
    are percent-encoded) and inserted into the configured search
    engine URL. The search then opens in the default browser.

    Args:
        query: The search terms (e.g. "python tutorials").

    Returns:
        dict: {success, message, data}
    """
    logger.info("Browser operation requested: search_web")

    if not query or not query.strip():
        return _result(False, "No search query provided.", {})

    query = query.strip()

    # The search engine comes from configuration; the resulting URL is
    # still validated below so a malformed setting cannot introduce
    # an unsafe scheme.
    encoded = quote_plus(query)
    url = _get_search_engine().replace("{query}", encoded)

    if not is_valid_url(url):
        logger.warning(
            f"Search URL failed validation: {_safe_url_for_log(url)}"
        )
        return _result(False, "Invalid search URL.", {})

    try:
        opened = webbrowser.open(url)
    except Exception:
        logger.exception("Failed to open web search")
        return _result(False, "Failed to open the web search.", {})

    if not opened:
        logger.warning("Browser did not open the web search")
        return _result(False, "Failed to open the web search.", {})

    logger.info(f"Web search requested: {sanitize_history_text(query)}")
    return _result(
        True,
        "Opened web search.",
        {"query": query, "url": url},
    )


def open_url_in_browser(browser: str, url: str) -> dict[str, Any]:
    """
    Open a URL in a specific browser.

    The browser is looked up in the Application Manager registry —
    the single source of truth for browser names, aliases, and
    executable paths. So "chrome" and "google chrome" always refer
    to the same application, and no browser paths are duplicated.

    The URL must pass the same http/https validation, and the
    resolved executable and URL are passed as separate process
    arguments — no shell is involved.

    Args:
        browser: Application alias (e.g. "chrome", "google chrome").
        url: A complete URL (e.g. "https://github.com").

    Returns:
        dict: {success, message, data}
    """
    logger.info("Browser operation requested: open_url_in_browser")

    if not url or not url.strip():
        return _result(False, "No URL provided.", {})

    url = url.strip()

    if not is_valid_url(url):
        logger.warning(
            f"Invalid URL rejected: {_safe_url_for_log(url)}"
        )
        return _result(False, "Invalid URL.", {})

    # One source of truth: browser aliases live in the Application Manager
    app = get_application(browser)
    if app is None:
        logger.warning(f"Browser not found: {browser}")
        return _result(
            False,
            f"Browser '{browser}' is not allowed or configured.",
            {},
        )
    if not app.is_browser:
        return _result(False, f"'{browser}' is not a browser.", {})

    executable = resolve_application(browser)
    if executable is None:
        logger.warning(f"Executable not found for {browser}")
        return _result(
            False,
            f"{app.name.title()} is not installed or could not be located.",
            {},
        )

    try:
        # Resolved executable path + validated URL as separate
        # arguments — never a constructed command string
        subprocess.Popen([executable, url], shell=False)
        logger.info(f"Opening URL in {app.name}: {_safe_url_for_log(url)}")
        return _result(
            True,
            f"Opened URL in {app.name.title()}.",
            {"browser": app.name, "url": url},
        )
    except Exception:
        logger.exception(f"Failed to open URL in {app.name}")
        return _result(False, f"Failed to open URL in {app.name.title()}.", {})


def _resolve_default_browser() -> str | None:
    """
    Resolve the default https:// handler to an executable path.

    Asks Windows (per-user association + file-association command)
    instead of guessing which browser is installed. Returns None
    when the association cannot be read.
    """
    if winreg is None:
        return None

    try:
        # Which program does this user associate with https:// URLs?
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _USER_CHOICE_KEY) as key:
            prog_id, _ = winreg.QueryValueEx(key, "ProgId")

        # The matching command lives in the merged classes view
        # (HKCU classes + HKLM classes). The default value of a key
        # has the name "" (empty string) when read via winreg.
        with winreg.OpenKey(
            winreg.HKEY_CLASSES_ROOT, rf"{prog_id}\shell\open\command"
        ) as key:
            command, _ = winreg.QueryValueEx(key, "")
    except OSError:
        return None

    if not command:
        return None

    # The command looks like:  "C:\...\msedge.exe" --single-argument %1
    match = re.match(r'"([^"]+)"', command)
    if match:
        return match.group(1)
    return command.split(" ")[0]


def _result(success: bool, message: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Build the standard result structure used by every manager.

    Args:
        success: Whether the operation succeeded.
        message: Human-readable explanation.
        data: Optional extra information (defaults to an empty dict).

    Returns:
        dict: {success, message, data}
    """
    return {
        "success": success,
        "message": message,
        "data": data if data is not None else {},
    }
