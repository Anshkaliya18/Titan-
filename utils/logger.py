"""
Centralized Logging Module

Single source of logging configuration for TITAN.

Design:
    configure_logging()     attach handlers (file always, console in debug mode)
    get_logger(name)        return a module logger, configuring logging once
    reset_logging()         remove handlers we attached (used by tests)

Destination:
    data root/TITAN.log   rotating file, always written (DEBUG and up)
    console                only when general.debug is enabled in configuration

    The data root comes from utils.paths (Step 23): <project>/logs in
    source mode, %APPDATA%\\TITAN\\logs when packaged. It is anchored to
    the installation, never to the current working directory, so a
    launch from a shortcut, another directory, the Startup folder, or
    Task Scheduler writes to the same stable, writable location.

Rules:
- The rotating file handler is attached exactly once — repeated calls to
  configure_logging() never create duplicate handlers.
- The debug setting is read through the Configuration Manager (lazily, to
  avoid a circular import), never by parsing settings.json directly.
- Logging must never crash the application: a failure to create the log
  file is logged and configuration is skipped.
- Nothing is ever written to the console unless debug mode is enabled, so
  the normal user CLI stays clean.
"""

import logging
import logging.handlers
from pathlib import Path

from utils import paths

# Default log location: writable data root + logs/ (Step 23 — stable
# regardless of the current working directory; in source mode this is
# <project>/logs, when packaged it is %APPDATA%\TITAN\logs).
DEFAULT_LOG_DIR = paths.log_dir()
LOG_FILENAME = "TITAN.log"

# Rotation: TITAN.log plus up to three rotated backups, 5 MB each
MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 3

# File format: 2026-10-10 15:30:21 | INFO | core.executor | message
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Console format (debug mode only): "DEBUG: message"
CONSOLE_FORMAT = "%(levelname)s: %(message)s"

# Handlers attached by this module (tracked to prevent duplicates)
_file_handler: logging.handlers.RotatingFileHandler | None = None
_console_handler: logging.StreamHandler | None = None


def _debug_enabled() -> bool:
    """
    Return whether debug mode is enabled in the configuration.

    Reads general.debug through the Configuration Manager. The import is
    lazy because the Configuration Manager itself uses this module, and
    while it is still initialising get_value is not yet available — in
    that case debug mode is treated as disabled and the caller may
    re-apply configuration later.
    """
    try:
        from core.config_manager import get_value
    except ImportError:
        # Configuration Manager not importable yet (circular import
        # during its own initialisation) — treat debug as disabled.
        return False
    value = get_value("general", "debug", False)
    return value if isinstance(value, bool) else False


# Lazily resolved secret redactor (see _redact_message).
_redactor = None


def _redact_message(message: str) -> str:
    """
    Redact obvious secrets from a log message.

    Uses the same deterministic keyword-based rules as Command History
    (password=..., token: ..., key is ...). The import is lazy to avoid
    a circular import — Command History uses this module's get_logger —
    and a failure keeps the original message, retried on the next
    record, so logging can never crash the application.
    """
    global _redactor
    if _redactor is None:
        try:
            from core.command_history import sanitize_history_text
        except Exception:
            # Not importable yet — try again on the next record.
            return message
        _redactor = sanitize_history_text
    try:
        return _redactor(message)
    except Exception:
        return message


class _RedactionFilter(logging.Filter):
    """
    Redact secret-looking content from every record its handler writes.

    Attached to the handlers this module creates, so no call site has
    to remember to sanitize: passwords, tokens, API keys, cookies, and
    similar values never reach the log file even when a message echoes
    them back. Filtering failures never propagate — a broken filter
    must not stop logging.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
            safe = _redact_message(message)
            if safe != message:
                record.msg = safe
                record.args = ()
        except Exception:
            pass
        return True


class _SizeRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """
    RotatingFileHandler with a cheap per-record rollover fast path.

    CPython's ``shouldRollover()`` runs for every single record and:
      - calls ``os.path.exists()`` + ``os.path.isfile()`` on the log
        path (the bpo-45401 non-regular-file guard — two stat
        syscalls),
      - formats the complete record including the timestamp, and
      - seeks the stream to measure the current size.

    Step 22 profiling showed this was the largest single per-command
    cost in the CLI (~40% of command latency). Almost every record
    lands far below the rotation limit, so this subclass first
    projects the record's size with an upper bound on the formatting
    overhead and returns ``False`` immediately when rotation is
    mathematically impossible. Whenever the file is near the limit
    (or the size cannot be determined), it delegates to the stock
    implementation — so the regularity guard and the exact rotation
    decision are identical to CPython for every record that could
    actually rotate. Levels, record content, format and the rotation
    threshold are unchanged.
    """

    # Upper bound on the formatted-record overhead (timestamp, level,
    # logger name, separators, newline) beyond the raw message text.
    # LOG_FORMAT's fixed part is well under 128 bytes; 256 leaves a
    # generous safety margin.
    _FORMAT_OVERHEAD = 256

    def shouldRollover(self, record) -> bool:
        """Cheap size pre-check; defer to CPython when near the limit."""
        if self.stream is None:
            self.stream = self._open()
        if self.maxBytes <= 0:
            return False
        try:
            # Same non-posix seek CPython performs before measuring.
            self.stream.seek(0, 2)
            projected = (
                self.stream.tell()
                + len(record.getMessage())
                + self._FORMAT_OVERHEAD
            )
        except (OSError, ValueError):
            # Undeterminable size — use the exact stock check.
            return super().shouldRollover(record)
        if projected < self.maxBytes:
            # Even with maximum formatting overhead the formatted
            # record cannot reach maxBytes: no rotation possible.
            return False
        # Near/over the limit: run the authoritative stock check
        # (regularity guard + exact formatted length).
        return super().shouldRollover(record)


def configure_logging(
    log_dir: Path | None = None,
    max_bytes: int | None = None,
    backup_count: int | None = None,
) -> None:
    """
    Configure logging destinations. Safe to call repeatedly.

    - Attaches one RotatingFileHandler writing to <log_dir>/TITAN.log
      (or log_dir when given, which is used by tests; the default comes
      from utils.paths — see the module docstring).
    - Attaches a console StreamHandler only when debug mode is enabled,
      and removes it again if debug mode is off. The debug setting may
      become readable later (configuration loads after the first call),
      so it is re-checked on every call.
    - Sets the root level to DEBUG so diagnostics always reach the
      file; the console handler filters what the user sees.

    Args:
        log_dir: Directory for the log file. Defaults to logs/.
        max_bytes: Rotation size limit. Defaults to 5 MB.
        backup_count: Number of rotated backups. Defaults to 3.
    """
    global _file_handler, _console_handler

    root = logging.getLogger()

    if _file_handler is None:
        target_dir = Path(log_dir) if log_dir is not None else DEFAULT_LOG_DIR
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            handler = _SizeRotatingFileHandler(
                target_dir / LOG_FILENAME,
                maxBytes=max_bytes if max_bytes is not None else MAX_BYTES,
                backupCount=(
                    backup_count if backup_count is not None else BACKUP_COUNT
                ),
                encoding="utf-8",
            )
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(
                logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
            )
            handler.addFilter(_RedactionFilter())
        except OSError:
            # Logging must never crash the application. Report through
            # Python's fallback stream and continue without a file.
            logging.getLogger(__name__).error(
                "Could not create the log file in %s", target_dir,
                exc_info=True,
            )
            return
        root.addHandler(handler)
        _file_handler = handler

    root.setLevel(logging.DEBUG)

    # Console handler follows the debug configuration. Debug may only
    # become readable after the Configuration Manager has loaded, so
    # this is evaluated on every call (the handler-tracking guards
    # prevent duplicates).
    debug = _debug_enabled()
    if debug and _console_handler is None:
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        console.setFormatter(logging.Formatter(CONSOLE_FORMAT))
        console.addFilter(_RedactionFilter())
        root.addHandler(console)
        _console_handler = console
    elif not debug and _console_handler is not None:
        root.removeHandler(_console_handler)
        _console_handler.close()
        _console_handler = None


def get_logger(name: str) -> logging.Logger:
    """
    Return a module logger, ensuring logging is configured once.

    Args:
        name: The logger name, normally the module's __name__
              (e.g. "core.executor").

    Returns:
        logging.Logger: The named logger.
    """
    configure_logging()
    return logging.getLogger(name)


def setup_logger(name: str = "TITAN", level: int = logging.INFO) -> logging.Logger:
    """
    Backward-compatible alias for get_logger().

    Kept so older call sites keep working; new code should use
    get_logger(__name__).

    Args:
        name: The logger name.
        level: Level applied to this specific logger.

    Returns:
        logging.Logger: The named logger.
    """
    module_logger = get_logger(name)
    module_logger.setLevel(level)
    return module_logger


def reset_logging() -> None:
    """
    Remove the handlers attached by this module and close them.

    Intended for tests, which reconfigure logging against temporary
    directories. Handlers added by other components (for example
    pytest's capture handlers) are left untouched.
    """
    global _file_handler, _console_handler

    root = logging.getLogger()
    for handler in (_file_handler, _console_handler):
        if handler is not None:
            root.removeHandler(handler)
            handler.close()
    _file_handler = None
    _console_handler = None


def is_configured() -> bool:
    """Return True when this module's handlers are attached."""
    return _file_handler is not None


def get_log_file() -> Path:
    """
    Return the path of the active log file.

    Falls back to the default location when logging has not been
    configured yet.
    """
    if _file_handler is not None:
        return Path(_file_handler.baseFilename)
    return DEFAULT_LOG_DIR / LOG_FILENAME
