"""
Tests for the centralized logging system and error boundaries.

Every test runs against a temporary log directory; the real project
logs/TITAN.log is never written by this suite.
"""

import io
import logging
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

# Add parent directory to path so we can import core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import utils.logger as logger_module
from utils.logger import (
    BACKUP_COUNT,
    DEFAULT_LOG_DIR,
    LOG_FILENAME,
    MAX_BYTES,
    configure_logging,
    get_log_file,
    get_logger,
    is_configured,
    reset_logging,
    setup_logger,
)


class _LoggingTestCase(unittest.TestCase):
    """
    Base class isolating each test in a temporary log directory.

    The configuration active before the test (for example the
    session-wide temporary directory from conftest.py) is restored
    afterwards, so neither the real project log nor other tests are
    affected.
    """

    def setUp(self):
        self._restore_dir = (
            get_log_file().parent if is_configured() else None
        )
        reset_logging()
        self.log_dir = Path(tempfile.mkdtemp(prefix="TITAN-logger-test-"))
        configure_logging(log_dir=self.log_dir)

    def tearDown(self):
        reset_logging()
        if self._restore_dir is not None:
            configure_logging(log_dir=self._restore_dir)
        shutil.rmtree(self.log_dir, ignore_errors=True)

    def read_log(self) -> str:
        """Return the current contents of the temporary log file."""
        return (self.log_dir / LOG_FILENAME).read_text(encoding="utf-8")

    @staticmethod
    def _stream_handler_count() -> int:
        """Count StreamHandlers on the root logger (file included)."""
        return sum(
            isinstance(handler, logging.StreamHandler)
            for handler in logging.getLogger().handlers
        )


class TestGetLogger(_LoggingTestCase):
    """get_logger / setup_logger behaviour."""

    def test_logger_has_expected_name(self):
        logger = get_logger("core.executor")
        self.assertEqual(logger.name, "core.executor")

    def test_returns_logging_logger(self):
        self.assertIsInstance(get_logger("test.module"), logging.Logger)

    def test_setup_logger_alias_works(self):
        legacy = setup_logger("legacy_module")
        self.assertEqual(legacy.name, "legacy_module")


class TestLogFile(_LoggingTestCase):
    """Log file creation and content."""

    def test_log_file_created_in_given_dir(self):
        self.assertTrue((self.log_dir / LOG_FILENAME).exists())

    def test_get_log_file_points_at_active_file(self):
        self.assertEqual(get_log_file(), self.log_dir / LOG_FILENAME)

    def test_default_log_location(self):
        self.assertEqual(DEFAULT_LOG_DIR.name, "logs")
        self.assertEqual(DEFAULT_LOG_DIR / LOG_FILENAME,
                         DEFAULT_LOG_DIR / "TITAN.log")

    def test_info_message_is_logged(self):
        test_logger = get_logger("test.info")
        test_logger.info("hello info message")
        self.assertIn("hello info message", self.read_log())

    def test_log_line_format(self):
        get_logger("test.format").info("formatted message")
        pattern = (
            r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| INFO \| "
            r"test\.format \| formatted message$"
        )
        lines = [l for l in self.read_log().splitlines()
                 if "formatted message" in l]
        self.assertEqual(len(lines), 1)
        self.assertRegex(lines[0], pattern)

    def test_debug_messages_always_reach_the_file(self):
        get_logger("test.debug").debug("diagnostic detail")
        self.assertIn("diagnostic detail", self.read_log())

    def test_exception_writes_traceback(self):
        test_logger = get_logger("test.exception")
        try:
            raise ValueError("boom value")
        except ValueError:
            test_logger.exception("operation failed")
        content = self.read_log()
        self.assertIn("operation failed", content)
        self.assertIn("Traceback (most recent call last)", content)
        self.assertIn("ValueError", content)


class TestNoDuplicateHandlers(_LoggingTestCase):
    """Repeated configuration never duplicates handlers or messages."""

    def test_repeated_configuration_does_not_duplicate_handlers(self):
        before = len(logging.getLogger().handlers)
        configure_logging()
        configure_logging()
        configure_logging()
        self.assertEqual(len(logging.getLogger().handlers), before)

    def test_message_appears_once_in_log(self):
        get_logger("test.single").info("unique marker message")
        content = self.read_log()
        self.assertEqual(content.count("unique marker message"), 1)


class TestDebugConsole(_LoggingTestCase):
    """Console output follows the general.debug configuration."""

    def _configure_with_debug(self, debug: bool) -> int:
        """
        (Re)configure logging with the given debug flag.

        Returns the change in StreamHandler count, which is robust
        against handlers added by other components (e.g. pytest).
        """
        reset_logging()
        baseline = self._stream_handler_count()
        with patch("core.config_manager.get_value", return_value=debug):
            configure_logging(log_dir=self.log_dir)
        return self._stream_handler_count() - baseline

    def test_console_absent_when_debug_disabled(self):
        # Only the rotating file handler is added
        self.assertEqual(self._configure_with_debug(False), 1)

    def test_console_present_when_debug_enabled(self):
        # File handler plus console handler
        self.assertEqual(self._configure_with_debug(True), 2)

    def test_console_removed_when_debug_disabled_again(self):
        self.assertEqual(self._configure_with_debug(True), 2)
        self.assertEqual(self._configure_with_debug(False), 1)


class TestRotation(_LoggingTestCase):
    """The log file rotates and never grows forever."""

    def test_rotating_handler_is_configured(self):
        handler = logger_module._file_handler
        self.assertIsInstance(handler, logging.handlers.RotatingFileHandler)
        self.assertEqual(handler.maxBytes, MAX_BYTES)
        self.assertEqual(handler.backupCount, BACKUP_COUNT)

    def test_max_bytes_default_is_five_mb(self):
        self.assertEqual(MAX_BYTES, 5 * 1024 * 1024)

    def test_backup_count_default_is_three(self):
        self.assertEqual(BACKUP_COUNT, 3)

    def test_rotation_creates_backup_files(self):
        reset_logging()
        small_dir = Path(tempfile.mkdtemp(prefix="TITAN-rotation-test-"))
        try:
            configure_logging(log_dir=small_dir, max_bytes=400,
                              backup_count=2)
            noisy = get_logger("test.rotation")
            for i in range(20):
                noisy.info("rotation filler message %d %s", i, "x" * 80)
            self.assertTrue((small_dir / LOG_FILENAME).exists())
            self.assertTrue((small_dir / f"{LOG_FILENAME}.1").exists())
        finally:
            reset_logging()
            configure_logging(log_dir=self.log_dir)
            shutil.rmtree(small_dir, ignore_errors=True)


class TestExecutorErrorBoundary(_LoggingTestCase):
    """A failing manager never crashes the pipeline or leaks details."""

    def _execute_open_chrome(self):
        from core.executor import execute_command

        return execute_command({
            "intent": "OPEN_APPLICATION",
            "target": "chrome",
            "arguments": {},
        })

    def test_unexpected_exception_returns_safe_result(self):
        with patch(
            "core.executor.launch_application",
            side_effect=RuntimeError("internal detail xyz"),
        ):
            result = self._execute_open_chrome()
        self.assertFalse(result["success"])
        self.assertEqual(
            result["message"],
            "Something went wrong while processing that command.",
        )

    def test_user_message_never_contains_traceback_or_exception(self):
        with patch(
            "core.executor.launch_application",
            side_effect=RuntimeError("internal detail xyz"),
        ):
            result = self._execute_open_chrome()
        message = result["message"]
        self.assertNotIn("Traceback", message)
        self.assertNotIn("RuntimeError", message)
        self.assertNotIn("internal detail xyz", message)

    def test_traceback_is_written_to_the_log(self):
        with patch(
            "core.executor.launch_application",
            side_effect=RuntimeError("internal detail xyz"),
        ):
            self._execute_open_chrome()
        content = self.read_log()
        self.assertIn("Traceback (most recent call last)", content)
        self.assertIn("RuntimeError", content)

    def test_TITAN_error_returns_safe_result(self):
        from core.exceptions import ApplicationError

        with patch(
            "core.executor.launch_application",
            side_effect=ApplicationError("low level manager detail"),
        ):
            result = self._execute_open_chrome()
        self.assertFalse(result["success"])
        self.assertEqual(result["message"], "I couldn't complete that command.")
        # The technical detail belongs in the log, not with the user
        self.assertIn("low level manager detail", self.read_log())
        self.assertNotIn("low level manager detail", result["message"])

    def test_pipeline_continues_after_failure(self):
        from core.executor import execute_command

        with patch(
            "core.executor.launch_application",
            side_effect=RuntimeError("boom"),
        ):
            failed = self._execute_open_chrome()
        self.assertFalse(failed["success"])

        # The very next command still works normally
        healthy = execute_command({
            "intent": "HELP",
            "target": None,
            "arguments": {},
        })
        self.assertTrue(healthy["success"])


class TestMainErrorBoundary(_LoggingTestCase):
    """main.py survives unexpected exceptions and interrupts."""

    def test_unexpected_error_is_logged_and_loop_continues(self):
        import main

        buffer = io.StringIO()
        with patch(
            "main._process_once",
            side_effect=[RuntimeError("loop failure"), KeyboardInterrupt()],
        ):
            with redirect_stdout(buffer):
                main.main()

        output = buffer.getvalue()
        self.assertIn("Something went wrong. Please try again.", output)
        self.assertIn("Goodbye!", output)
        content = self.read_log()
        self.assertIn("Traceback (most recent call last)", content)
        self.assertIn("RuntimeError", content)


class TestSensitiveDataStaysOutOfLogs(_LoggingTestCase):
    """Raw user input and URL secrets never reach the log file."""

    def test_raw_input_content_is_not_logged(self):
        from core.input import get_input

        with patch("builtins.input", return_value="my password123"):
            get_input()
        self.assertNotIn("password123", self.read_log())
        # Only the length is recorded
        self.assertIn("characters", self.read_log())

    def test_url_query_strings_are_redacted(self):
        from core.browser_manager import _safe_url_for_log

        safe = _safe_url_for_log(
            "https://example.com/page?token=secret123#fragment"
        )
        self.assertNotIn("secret123", safe)
        self.assertNotIn("fragment", safe)
        self.assertIn("example.com/page", safe)
        self.assertIn("[query omitted]", safe)

    def test_plain_url_is_kept_readable(self):
        from core.browser_manager import _safe_url_for_log

        self.assertEqual(
            _safe_url_for_log("https://github.com"), "https://github.com"
        )

    def test_url_userinfo_is_never_logged(self):
        """Embedded credentials (user:pass@host) never reach the log."""
        from core.browser_manager import _safe_url_for_log

        safe = _safe_url_for_log("https://user:pass@example.com/account")
        self.assertNotIn("user:pass", safe)
        self.assertIn("example.com/account", safe)

    def test_secret_in_any_log_record_is_redacted(self):
        """The handler-level redaction filter catches secrets from any
        module before they reach the log file."""
        logging.getLogger("core.any_module").info(
            "Connection failed for password=supersecret123"
        )
        content = self.read_log()
        self.assertNotIn("supersecret123", content)
        self.assertIn("password=[REDACTED]", content)

    def test_tokenizer_logs_counts_not_contents(self):
        """Token contents (which may echo user text) are never logged."""
        from core.tokenizer import tokenize

        tokenize("search web my password is abc123")
        content = self.read_log()
        self.assertNotIn("abc123", content)
        self.assertIn("Tokenization completed", content)

    def test_command_parser_logs_intent_not_target(self):
        """Parsed targets may echo user text — only intent is logged."""
        from core.command_parser import parse_command
        from core.intent import detect_intent

        tokens = ["search", "web", "token", "is", "abc123"]
        parse_command(tokens, detect_intent(tokens))
        content = self.read_log()
        self.assertNotIn("abc123", content)

    def test_browser_search_query_is_sanitized_in_log(self):
        """A secret typed as a search query never reaches the log."""
        from core.browser_manager import search_web

        with patch(
            "core.browser_manager.webbrowser.open", return_value=True
        ):
            search_web("my password is hunter2secret")
        content = self.read_log()
        self.assertNotIn("hunter2secret", content)
        self.assertIn("[REDACTED]", content)

    def test_redaction_never_crashes_logging(self):
        """A redaction failure must never propagate to the caller."""
        from utils.logger import _redact_message

        self.assertEqual(
            _redact_message("plain message"), "plain message"
        )
        record = logging.LogRecord(
            "x", logging.INFO, __file__, 1, "password=%s", ("abc",), None
        )
        filt = logger_module._RedactionFilter()
        self.assertTrue(filt.filter(record))
        self.assertNotIn("abc", record.getMessage())


if __name__ == "__main__":
    unittest.main()
