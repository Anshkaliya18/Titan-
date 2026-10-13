"""
Tests for the custom exception hierarchy.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Add parent directory to path so we can import core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.exceptions import (
    ApplicationError,
    BrowserError,
    CommandError,
    ConfigurationError,
    FileOperationError,
    JarvisError,
    SystemOperationError,
    ValidationError,
)

ALL_SPECIALIZED = [
    CommandError,
    ValidationError,
    ApplicationError,
    FileOperationError,
    BrowserError,
    SystemOperationError,
    ConfigurationError,
]


class TestJarvisError(unittest.TestCase):
    """The base exception."""

    def test_is_exception_subclass(self):
        self.assertTrue(issubclass(JarvisError, Exception))

    def test_raise_and_catch(self):
        with self.assertRaises(JarvisError):
            raise JarvisError("something failed")

    def test_message_is_preserved(self):
        try:
            raise JarvisError("something failed")
        except JarvisError as e:
            self.assertEqual(str(e), "something failed")

    def test_not_related_to_stdlib_errors(self):
        self.assertFalse(issubclass(ValueError, JarvisError))


class TestSpecializedExceptions(unittest.TestCase):
    """Each specialized exception integrates with the hierarchy."""

    def test_all_subclass_jarvis_error(self):
        for cls in ALL_SPECIALIZED:
            with self.subTest(cls=cls.__name__):
                self.assertTrue(issubclass(cls, JarvisError))

    def test_all_subclass_exception(self):
        for cls in ALL_SPECIALIZED:
            with self.subTest(cls=cls.__name__):
                self.assertTrue(issubclass(cls, Exception))

    def test_catchable_as_jarvis_error(self):
        for cls in ALL_SPECIALIZED:
            with self.subTest(cls=cls.__name__):
                with self.assertRaises(JarvisError):
                    raise cls("detail")

    def test_catchable_as_own_type(self):
        with self.assertRaises(ApplicationError):
            raise ApplicationError("app detail")

    def test_inheritance_chain(self):
        self.assertTrue(issubclass(ApplicationError, JarvisError))
        self.assertTrue(issubclass(JarvisError, Exception))
        self.assertIs(ApplicationError.__mro__[1], JarvisError)


class TestConfigurationErrorUsage(unittest.TestCase):
    """ConfigurationError is raised when settings cannot be persisted."""

    def _impossible_settings_path(self) -> Path:
        """Return a settings path whose parent cannot be a directory."""
        handle = tempfile.NamedTemporaryFile(delete=False)
        handle.close()
        self.addCleanup(os.remove, handle.name)
        # The parent is an existing FILE, so mkdir must fail
        return Path(handle.name) / "settings.json"

    def test_save_failure_raises_configuration_error(self):
        import core.config_manager as config_manager

        with patch.object(
            config_manager,
            "_settings_path",
            return_value=self._impossible_settings_path(),
        ):
            with self.assertRaises(ConfigurationError):
                config_manager.set_value("general", "assistant_name", "X")
        # Restore the real configuration in memory
        config_manager.reload_config()

    def test_save_failure_leaves_real_settings_untouched(self):
        import core.config_manager as config_manager

        real_path = config_manager._settings_path()
        before = real_path.read_text(encoding="utf-8")
        with patch.object(
            config_manager,
            "_settings_path",
            return_value=self._impossible_settings_path(),
        ):
            with self.assertRaises(ConfigurationError):
                config_manager.set_value("general", "debug", True)
        config_manager.reload_config()
        self.assertEqual(
            real_path.read_text(encoding="utf-8"), before
        )


if __name__ == "__main__":
    unittest.main()
