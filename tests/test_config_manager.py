"""
Tests for the configuration manager.

All tests use temporary files / mocked paths so the real
config/settings.json is never modified.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Add parent directory to path so we can import core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import core.config_manager as config_manager
from core.config_manager import (
    DEFAULT_CONFIG,
    get_config,
    get_section,
    get_value,
    reload_config,
    set_value,
)


class TestDefaults(unittest.TestCase):
    """Built-in defaults are safe."""

    def test_default_config_structure(self):
        self.assertIn("general", DEFAULT_CONFIG)
        self.assertIn("browser", DEFAULT_CONFIG)
        self.assertIn("filesystem", DEFAULT_CONFIG)
        self.assertIn("safety", DEFAULT_CONFIG)

    def test_default_assistant_name(self):
        self.assertEqual(DEFAULT_CONFIG["general"]["assistant_name"], "Jarvis")

    def test_default_debug_false(self):
        self.assertFalse(DEFAULT_CONFIG["general"]["debug"])

    def test_default_search_engine(self):
        engine = DEFAULT_CONFIG["browser"]["default_search_engine"]
        self.assertIn("{query}", engine)
        self.assertTrue(engine.startswith("https://"))

    def test_default_workspace(self):
        self.assertEqual(DEFAULT_CONFIG["filesystem"]["workspace"], "jarvis_workspace")

    def test_safety_defaults_true(self):
        self.assertTrue(
            DEFAULT_CONFIG["safety"]["require_confirmation_for_shutdown"]
        )
        self.assertTrue(
            DEFAULT_CONFIG["safety"]["require_confirmation_for_restart"]
        )
        self.assertTrue(
            DEFAULT_CONFIG["safety"]["require_confirmation_for_delete"]
        )


class TestLoading(unittest.TestCase):
    """Loading from a temporary settings file."""

    def _write_temp_config(self, data: dict) -> str:
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return path

    def test_load_valid_config(self):
        path = self._write_temp_config({
            "general": {"assistant_name": "JARVIS", "debug": True},
        })
        try:
            with patch.object(config_manager, "_settings_path", return_value=Path(path)):
                config = reload_config()
            self.assertEqual(get_value("general", "assistant_name"), "JARVIS")
            self.assertTrue(get_value("general", "debug"))
            # Untouched sections keep defaults
            self.assertTrue(
                get_value("safety", "require_confirmation_for_shutdown")
            )
        finally:
            os.remove(path)

    def test_missing_file_uses_defaults(self):
        with patch.object(
            config_manager, "_settings_path",
            return_value=Path(tempfile.gettempdir()) / "nonexistent-xyz.json",
        ), patch.object(config_manager, "_write_defaults_file"):
            config = reload_config()
        self.assertEqual(
            config["general"]["assistant_name"],
            DEFAULT_CONFIG["general"]["assistant_name"],
        )

    def test_invalid_json_falls_back(self):
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write('{"general":')  # malformed
        try:
            with patch.object(config_manager, "_settings_path", return_value=Path(path)):
                config = reload_config()
            # Falls back to defaults
            self.assertEqual(
                config["general"]["assistant_name"],
                DEFAULT_CONFIG["general"]["assistant_name"],
            )
        finally:
            os.remove(path)

    def test_non_object_json_falls_back(self):
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write('["not", "an", "object"]')
        try:
            with patch.object(config_manager, "_settings_path", return_value=Path(path)):
                config = reload_config()
            self.assertEqual(config["general"], DEFAULT_CONFIG["general"])
        finally:
            os.remove(path)


class TestTypeValidation(unittest.TestCase):
    """Wrong-typed values never weaken safety."""

    def _load_with(self, data: dict) -> dict:
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f)
        try:
            with patch.object(config_manager, "_settings_path", return_value=Path(path)):
                return reload_config()
        finally:
            os.remove(path)

    def test_debug_string_not_becomes_false(self):
        config = self._load_with({"general": {"debug": "false"}})
        # "false" is a string, not a bool → reset to safe default (False)
        self.assertIsInstance(config["general"]["debug"], bool)
        self.assertFalse(config["general"]["debug"])

    def test_debug_string_true_not_becomes_true(self):
        config = self._load_with({"general": {"debug": "true"}})
        # "true" is a string, not a bool → reset to safe default (False)
        self.assertFalse(config["general"]["debug"])

    def test_safety_wrong_type_forced_true(self):
        config = self._load_with({
            "safety": {"require_confirmation_for_shutdown": "false"}
        })
        self.assertTrue(config["safety"]["require_confirmation_for_shutdown"])

    def test_safety_missing_forced_true(self):
        config = self._load_with({"safety": {}})
        self.assertTrue(config["safety"]["require_confirmation_for_shutdown"])
        self.assertTrue(config["safety"]["require_confirmation_for_restart"])
        self.assertTrue(config["safety"]["require_confirmation_for_delete"])

    def test_safety_valid_false_stored_but_managed_safely(self):
        """A valid False is stored, but managers still require confirmation."""
        config = self._load_with({
            "safety": {"require_confirmation_for_shutdown": False}
        })
        # The value is stored as-is (valid bool)...
        self.assertFalse(config["safety"]["require_confirmation_for_shutdown"])
        # ...but the System Manager always enforces confirmation.
        from core.system_manager import _confirmation_enabled

        self.assertTrue(_confirmation_enabled("require_confirmation_for_shutdown"))


class TestGetters(unittest.TestCase):
    """get_value / get_section / get_config."""

    def test_get_value(self):
        self.assertEqual(get_value("general", "assistant_name"), "Jarvis")

    def test_get_value_default(self):
        self.assertEqual(get_value("unknown", "unknown", "fallback"), "fallback")

    def test_get_value_missing_returns_none(self):
        self.assertIsNone(get_value("unknown", "unknown"))

    def test_get_section(self):
        section = get_section("browser")
        self.assertIn("default_search_engine", section)
        self.assertIsInstance(section, dict)

    def test_get_section_missing_returns_empty(self):
        self.assertEqual(get_section("nonexistent"), {})

    def test_get_config_returns_copy(self):
        config = get_config()
        config["general"]["debug"] = "mutated"
        # Original is unchanged
        self.assertIsInstance(get_value("general", "debug"), bool)


class TestSetValue(unittest.TestCase):
    """set_value validates and persists atomically.

    All tests run against a temporary settings file so the real
    config/settings.json is never modified.
    """

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump({"general": {"debug": False}}, self._tmp)
        self._tmp.close()
        self._path_patcher = patch.object(
            config_manager, "_settings_path", return_value=Path(self._tmp.name)
        )
        self._path_patcher.start()
        reload_config()

    def tearDown(self):
        self._path_patcher.stop()
        os.remove(self._tmp.name)
        reload_config()  # reload the real config

    def test_set_value_updates_in_memory(self):
        set_value("general", "debug", True)
        self.assertTrue(get_value("general", "debug"))

    def test_set_value_persists_to_disk(self):
        set_value("general", "assistant_name", "JARVIS")
        # Read the temp file directly
        with open(self._tmp.name, encoding="utf-8") as f:
            saved = json.load(f)
        self.assertEqual(saved["general"]["assistant_name"], "JARVIS")

    def test_set_value_type_validation(self):
        with self.assertRaises(TypeError):
            set_value("general", "debug", "not-a-bool")

    def test_set_value_cannot_disable_safety(self):
        set_value("safety", "require_confirmation_for_shutdown", False)
        # Forced back to True
        self.assertTrue(get_value("safety", "require_confirmation_for_shutdown"))

    def test_set_value_safety_true_ok(self):
        set_value("safety", "require_confirmation_for_restart", True)
        self.assertTrue(get_value("safety", "require_confirmation_for_restart"))

    def test_set_value_atomic_no_corruption(self):
        """A valid set leaves the file parseable."""
        set_value("general", "debug", True)
        with open(self._tmp.name, encoding="utf-8") as f:
            saved = json.load(f)
        self.assertIsInstance(saved["general"]["debug"], bool)


class TestIntegration(unittest.TestCase):
    """Managers use the Configuration Manager."""

    def test_browser_search_engine_from_config(self):
        from core.browser_manager import _get_search_engine

        engine = _get_search_engine()
        self.assertIn("{query}", engine)
        self.assertTrue(engine.startswith("https://"))

    def test_browser_invalid_config_falls_back(self):
        from core.browser_manager import DEFAULT_SEARCH_ENGINE, _get_search_engine

        with patch.object(config_manager, "get_value", return_value=12345):
            self.assertEqual(_get_search_engine(), DEFAULT_SEARCH_ENGINE)

    def test_file_manager_workspace_from_config(self):
        from core.file_manager import _workspace_subdir

        name = _workspace_subdir()
        self.assertEqual(name, "jarvis_workspace")
        self.assertNotIn("/", name)
        self.assertNotIn("..", name)

    def test_file_manager_invalid_workspace_rejected(self):
        from core.file_manager import _workspace_subdir

        with patch.object(config_manager, "get_value", return_value="../../etc"):
            self.assertEqual(_workspace_subdir(), "jarvis_workspace")

    def test_system_manager_always_requires_confirmation(self):
        from core.system_manager import _confirmation_enabled

        self.assertTrue(_confirmation_enabled("require_confirmation_for_shutdown"))
        self.assertTrue(_confirmation_enabled("require_confirmation_for_restart"))
        self.assertTrue(_confirmation_enabled("require_confirmation_for_delete"))

    def test_system_manager_confirmation_not_disabled_by_config(self):
        from core.system_manager import _confirmation_enabled

        # Even if config were False, confirmation stays enabled
        with patch.object(config_manager, "get_value", return_value=False):
            self.assertTrue(
                _confirmation_enabled("require_confirmation_for_shutdown")
            )


class TestSecurity(unittest.TestCase):
    """The Configuration Manager never executes anything."""

    def test_no_eval_exec_in_source(self):
        source = Path(config_manager.__file__).read_text(encoding="utf-8")
        self.assertNotIn("eval(", source)
        self.assertNotIn("exec(", source)
        self.assertNotIn("os.system", source)
        self.assertNotIn("subprocess", source)

    def test_no_secrets_in_default_config(self):
        config = get_config()
        flattened = json.dumps(config).lower()
        for secret in ("api_key", "password", "token", "secret", "cookie"):
            self.assertNotIn(secret, flattened)

    def test_set_value_path_traversal_rejected(self):
        """Section/key are used as fixed keys, not file paths."""
        # set_value writes only to the fixed settings file; a traversal
        # attempt in a key cannot escape because keys are dict lookups.
        # The write is redirected to a temporary file so the real
        # config/settings.json is never modified by this test.
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            with patch.object(
                config_manager, "_settings_path", return_value=Path(path)
            ):
                set_value("general", "debug", True)
            with open(path, encoding="utf-8") as f:
                saved = json.load(f)
            self.assertTrue(saved["general"]["debug"])
        finally:
            os.remove(path)
            reload_config()  # restore the real configuration in memory

        # The real settings path is fixed under config/ — section and
        # key values never influence where the file lives.
        real_path = config_manager._settings_path()
        self.assertEqual(real_path.name, "settings.json")
        self.assertEqual(real_path.parent.name, "config")


class TestConfirmationTimeoutConfig(unittest.TestCase):
    """safety.confirmation_timeout_seconds validates to a safe positive int.

    A malformed value must never make confirmations immortal or
    instantly expiring. Tests run against a temporary settings file so
    the real config/settings.json is never modified.
    """

    def _load_with(self, data: dict) -> dict:
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f)
        try:
            with patch.object(
                config_manager, "_settings_path", return_value=Path(path)
            ):
                return reload_config()
        finally:
            os.remove(path)

    def tearDown(self):
        reload_config()  # restore the real configuration in memory

    def test_default_is_60(self):
        self.assertEqual(
            DEFAULT_CONFIG["safety"]["confirmation_timeout_seconds"], 60
        )

    def test_valid_value_kept(self):
        config = self._load_with({
            "safety": {"confirmation_timeout_seconds": 30}
        })
        self.assertEqual(
            config["safety"]["confirmation_timeout_seconds"], 30
        )

    def test_string_value_reset_to_default(self):
        config = self._load_with({
            "safety": {"confirmation_timeout_seconds": "600"}
        })
        self.assertEqual(
            config["safety"]["confirmation_timeout_seconds"], 60
        )

    def test_bool_value_reset_to_default(self):
        # isinstance(True, int) is True in Python — a bool must not
        # pass as a 1-second timeout.
        config = self._load_with({
            "safety": {"confirmation_timeout_seconds": True}
        })
        self.assertEqual(
            config["safety"]["confirmation_timeout_seconds"], 60
        )

    def test_zero_reset_to_default(self):
        config = self._load_with({
            "safety": {"confirmation_timeout_seconds": 0}
        })
        self.assertEqual(
            config["safety"]["confirmation_timeout_seconds"], 60
        )

    def test_negative_reset_to_default(self):
        config = self._load_with({
            "safety": {"confirmation_timeout_seconds": -5}
        })
        self.assertEqual(
            config["safety"]["confirmation_timeout_seconds"], 60
        )

    def test_set_value_rejects_non_positive(self):
        with self.assertRaises(ValueError):
            set_value("safety", "confirmation_timeout_seconds", 0)
        with self.assertRaises(ValueError):
            set_value("safety", "confirmation_timeout_seconds", -1)
        with self.assertRaises(ValueError):
            set_value("safety", "confirmation_timeout_seconds", True)

    def test_set_value_rejects_wrong_type(self):
        with self.assertRaises(TypeError):
            set_value("safety", "confirmation_timeout_seconds", "60")


class TestHistoryMaxEntriesConfig(unittest.TestCase):
    """history.max_entries validates to a safe positive int (Step 20).

    A malformed value must never cause unlimited history growth (or a
    zero-sized history). Tests run against a temporary settings file
    so the real config/settings.json is never modified.
    """

    def _load_with(self, data: dict) -> dict:
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f)
        try:
            with patch.object(
                config_manager, "_settings_path", return_value=Path(path)
            ):
                return reload_config()
        finally:
            os.remove(path)

    def tearDown(self):
        reload_config()  # restore the real configuration in memory

    def test_default_is_100(self):
        self.assertEqual(DEFAULT_CONFIG["history"]["max_entries"], 100)

    def test_valid_value_kept(self):
        config = self._load_with({"history": {"max_entries": 25}})
        self.assertEqual(config["history"]["max_entries"], 25)

    def test_string_value_reset_to_default(self):
        config = self._load_with({"history": {"max_entries": "1000"}})
        self.assertEqual(config["history"]["max_entries"], 100)

    def test_bool_value_reset_to_default(self):
        # isinstance(True, int) is True in Python — a bool must not
        # pass as a history limit.
        config = self._load_with({"history": {"max_entries": True}})
        self.assertEqual(config["history"]["max_entries"], 100)

    def test_zero_reset_to_default(self):
        config = self._load_with({"history": {"max_entries": 0}})
        self.assertEqual(config["history"]["max_entries"], 100)

    def test_negative_reset_to_default(self):
        config = self._load_with({"history": {"max_entries": -5}})
        self.assertEqual(config["history"]["max_entries"], 100)

    def test_null_reset_to_default(self):
        config = self._load_with({"history": {"max_entries": None}})
        self.assertEqual(config["history"]["max_entries"], 100)

    def test_set_value_rejects_non_positive(self):
        with self.assertRaises(ValueError):
            set_value("history", "max_entries", 0)
        with self.assertRaises(ValueError):
            set_value("history", "max_entries", -1)
        with self.assertRaises(ValueError):
            set_value("history", "max_entries", True)

    def test_set_value_rejects_wrong_type(self):
        with self.assertRaises(TypeError):
            set_value("history", "max_entries", "100")

    def test_set_value_accepts_valid_limit(self):
        set_value("history", "max_entries", 250)
        self.assertEqual(get_value("history", "max_entries"), 250)


class TestEnvironmentLoading(unittest.TestCase):
    def test_dotenv_loads_from_data_root_without_overriding_process(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "JARVIS_DOTENV_TEST=from-file\n"
                "JARVIS_DOTENV_PRECEDENCE=from-file\n",
                encoding="utf-8",
            )
            with patch.object(config_manager.paths, "data_root", return_value=Path(temp_dir)):
                with patch.dict(os.environ, {"JARVIS_DOTENV_PRECEDENCE": "from-process"}, clear=False):
                    os.environ.pop("JARVIS_DOTENV_TEST", None)
                    config_manager._load_environment()
                    self.assertEqual(os.environ["JARVIS_DOTENV_TEST"], "from-file")
                    self.assertEqual(os.environ["JARVIS_DOTENV_PRECEDENCE"], "from-process")
                    os.environ.pop("JARVIS_DOTENV_TEST", None)


if __name__ == "__main__":
    unittest.main()
