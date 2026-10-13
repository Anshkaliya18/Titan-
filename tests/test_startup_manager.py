"""
Startup Manager tests (Step 23 — Phases 13–15, 28).

What is proven here:
- The public API (enable/disable/status) writes exactly ONE registry
  value, is idempotent, and never raises into the pipeline.
- enable_startup() accepts NO user-supplied path: the command comes
  from the running installation only, and is validated before use.
- No shell/os.system/eval-style primitives exist anywhere in the new
  source (AST scan, same rules as the Step 21 security suite).
- Malformed / unexpected registry state is reported, never repaired
  implicitly and never allowed to crash (Phase 28).
- Failure on a non-Windows machine fails safe instead of raising.

Safety model for this file:
- The unit tests replace the three registry leaves with an in-memory
  store — the real user's startup entry is never touched.
- ONE integration test exercises the real registry, but under an
  isolated, Jarvis-test-specific key that is removed again in a
  finally block. It never writes the real "Jarvis" Run value, and the
  command it stores is never executed.
"""

import ast
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

# Make project modules importable (mirrors every other test file)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import startup_manager as sm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Files written by Step 23 that must stay free of dangerous primitives.
NEW_SOURCE_FILES = (
    ROOT / "core" / "startup_manager.py",
    ROOT / "core" / "paths.py" if (ROOT / "core" / "paths.py").exists() else None,
    ROOT / "core" / "version.py",
    ROOT / "core" / "cli.py",
    ROOT / "utils" / "paths.py",
    ROOT / "build.py",
    ROOT / "jarvis.spec",
)
NEW_SOURCE_FILES = tuple(p for p in NEW_SOURCE_FILES if p is not None)


@contextmanager
def fake_registry(initial: dict | None = None):
    """
    Replace the three registry leaves with an in-memory store.

    Yields the store so a test can inspect what was written. The real
    HKCU Run key is never opened.
    """
    store: dict[str, str] = dict(initial or {})

    def _read():
        return True, store.get(sm.VALUE_NAME)

    def _write(command: str) -> None:
        store[sm.VALUE_NAME] = command

    def _delete() -> bool:
        return store.pop(sm.VALUE_NAME, None) is not None

    with patch.object(sm, "_registry_read", _read), patch.object(
        sm, "_registry_write", _write
    ), patch.object(sm, "_registry_delete", _delete):
        yield store


@contextmanager
def missing_registry():
    """Simulate an environment where the registry is unavailable."""
    with patch.object(sm, "winreg", None):
        yield


class TestEnableDisableCycle(unittest.TestCase):
    """The full cycle against an in-memory registry store."""

    def test_enable_writes_exactly_one_value(self):
        with fake_registry() as store:
            result = sm.enable_startup()
        self.assertTrue(result["success"])
        self.assertEqual(list(store.keys()), [sm.VALUE_NAME])
        # The stored command is the validated install command.
        self.assertEqual(store[sm.VALUE_NAME], sm._resolve_install_command())
        # Data tells the user exactly where the change landed.
        self.assertEqual(result["data"]["location"], sm.RUN_KEY_DISPLAY)
        self.assertEqual(result["data"]["value"], sm.VALUE_NAME)
        self.assertIn("command", result["data"])

    def test_enable_is_idempotent(self):
        """Enabling twice must never create a duplicate entry."""
        with fake_registry() as store:
            first = sm.enable_startup()
            second = sm.enable_startup()
            # A registry Run key holds one value per name — the store
            # mirrors that: still exactly one entry.
            self.assertEqual(list(store.keys()), [sm.VALUE_NAME])
            self.assertEqual(len(store), 1)
        self.assertTrue(first["success"])
        self.assertTrue(second["success"])
        self.assertIn("already starts", second["message"])

    def test_enable_then_disable_removes_only_our_value(self):
        with fake_registry({"SomeOtherApp": "other.exe"}) as store:
            sm.enable_startup()
            result = sm.disable_startup()
            removed = set(store.keys())
        self.assertTrue(result["success"])
        self.assertIn("no longer start", result["message"])
        # Only our value is gone; the unrelated entry survives.
        self.assertEqual(removed, {"SomeOtherApp"})
        self.assertFalse(sm.is_startup_enabled())

    def test_disable_twice_is_safe(self):
        with fake_registry():
            first = sm.disable_startup()
            second = sm.disable_startup()
        self.assertTrue(first["success"])
        self.assertTrue(second["success"])
        # Second call reports the state instead of failing.
        self.assertIn("does not start", second["message"])

    def test_disable_without_enabling_is_a_safe_noop(self):
        with fake_registry():
            result = sm.disable_startup()
            enabled = sm.is_startup_enabled()
        self.assertTrue(result["success"])
        self.assertFalse(enabled)

    def test_status_reflects_state_both_ways(self):
        with fake_registry():
            off = sm.startup_status()
            sm.enable_startup()
            on = sm.startup_status()
            sm.disable_startup()
            after = sm.startup_status()
        self.assertTrue(off["success"])
        self.assertFalse(off["data"]["enabled"])
        self.assertIn("does not start", off["message"])
        self.assertTrue(on["data"]["enabled"])
        self.assertIn("starts with Windows", on["message"])
        self.assertFalse(after["data"]["enabled"])

    def test_status_is_read_only(self):
        """Querying status never creates, changes, or removes a value."""
        with fake_registry() as store:
            sm.startup_status()
            sm.startup_status()
            self.assertEqual(store, {})
        with fake_registry() as store:
            sm.enable_startup()
            before = dict(store)
            sm.startup_status()
            self.assertEqual(store, before)

    def test_status_includes_command_for_diagnosis(self):
        with fake_registry():
            sm.enable_startup()
            status = sm.startup_status()
        command = status["data"].get("command", "")
        self.assertTrue(command)
        # The stored target is this installation's own program.
        target = sm._target_path(command)
        self.assertIsNotNone(target)
        self.assertTrue(target.is_file())
        self.assertTrue(status["data"].get("target_exists"))


class TestInstallationValidation(unittest.TestCase):
    """Phase 15: the executable comes from the install, never input."""

    def test_public_api_takes_no_arguments(self):
        """No path (or any other) parameter can ever be supplied."""
        import inspect

        for function in (sm.enable_startup, sm.disable_startup, sm.startup_status):
            with self.subTest(function=function.__name__):
                parameters = inspect.signature(function).parameters
                self.assertEqual(list(parameters), [])

    def test_source_mode_command_is_interpreter_plus_entry_point(self):
        if sm.is_frozen():
            self.skipTest("source-mode check (running packaged)")
        command = sm._resolve_install_command()
        self.assertTrue(command.startswith('"'))
        self.assertTrue(command.endswith('"'))
        # Quoted interpreter first, quoted main.py last.
        self.assertIn("python", command.lower())
        self.assertTrue(command.rstrip().endswith('main.py"'))
        # Two quoted tokens, nothing else — no extra arguments.
        self.assertEqual(command.count('"'), 4)

    def test_unverifiable_interpreter_is_refused(self):
        with fake_registry() as store:
            with patch.object(
                sm.sys,
                "executable",
                str(Path(tempfile.gettempdir()) / "not-a-real-python.exe"),
            ):
                result = sm.enable_startup()
            self.assertEqual(store, {})
        self.assertFalse(result["success"])
        self.assertIn("verify", result["message"].lower())

    def test_unexpected_program_type_is_refused(self):
        """Only .exe/.py programs are acceptable (a stray .txt is not)."""
        handle, path = tempfile.mkstemp(suffix=".txt")
        os.close(handle)
        try:
            with fake_registry() as store:
                with patch.object(sm.sys, "executable", path):
                    result = sm.enable_startup()
                self.assertEqual(store, {})
            self.assertFalse(result["success"])
        finally:
            os.remove(path)

    def test_relative_interpreter_path_is_refused(self):
        with fake_registry() as store:
            with patch.object(sm.sys, "executable", "python.exe"):
                result = sm.enable_startup()
            self.assertEqual(store, {})
        self.assertFalse(result["success"])

    def test_pythonw_prefers_console_python(self):
        """Phase 19: a packaged registration must keep its console."""
        with tempfile.TemporaryDirectory() as tmp:
            pythonw = Path(tmp) / "pythonw.exe"
            python = Path(tmp) / "python.exe"
            pythonw.write_bytes(b"")
            python.write_bytes(b"")
            with patch.object(sm.sys, "executable", str(pythonw)):
                command = sm._resolve_install_command()
        self.assertIn("python.exe", command)
        self.assertNotIn("pythonw.exe", command)

    def test_target_path_parsing_is_defensive(self):
        self.assertIsNone(sm._target_path(""))
        self.assertIsNone(sm._target_path("   "))
        self.assertIsNone(sm._target_path('"'))
        self.assertIsNone(sm._target_path('""'))
        self.assertEqual(sm._target_path('"C:\\a\\b.exe" arg'), Path("C:\\a\\b.exe"))
        self.assertEqual(sm._target_path("C:\\a\\b.exe rest"), Path("C:\\a\\b.exe"))


class TestMalformedRegistryState(unittest.TestCase):
    """Phase 28: unexpected state is reported, never crashed on."""

    def test_empty_value_is_not_reported_as_enabled(self):
        with fake_registry({sm.VALUE_NAME: ""}):
            enabled = sm.is_startup_enabled()
            status = sm.startup_status()
        self.assertFalse(enabled)
        self.assertFalse(status["data"]["enabled"])
        self.assertIn("does not start", status["message"])

    def test_stale_entry_is_reported_not_repaired(self):
        ghost = '"C:\\definitely\\gone\\jarvis-fake.exe"'
        with fake_registry({sm.VALUE_NAME: ghost}):
            status = sm.startup_status()
            # Status alone never repairs it.
            still_there = sm.is_startup_enabled()
        self.assertTrue(status["data"]["enabled"])
        self.assertFalse(status["data"]["target_exists"])
        self.assertIn("no longer exists", status["message"])
        self.assertTrue(still_there)

    def test_non_string_registry_value_is_tolerated(self):
        """A wrong-typed value must not crash status reporting."""

        def _read():
            return True, 12345  # wrong type, but readable

        with patch.object(sm, "_registry_read", _read):
            enabled = sm.is_startup_enabled()
            status = sm.startup_status()
        self.assertTrue(enabled)
        self.assertTrue(status["success"])

    def test_registry_read_error_fails_safe(self):
        """A read failure reports \"disabled\" rather than raising."""

        def _read():
            raise OSError("access denied")

        with patch.object(sm, "_registry_read", _read):
            try:
                enabled = sm.is_startup_enabled()
                self.assertFalse(enabled)
            except Exception:  # pragma: no cover - the failure we forbid
                self.fail("is_startup_enabled must not raise")


class TestUnavailableRegistry(unittest.TestCase):
    """Non-Windows (or unavailable registry) fails safe, never raises."""

    def test_status_enable_disable_fail_safe(self):
        with missing_registry():
            status = sm.startup_status()
            enable = sm.enable_startup()
            disable = sm.disable_startup()
            enabled = sm.is_startup_enabled()
        for result in (status, enable, disable):
            self.assertFalse(result["success"])
            self.assertIn("Windows", result["message"])
        self.assertFalse(enabled)

    def test_failure_result_has_standard_shape(self):
        with missing_registry():
            result = sm.enable_startup()
        self.assertEqual(set(result.keys()), {"success", "message", "data"})
        self.assertIsInstance(result["message"], str)
        self.assertIsInstance(result["data"], dict)


class TestRealRegistryIsolatedKey(unittest.TestCase):
    """
    Integration: the REAL registry API, under a disposable test key.

    Writes HKCU\\Software\\JarvisStep23Test_<pid> (never the real
    Jarvis Run value) and removes it again in tearDown.
    """

    def setUp(self):
        if sm.winreg is None:
            self.skipTest("winreg unavailable (non-Windows)")
        self.test_key = rf"Software\JarvisStep23Test_{os.getpid()}"
        self.test_name = "JarvisTestEntry"
        # Point every operation at the disposable key.
        patcher_key = patch.object(sm, "RUN_KEY", self.test_key)
        patcher_name = patch.object(sm, "VALUE_NAME", self.test_name)
        patcher_key.start()
        patcher_name.start()
        self.addCleanup(patcher_name.stop)
        self.addCleanup(patcher_key.stop)
        # Guarantee a clean slate (and verify we start from nothing).
        self._delete_test_key()
        self.assertFalse(sm.is_startup_enabled())

    def tearDown(self):
        self._delete_test_key()

    def _delete_test_key(self):
        try:
            with sm.winreg.OpenKey(
                sm.winreg.HKEY_CURRENT_USER, self.test_key, 0, sm.winreg.KEY_SET_VALUE
            ) as key:
                try:
                    sm.winreg.DeleteValue(key, self.test_name)
                except FileNotFoundError:
                    pass
        except OSError:
            pass
        try:
            sm.winreg.DeleteKey(sm.winreg.HKEY_CURRENT_USER, self.test_key)
        except OSError:
            pass

    def _raw_read(self):
        try:
            with sm.winreg.OpenKey(
                sm.winreg.HKEY_CURRENT_USER, self.test_key, 0, sm.winreg.KEY_READ
            ) as key:
                value, _kind = sm.winreg.QueryValueEx(key, self.test_name)
                return value
        except FileNotFoundError:
            return None

    def test_full_cycle_against_real_registry(self):
        self.assertFalse(sm.is_startup_enabled())
        self.assertIsNone(self._raw_read())

        enabled = sm.enable_startup()
        self.assertTrue(enabled["success"])
        self.assertTrue(sm.is_startup_enabled())
        # The value the registry really holds matches what we report.
        self.assertEqual(self._raw_read(), enabled["data"]["command"])

        disabled = sm.disable_startup()
        self.assertTrue(disabled["success"])
        self.assertFalse(sm.is_startup_enabled())
        self.assertIsNone(self._raw_read())

    def test_real_registry_enable_is_idempotent(self):
        sm.enable_startup()
        sm.enable_startup()
        # One name, one value — the raw read is a single string.
        self.assertIsInstance(self._raw_read(), str)
        sm.disable_startup()
        self.assertIsNone(self._raw_read())

    def test_command_points_at_this_installation(self):
        sm.enable_startup()
        raw = self._raw_read()
        sm.disable_startup()
        self.assertIsNotNone(raw)
        target = sm._target_path(raw)
        self.assertIsNotNone(target)
        self.assertTrue(target.is_file())
        # Never an arbitrary path: it is our resolved install command.
        self.assertEqual(raw, sm._resolve_install_command())


class TestSecurity(unittest.TestCase):
    """No dangerous execution primitives in any Step 23 source file."""

    def _forbidden_imports(self, tree):
        found = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in {"pickle", "marshal", "shelve"}:
                        found.append(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                if root in {"pickle", "marshal", "shelve"}:
                    found.append(f"from {node.module} import ...")
        return found

    def _forbidden_calls(self, tree):
        found = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            # eval(...) / exec(...) / __import__(...) / compile(...)
            if isinstance(func, ast.Name) and func.id in {
                "eval",
                "exec",
                "__import__",
                "compile",
            }:
                found.append(f"call to {func.id}()")
            # os.system(...) / os.popen(...)
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                if func.value.id == "os" and func.attr in {"system", "popen"}:
                    found.append(f"call to os.{func.attr}()")
            # shell=True keyword anywhere
            for keyword in node.keywords:
                if keyword.arg == "shell" and not (
                    isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is False
                ):
                    found.append("shell=True")
        return found

    def test_no_dangerous_primitives_in_new_source(self):
        for path in NEW_SOURCE_FILES:
            with self.subTest(file=path.name):
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(path))
                violations = self._forbidden_imports(tree) + self._forbidden_calls(tree)
                self.assertEqual(violations, [], f"{path.name}: {violations}")

    def test_startup_manager_never_imports_subprocess(self):
        """It builds a command string but never runs one itself."""
        source = (ROOT / "core" / "startup_manager.py").read_text(encoding="utf-8")
        self.assertNotIn("import subprocess", source)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertNotIn(
                    "subprocess", [alias.name for alias in node.names]
                )

    def test_startup_manager_only_touches_the_run_key(self):
        """Registry operations are pinned to the HKCU Run key + value."""
        source = (ROOT / "core" / "startup_manager.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        attributes = [
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "winreg"
        ]
        # Only the documented operations may appear.
        allowed = {
            "HKEY_CURRENT_USER",
            "OpenKey",
            "CreateKeyEx",
            "QueryValueEx",
            "SetValueEx",
            "DeleteValue",
            "KEY_READ",
            "KEY_SET_VALUE",
            "REG_SZ",
        }
        self.assertTrue(set(attributes), "expected winreg usage")
        self.assertTrue(
            set(attributes) <= allowed,
            f"unexpected winreg operations: {set(attributes) - allowed}",
        )
        # The key/value themselves are module constants, not input.
        self.assertIn("RUN_KEY = r", source)
        self.assertIn("VALUE_NAME", source)


if __name__ == "__main__":
    unittest.main()
