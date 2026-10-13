"""
Security-focused integration tests (Step 21 / Phase 22).

Core guarantee under test:

    USER INPUT MUST NEVER BECOVEROME ARBITRARY SHELL EXECUTION.

The suite covers shell/command injection, path traversal, dangerous
URL schemes, arbitrary executable attempts, confirmation bypass,
history replay, configuration safety, and dangerous multi-command
chaining — plus a static scan proving no shell=True / os.system /
eval / exec / pickle exists anywhere in the source tree.

Every test either uses temporary directories or patches the leaf
functions that could change real system state. Nothing in this file
can launch an application, open a browser window, delete a real
file, or touch Windows power state.
"""

import ast
import io
import os
import sys
import tempfile
import time
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# Add parent directory to path so we can import core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import main
from core.application_manager import launch_application
from core.browser_manager import is_valid_url, open_url, open_url_in_browser
from core.command_history import CommandHistory, set_command_history
from core.command_validator import validate_command
from core.file_manager import (
    create_folder,
    delete_file,
    is_safe_path,
    open_file,
)
from core.session_state import SessionState

ROOT = Path(__file__).resolve().parent.parent


@contextmanager
def no_system_effects():
    """
    Patch every leaf function that could change real system state.

    Any attempt to launch a process, run a shell command, open a
    real browser/file, lock, shut down, or restart is captured by a
    mock instead of reaching the machine.
    """
    with patch("subprocess.Popen") as popen, \
            patch("subprocess.run") as run, \
            patch("os.system") as system, \
            patch("os.startfile", create=True) as startfile, \
            patch("webbrowser.open") as web_open, \
            patch("core.executor.shutdown_computer") as shutdown, \
            patch("core.executor.restart_computer") as restart, \
            patch("core.executor.lock_computer") as lock:
        yield SimpleNamespace(
            popen=popen,
            run=run,
            system=system,
            startfile=startfile,
            web_open=web_open,
            shutdown=shutdown,
            restart=restart,
            lock=lock,
        )


class _SecurityTurnTestCase(unittest.TestCase):
    """Fresh session/history per test with a captured-stdout turn."""

    def setUp(self):
        self._original_session = main.session
        main.session = SessionState()
        self.history = CommandHistory(max_entries=50)
        set_command_history(self.history)
        self._buffer = io.StringIO()

    def tearDown(self):
        main.session = self._original_session
        set_command_history(None)

    def _run_turn(self, text):
        with patch("main.get_input", return_value=text):
            with redirect_stdout(self._buffer):
                main._process_once()
        return self._buffer.getvalue()

    def assert_no_effects(self, effects):
        for name, mock in sorted(vars(effects).items()):
            mock.assert_not_called()


# ---------------------------------------------------------------------------
# 1. Static source audit
# ---------------------------------------------------------------------------

class TestStaticSourceSafety(unittest.TestCase):
    """No dangerous execution primitive exists anywhere in the source."""

    FORBIDDEN_BUILTINS = {"eval", "exec", "__import__", "compile"}
    FORBIDDEN_MODULES = {"pickle", "marshal", "shelve"}

    @staticmethod
    def _source_files():
        files = [ROOT / "main.py"]
        for dirname in ("core", "utils", "commands", "config"):
            directory = ROOT / dirname
            if directory.is_dir():
                files.extend(sorted(directory.glob("*.py")))
        return [f for f in files if f.is_file()]

    def _violations(self, path):
        violations = []
        rel = path.relative_to(ROOT)
        tree = ast.parse(
            path.read_text(encoding="utf-8"), filename=str(path)
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in self.FORBIDDEN_MODULES:
                        violations.append(
                            f"{rel}:{node.lineno} imports {alias.name}"
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] in \
                        self.FORBIDDEN_MODULES:
                    violations.append(
                        f"{rel}:{node.lineno} imports {node.module}"
                    )
            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and \
                        func.id in self.FORBIDDEN_BUILTINS:
                    violations.append(
                        f"{rel}:{node.lineno} calls {func.id}()"
                    )
                if isinstance(func, ast.Attribute):
                    owner = func.value.id if isinstance(
                        func.value, ast.Name
                    ) else None
                    if owner == "os" and func.attr in ("system", "popen"):
                        violations.append(
                            f"{rel}:{node.lineno} calls os.{func.attr}()"
                        )
                for kw in node.keywords:
                    if kw.arg == "shell" and isinstance(kw.value, ast.Constant) \
                            and kw.value.value is True:
                        violations.append(
                            f"{rel}:{node.lineno} uses shell=True"
                        )
        return violations

    def test_no_dangerous_execution_primitives_in_source(self):
        all_violations = []
        for path in self._source_files():
            all_violations.extend(self._violations(path))
        self.assertEqual(
            all_violations, [],
            "Unsafe execution patterns found: "
            + "; ".join(all_violations),
        )

    def test_every_source_file_was_scanned(self):
        # Guard against the scan silently skipping the tree
        files = self._source_files()
        names = {f.name for f in files}
        self.assertIn("main.py", names)
        self.assertIn("executor.py", names)
        self.assertIn("file_manager.py", names)
        self.assertGreater(len(files), 15)


# ---------------------------------------------------------------------------
# 2. Shell / command injection
# ---------------------------------------------------------------------------

class TestShellInjectionNeverExecutes(_SecurityTurnTestCase):
    """Malicious metacharacter input never reaches a shell."""

    PAYLOADS = [
        "; shutdown",
        "& whoami",
        "| dir",
        "&& del important.txt",
        "open chrome & whoami",
        "open chrome | shutdown computer",
        "`whoami`",
        "$(whoami)",
        "open chrome; del important.txt",
        "%PATH%",
    ]

    def test_payloads_never_reach_a_shell(self):
        for payload in self.PAYLOADS:
            with self.subTest(payload=payload):
                main.session = SessionState()
                with no_system_effects() as effects:
                    output = self._run_turn(payload)
                self.assert_no_effects(effects)
                # The user always receives a safe response
                self.assertTrue(output.strip())

    def test_benign_command_still_works_under_tripwires(self):
        """Positive control: the tripwires do not break normal flow."""
        with no_system_effects() as effects:
            output = self._run_turn("help")
        self.assert_no_effects(effects)
        self.assertIn("open", output.lower())


# ---------------------------------------------------------------------------
# 3. Path traversal
# ---------------------------------------------------------------------------

@contextmanager
def temp_workspace():
    """Isolated safe directory + workspace; real user folders untouched."""
    with tempfile.TemporaryDirectory() as tmp:
        safe = Path(tmp) / "safe"
        safe.mkdir()
        (safe / "inside.txt").write_text("in", encoding="utf-8")
        outside = Path(tmp) / "outside.txt"
        outside.write_text("out", encoding="utf-8")
        sibling = Path(tmp) / "safe2"
        sibling.mkdir()
        (sibling / "file.txt").write_text("sib", encoding="utf-8")
        with patch(
            "core.file_manager.get_safe_directories", return_value=[safe]
        ), patch("core.file_manager.DEFAULT_WORKSPACE", safe):
            yield safe, outside, sibling


class TestPathTraversalSafety(unittest.TestCase):
    """Resolved-Path containment — never string-prefix checks."""

    @patch("core.file_manager.os.startfile", create=True)
    def test_open_dotdot_traversal_denied(self, mock_startfile):
        with temp_workspace() as (safe, outside, _):
            for name in (
                "../outside.txt",
                str(safe / ".." / "outside.txt"),
                str(Path("..") / "outside.txt"),
            ):
                with self.subTest(name=name):
                    result = open_file(name)
                    self.assertFalse(result["success"])
                    self.assertIn("access denied", result["message"].lower())
            mock_startfile.assert_not_called()
            self.assertTrue(outside.exists())

    @patch("core.file_manager.os.startfile", create=True)
    def test_open_deep_traversal_denied(self, mock_startfile):
        with temp_workspace() as _:
            for name in ("../../secret.txt", "../../../etc/passwd"):
                with self.subTest(name=name):
                    result = open_file(name)
                    self.assertFalse(result["success"])
            mock_startfile.assert_not_called()

    def test_delete_traversal_never_deletes_outside_file(self):
        with temp_workspace() as (_, outside, _):
            for name in ("../outside.txt", str(Path("..") / "outside.txt")):
                with self.subTest(name=name):
                    result = delete_file(name, confirmed=True)
                    self.assertFalse(result["success"])
            self.assertTrue(outside.exists())

    def test_delete_deep_traversal_is_unsafe(self):
        with temp_workspace() as _:
            self.assertFalse(
                is_safe_path(Path("..") / ".." / "secret.txt")
            )

    def test_create_folder_traversal_denied(self):
        with temp_workspace() as (safe, _, _):
            for name in ("../evil", "..\\..\\evil", str(Path("..") / "evil")):
                with self.subTest(name=name):
                    result = create_folder(name)
                    self.assertFalse(result["success"])
            self.assertFalse((safe.parent / "evil").exists())
            self.assertFalse((safe.parent.parent / "evil").exists())

    def test_create_folder_absolute_path_denied(self):
        with temp_workspace() as _:
            result = create_folder("C:\\Windows\\System32\\evil")
            self.assertFalse(result["success"])
            self.assertFalse(
                os.path.exists("C:\\Windows\\System32\\evil")
            )

    def test_containment_uses_resolved_paths_not_string_prefix(self):
        """A path that merely shares a string prefix must not match,
        and a path whose unresolved form starts inside but resolves
        outside must be rejected."""
        with temp_workspace() as (safe, _, sibling):
            # Sibling directory sharing the "safe" string prefix
            self.assertTrue(
                str(sibling / "file.txt").startswith(str(safe))
            )
            self.assertFalse(is_safe_path(sibling / "file.txt"))
            # Resolves outside despite starting under safe/
            escaping = safe / "sub" / ".." / ".." / "outside.txt"
            self.assertFalse(is_safe_path(escaping))

    @patch("core.file_manager.os.startfile", create=True)
    def test_windows_system_paths_are_unsafe(self, mock_startfile):
        with temp_workspace() as _:
            self.assertFalse(is_safe_path("C:\\Windows\\System32\\cmd.exe"))
            result = open_file("C:\\Windows\\System32\\drivers\\etc\\hosts")
            self.assertFalse(result["success"])
            mock_startfile.assert_not_called()


# ---------------------------------------------------------------------------
# 4. Dangerous URL schemes
# ---------------------------------------------------------------------------

class TestUrlSchemeSafety(unittest.TestCase):
    DANGEROUS = [
        "file:///etc/passwd",
        "javascript:alert(1)",
        "data:text/html,<script>x</script>",
        "vbscript:msgbox(1)",
        "cmd:/c calc",
        "powershell:Get-Process",
        "shell:/bin/sh",
        "ftp://example.com/file.txt",
    ]
    VALID = [
        "https://example.com",
        "http://example.com",
        "https://github.com/user/repo",
    ]

    def test_dangerous_schemes_rejected_by_validator(self):
        for url in self.DANGEROUS:
            with self.subTest(url=url):
                result = validate_command(
                    {"intent": "open_url", "target": url, "arguments": {}}
                )
                self.assertFalse(result["valid"])

    def test_valid_urls_accepted_by_validator(self):
        for url in self.VALID:
            with self.subTest(url=url):
                result = validate_command(
                    {"intent": "open_url", "target": url, "arguments": {}}
                )
                self.assertTrue(result["valid"])

    def test_is_valid_url_rejects_dangerous_schemes(self):
        for url in self.DANGEROUS:
            with self.subTest(url=url):
                self.assertFalse(is_valid_url(url))
        for url in self.VALID:
            with self.subTest(url=url):
                self.assertTrue(is_valid_url(url))

    def test_browser_manager_refuses_dangerous_urls(self):
        for url in self.DANGEROUS:
            with self.subTest(url=url):
                with patch("core.browser_manager.webbrowser.open") as web_open:
                    result = open_url(url)
                self.assertFalse(result["success"])
                web_open.assert_not_called()

    def test_specific_browser_refuses_dangerous_url(self):
        with patch("core.browser_manager.subprocess.Popen") as popen:
            result = open_url_in_browser("chrome", "javascript:alert(1)")
        self.assertFalse(result["success"])
        popen.assert_not_called()

    def test_search_query_reaches_only_mocked_webbrowser(self):
        from core.browser_manager import search_web

        with patch(
            "core.browser_manager.webbrowser.open", return_value=True
        ) as web_open:
            result = search_web("cats and dogs")
        self.assertTrue(result["success"])
        web_open.assert_called_once()
        opened = web_open.call_args.args[0]
        self.assertTrue(opened.startswith("https://"))
        self.assertIn("cats+and+dogs", opened)

    def test_pipeline_dangerous_url_never_opens(self):
        """Even a full pipeline attempt with a dangerous URL never
        reaches the browser."""
        original_session = main.session
        main.session = SessionState()
        set_command_history(CommandHistory(max_entries=20))
        buffer = io.StringIO()
        try:
            with patch("core.browser_manager.webbrowser.open") as web_open, \
                    patch("main.get_input",
                          return_value="open javascript:alert(1)"):
                with redirect_stdout(buffer):
                    main._process_once()
            web_open.assert_not_called()
            self.assertTrue(buffer.getvalue().strip())
        finally:
            main.session = original_session
            set_command_history(None)


# ---------------------------------------------------------------------------
# 5. Arbitrary executable attempts
# ---------------------------------------------------------------------------

class TestArbitraryExecutionSafety(unittest.TestCase):
    """Only exact allowlisted names launch, always shell=False."""

    SMUGGLED_TARGETS = [
        "cmd /c whoami",
        "cmd /c del important.txt",
        "c:\\windows\\system32\\cmd.exe",
        "powershell -enc AAAA",
        "chrome & notepad",
        "chrome && shutdown",
    ]

    def test_argument_smuggling_rejected_by_allowlist(self):
        for target in self.SMUGGLED_TARGETS:
            with self.subTest(target=target):
                with patch(
                    "core.application_manager.subprocess.Popen"
                ) as popen:
                    result = launch_application(target)
                self.assertFalse(result["success"])
                popen.assert_not_called()

    def test_pipeline_smuggling_never_launches(self):
        original_session = main.session
        main.session = SessionState()
        set_command_history(CommandHistory(max_entries=20))
        buffer = io.StringIO()
        try:
            with patch("subprocess.Popen") as popen, \
                    patch("main.get_input",
                          return_value="open cmd /c whoami"):
                with redirect_stdout(buffer):
                    main._process_once()
            popen.assert_not_called()
            self.assertIn("not allowed", buffer.getvalue())
        finally:
            main.session = original_session
            set_command_history(None)

    def test_allowlisted_launch_uses_list_args_and_shell_false(self):
        """Positive control: the only launch mechanism is a list argv
        with shell=False — user input is never a shell string."""
        with patch(
            "core.application_manager._resolve_executable",
            return_value=r"C:\Windows\System32\cmd.exe",
        ), patch("core.application_manager.subprocess.Popen") as popen:
            result = launch_application("cmd")
        self.assertTrue(result["success"])
        popen.assert_called_once_with(
            [r"C:\Windows\System32\cmd.exe"], shell=False
        )

    @patch("core.file_manager.os.startfile", create=True)
    @patch("subprocess.Popen")
    def test_raw_path_input_never_launches_or_opens(
        self, mock_popen, mock_startfile
    ):
        original_session = main.session
        main.session = SessionState()
        set_command_history(CommandHistory(max_entries=20))
        buffer = io.StringIO()
        try:
            with patch("main.get_input",
                       return_value="open c:\\windows\\system32\\calc.exe"):
                with redirect_stdout(buffer):
                    main._process_once()
            mock_popen.assert_not_called()
            mock_startfile.assert_not_called()
            self.assertTrue(buffer.getvalue().strip())
        finally:
            main.session = original_session
            set_command_history(None)


# ---------------------------------------------------------------------------
# 6. Confirmation bypass + configuration safety + replay
# ---------------------------------------------------------------------------

class TestConfirmationBypassSafety(_SecurityTurnTestCase):
    """Configuration and input tricks can never bypass confirmation."""

    JUNK_TIMEOUT_VALUES = ["sixty", 0, -5, None, True, 3.5, "60"]

    def test_malformed_config_cannot_disable_confirmation(self):
        for junk in self.JUNK_TIMEOUT_VALUES:
            with self.subTest(junk=junk):
                main.session = SessionState()
                self.history = CommandHistory(max_entries=20)
                set_command_history(self.history)
                with patch(
                    "core.session_state.get_value", return_value=junk
                ), patch("main.execute_command") as mock_execute:
                    self._run_turn("shutdown computer")
                mock_execute.assert_not_called()
                status, _ = main.session.resolve_pending_confirmation()
                self.assertEqual(status, "active")
                # Timeout still falls back to the safe ~60s default —
                # a bad setting can never make confirmations immortal.
                remaining = (
                    main.session._pending_expires_at - time.monotonic()
                )
                self.assertGreater(remaining, 50)
                self.assertLessEqual(remaining, 70)

    def test_config_saying_false_cannot_skip_confirmation(self):
        """Even if safety.require_confirmation_for_* were read as
        False, the dangerous command must still park."""
        main.session = SessionState()

        def junk_get_value(*args, **kwargs):
            return False

        with patch("core.config_manager.get_value",
                   side_effect=junk_get_value), \
                patch("core.session_state.get_value",
                      side_effect=junk_get_value), \
                patch("main.execute_command") as mock_execute:
            output = self._run_turn("shutdown computer")
        mock_execute.assert_not_called()
        status, _ = main.session.resolve_pending_confirmation()
        self.assertEqual(status, "active")
        self.assertIn("Are you sure", output)

    def test_history_replay_attempts_execute_nothing(self):
        for phrase in (
            "repeat last command",
            "run command 3",
            "run the last command again",
        ):
            with self.subTest(phrase=phrase):
                main.session = SessionState()
                with patch("main.execute_command") as mock_execute:
                    output = self._run_turn(phrase)
                mock_execute.assert_not_called()
                self.assertTrue(output.strip())

    def test_history_display_performs_no_effects(self):
        with no_system_effects() as effects:
            with patch("main.execute_command"):
                self._run_turn("help")
            output = self._run_turn("history")
        self.assert_no_effects(effects)
        self.assertIn("1. help", output)

    def test_clear_history_performs_no_effects(self):
        with no_system_effects() as effects:
            self._run_turn("clear history")
        self.assert_no_effects(effects)
        self.assertEqual(self.history.count(), 0)


# ---------------------------------------------------------------------------
# 7. Dangerous multi-command chaining
# ---------------------------------------------------------------------------

class TestMultiCommandDangerousChaining(_SecurityTurnTestCase):
    """Chains stop at the dangerous command; it never runs inline."""

    def test_chain_stops_at_confirmation_boundary(self):
        with no_system_effects() as effects, patch(
            "core.multi_command_executor.execute_command"
        ) as mock_multi:
            mock_multi.return_value = {
                "success": True, "message": "URL opened.", "data": {},
            }
            output = self._run_turn(
                "open https://github.com and shutdown computer"
            )
        # First (safe) command executed, then the chain parked
        self.assertEqual(mock_multi.call_count, 1)
        self.assertEqual(
            mock_multi.call_args.args[0]["intent"], "open_url"
        )
        effects.shutdown.assert_not_called()
        status, pending = main.session.resolve_pending_confirmation()
        self.assertEqual(status, "active")
        self.assertEqual(pending.intent, "shutdown_computer")
        self.assertIn("Are you sure you want to shut down", output)

    def test_dangerous_command_first_stops_the_whole_chain(self):
        with no_system_effects() as effects, patch(
            "core.multi_command_executor.execute_command"
        ) as mock_multi:
            mock_multi.return_value = {
                "success": True, "message": "Chrome opened.", "data": {},
            }
            self._run_turn(
                "shutdown computer and open chrome"
            )
        # Nothing executed: the first command parked and the rest of
        # the chain was deliberately not resumed.
        mock_multi.assert_not_called()
        effects.shutdown.assert_not_called()
        effects.popen.assert_not_called()
        status, pending = main.session.resolve_pending_confirmation()
        self.assertEqual(status, "active")
        self.assertEqual(pending.intent, "shutdown_computer")

    def test_confirming_chain_parks_only_the_stored_command(self):
        """After 'yes', only the stored shutdown runs — the chain's
        remaining command is never resumed."""
        with patch("core.multi_command_executor.execute_command"):
            self._run_turn("shutdown computer and open chrome")
        with patch("main.execute_command") as mock_execute:
            mock_execute.return_value = {
                "success": True, "message": "Mocked shutdown.", "data": {},
            }
            with patch("core.multi_command_executor.execute_command") as mock_multi:
                self._run_turn("yes")
        mock_multi.assert_not_called()
        mock_execute.assert_called_once()
        self.assertEqual(
            mock_execute.call_args.args[0]["intent"], "shutdown_computer"
        )
        self.assertEqual(mock_execute.call_args.kwargs, {"confirmed": True})


if __name__ == "__main__":
    unittest.main()
