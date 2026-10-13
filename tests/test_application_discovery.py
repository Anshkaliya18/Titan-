"""Application discovery resolution and safety tests (no real apps launch)."""
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.application_discovery import (
    ApplicationDiscovery, DiscoveredApplication, validate_executable,
)


def app(name, exe=None, source="start_menu"):
    return DiscoveredApplication(name, name.casefold(), exe or fr"C:\Program Files\{name}\{name}.exe", source)


def index(*apps):
    discovery = ApplicationDiscovery()
    discovery._apps = tuple(apps)
    return discovery


def test_resolution_exact_case_and_whitespace():
    discovery = index(app("Discord"))
    assert discovery.resolve("  DISCORD ").application.name == "Discord"


def test_alias_resolution_vscode():
    assert index(app("Visual Studio Code")).resolve("vscode").status == "MATCH"


def test_ambiguous_short_phrase_does_not_pick():
    result = index(app("Visual Studio"), app("Visual Studio Code")).resolve("visual studio")
    assert result.status == "AMBIGUOUS"
    assert len(result.candidates) == 2
    assert result.application is None


def test_unknown_and_explicit_path_are_rejected():
    discovery = index(app("Discord"))
    assert discovery.resolve("photoshop").status == "UNKNOWN"
    assert discovery.resolve(r"C:\malicious.exe").status == "UNKNOWN"


def test_fuzzy_match_requires_threshold():
    discovery = index(app("Application"))
    assert discovery.resolve("applicationx").status == "MATCH"
    strict = ApplicationDiscovery(fuzzy_threshold=.99)
    strict._apps = discovery._apps
    assert strict.resolve("applicationx").status == "UNKNOWN"


def test_fuzzy_tie_is_ambiguous():
    result = index(app("Application", r"C:\Program Files\A\a.exe"),
                   app("Application", r"C:\Program Files\B\b.exe")).resolve("application")
    assert result.status == "AMBIGUOUS"


def test_result_count_is_hard_bounded():
    discovery = ApplicationDiscovery(max_results=10000)
    discovery._apps = tuple(app(f"App {n:03}") for n in range(200))
    assert len(discovery.get_all()) == 200
    assert len(discovery.find_candidates("app")) == 100
    assert discovery.max_results == 100


def test_validate_executable_rejects_non_windows_and_script():
    with patch("core.application_discovery.sys.platform", "linux"):
        assert validate_executable(r"C:\Program Files\Thing\thing.exe") is None
    assert validate_executable(r"C:\Program Files\Thing\thing.py") is None


def test_validate_executable_checks_existing_file_and_allowed_root(tmp_path):
    root = tmp_path / "Programs"
    root.mkdir()
    executable = root / "app.exe"
    executable.write_bytes(b"mock")
    outside = tmp_path / "outside.exe"
    outside.write_bytes(b"mock")
    with patch("core.application_discovery.sys.platform", "win32"), patch.dict(
        "os.environ", {"PROGRAMFILES": str(root), "PROGRAMFILES(X86)": "",
                       "LOCALAPPDATA": "", "APPDATA": ""}, clear=False
    ):
        assert validate_executable(str(executable)) == executable.resolve()
        assert validate_executable(str(outside)) is None
        assert validate_executable(str(root / "missing.exe")) is None
        assert validate_executable(str(root / "app.bat")) is None
        assert validate_executable(str(root / "cmd.exe")) is None
        assert validate_executable(str(executable) + " & whoami") is None
        assert validate_executable("https://example.com/app.exe") is None


def test_find_and_refresh_phrases_are_recognized():
    from core.natural_language import NaturalLanguageInterpreter
    interpreter = NaturalLanguageInterpreter()
    found = interpreter.interpret("find discord")
    assert found.success and found.intent == "FIND_APPLICATION" and found.target == "discord"
    listed = interpreter.interpret("find installed apps")
    assert listed.success and listed.intent == "FIND_APPLICATION" and listed.target is None
    refreshed = interpreter.interpret("refresh applications")
    assert refreshed.success and refreshed.intent == "REFRESH_APPLICATIONS"


def test_shortcut_resolution_uses_windows_dispatch_and_ignores_arguments():
    import sys
    import types
    from core.application_discovery import _shortcut_target
    shortcut = MagicMock()
    shortcut.TargetPath = r"C:\Program Files\Example\example.exe"
    # Arguments deliberately contain data; discovery must read only TargetPath.
    shortcut.Arguments = MagicMock(return_value="--unexpected")
    dispatch = MagicMock()
    dispatch.CreateShortcut.return_value = shortcut
    module = types.ModuleType("win32com.client")
    module.Dispatch = MagicMock(return_value=dispatch)
    package = types.ModuleType("win32com")
    package.client = module
    with patch.dict(sys.modules, {"win32com": package, "win32com.client": module}):
        assert _shortcut_target(Path("trusted.lnk")) == shortcut.TargetPath
    assert not shortcut.Arguments.called


def test_discovered_shortcut_launch_uses_only_revalidated_start_menu_link():
    from core.application_manager import Application, launch_application
    app = Application(
        name="Example", executable_paths=[r"C:\Program Files\Example\example.exe"],
        executable="example.exe", description="Discovered installed application",
        shortcut_path=r"C:\Users\User\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Example.lnk",
    )
    with patch("core.application_manager.sys.platform", "win32"), \
         patch("core.application_manager.get_application", return_value=app), \
         patch("core.application_manager._resolve_executable", return_value=app.executable_paths[0]), \
         patch("core.application_discovery.validate_shortcut", return_value=Path(app.shortcut_path)), \
         patch("core.application_manager.os.startfile", create=True) as startfile, \
         patch("core.application_manager.subprocess.Popen") as popen:
        result = launch_application("Example")
    assert result["success"]
    startfile.assert_called_once_with(app.shortcut_path)
    popen.assert_not_called()


def test_shortcut_validator_requires_standard_menu_root_and_unchanged_target(tmp_path):
    from core.application_discovery import validate_shortcut
    app_root = tmp_path / "Programs"
    app_root.mkdir()
    exe = app_root / "example.exe"
    exe.write_bytes(b"test")
    menu = tmp_path / "Roaming" / "Microsoft" / "Windows" / "Start Menu" / "Programs"
    menu.mkdir(parents=True)
    link = menu / "Example.lnk"
    link.write_bytes(b"mock shortcut")
    outside = tmp_path / "untrusted.lnk"
    outside.write_bytes(b"mock shortcut")
    with patch("core.application_discovery.sys.platform", "win32"), patch.dict(
        "os.environ", {"APPDATA": str(tmp_path / "Roaming"),
                       "PROGRAMDATA": "", "PROGRAMFILES": str(app_root),
                       "PROGRAMFILES(X86)": "", "LOCALAPPDATA": ""}, clear=False
    ), patch("core.application_discovery._shortcut_target", return_value=str(exe)):
        assert validate_shortcut(str(link), str(exe)) == link.resolve()
        assert validate_shortcut(str(outside), str(exe)) is None
        assert validate_shortcut(str(link), str(app_root / "different.exe")) is None


def test_cache_refresh_and_empty_non_windows():
    discovery = ApplicationDiscovery()
    with patch("core.application_discovery.sys.platform", "linux"):
        assert discovery.get_all() == ()
        assert discovery.refresh() == ()


def test_duplicate_display_name_paths_remain_ambiguous():
    result = index(app("Editor", r"C:\Program Files\A\a.exe"),
                   app("Editor", r"C:\Program Files\B\b.exe")).resolve("editor")
    assert result.status == "AMBIGUOUS"

