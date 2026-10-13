"""Bounded, read-only Windows installed application discovery.

Only Start Menu shortcuts and uninstall registry display metadata are read.
Discovered targets are revalidated before every launch; this module never
executes a file. AppX/Store activation is intentionally not supported.
"""
from __future__ import annotations

import os
import re
import sys
from itertools import chain
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

from utils.logger import get_logger

logger = get_logger(__name__)
MAX_INDEX_ENTRIES = 1000
MAX_SHORTCUTS = 2500
BLOCKED_EXE = {"cmd.exe", "powershell.exe", "pwsh.exe", "wscript.exe", "cscript.exe",
               "mshta.exe", "rundll32.exe", "regsvr32.exe", "schtasks.exe",
               "wmic.exe", "msiexec.exe"}
ALIASES = {
    "vscode": "visual studio code", "vs code": "visual studio code",
    "google chrome": "chrome", "microsoft edge": "edge",
    "calc": "calculator", "file explorer": "explorer",
}


def normalize(value: str) -> str:
    return " ".join(re.sub(r"[^\w\s.-]", " ", value.casefold()).split())


@dataclass(frozen=True)
class DiscoveredApplication:
    name: str
    normalized_name: str
    executable: str
    source: str
    aliases: tuple[str, ...] = ()
    confidence: float = 1.0
    shortcut_path: str | None = None


@dataclass(frozen=True)
class Resolution:
    status: str  # MATCH, AMBIGUOUS, UNKNOWN
    query: str
    application: DiscoveredApplication | None = None
    candidates: tuple[DiscoveredApplication, ...] = ()
    confidence: float = 0.0


def _allowed_roots() -> tuple[Path, ...]:
    env_names = ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA", "APPDATA")
    roots = []
    for key in env_names:
        value = os.environ.get(key)
        if value:
            try:
                roots.append(Path(value).resolve())
            except (OSError, RuntimeError):
                pass
    return tuple(roots)


def validate_shortcut(value: str | Path | None, expected_executable: str | Path | None = None) -> Path | None:
    """Validate a cached shortcut remains inside a standard Start Menu tree.

    Re-resolve the shortcut at launch time and require its target to match the
    already validated executable identity. Shortcut arguments stay owned by
    Windows' registered shortcut and are never composed from user input.
    """
    if sys.platform != "win32" or not value:
        return None
    try:
        shortcut = Path(value).resolve(strict=True)
        if not shortcut.is_file() or shortcut.suffix.casefold() != ".lnk":
            return None
        bases = []
        for key in ("APPDATA", "PROGRAMDATA"):
            root = os.environ.get(key)
            if root:
                bases.append((Path(root) / "Microsoft" / "Windows" / "Start Menu" / "Programs").resolve())
        if not any(shortcut != base and base in shortcut.parents for base in bases):
            return None
        target = validate_executable(_shortcut_target(shortcut))
        expected = validate_executable(expected_executable)
        if target is None or expected is None or target != expected:
            return None
        return shortcut
    except (OSError, RuntimeError, ValueError):
        return None


def validate_executable(value: str | Path | None) -> Path | None:
    """Accept only an existing .exe inside normal per-user/program roots."""
    if sys.platform != "win32" or not value:
        return None
    raw = str(value).strip().strip('"')
    if not raw or any(ch in raw for ch in ('\0', '\r', '\n', '|', '&', ';', '<', '>')):
        return None
    if "://" in raw or raw.casefold().startswith(("file:", "shell:")):
        return None
    try:
        path = Path(raw).resolve(strict=True)
        if not path.is_file() or path.suffix.casefold() != ".exe":
            return None
        if path.name.casefold() in BLOCKED_EXE:
            return None
        roots = _allowed_roots()
        if not any(path == root or root in path.parents for root in roots):
            return None
        return path
    except (OSError, RuntimeError, ValueError):
        return None


def _shortcut_target(path: Path) -> str | None:
    """Resolve .lnk through the Windows Shell COM automation interface.

    pywin32 exposes the late-bound IDispatch returned by WScript.Shell as
    Python properties. comtypes' raw IDispatch pointer does not, which was
    causing valid shortcuts to be silently discarded.
    """
    try:
        from win32com.client import Dispatch
        shortcut = Dispatch("WScript.Shell").CreateShortcut(str(path))
        return shortcut.TargetPath
    except Exception:
        return None


def _is_installer_metadata(name: str, executable: Path) -> bool:
    """Reject common uninstall/setup payloads misreported as app icons."""
    stem = executable.stem.casefold().replace(" ", "")
    label = name.casefold()
    if (stem.startswith(("unins", "uninstall", "setup", "installer", "install", "maintenance"))
            or any(token in stem for token in ("uninstaller", "uninstall", "setup", "installer"))
            or re.search(r"\b(uninstall|uninstaller|setup|installer|repair|modify)\b", label)):
        return True
    parts = {part.casefold().replace("_", " ").replace("-", " ") for part in executable.parts}
    if any("package cache" in part or "uninstall" in part for part in parts):
        return True
    # These entries are Windows/runtime dispatchers, not the user-facing app
    # binary; the Start Menu shortcut, when present, supplies the real target.
    if stem in {"officeclicktorun", "git-lfs", "tesseract-uninstall"}:
        return True
    return False


def _start_menu_entries() -> Iterable[tuple[str, str, str, str | None]]:
    appdata = os.environ.get("APPDATA")
    program_data = os.environ.get("PROGRAMDATA")
    bases = [Path(x) / "Microsoft" / "Windows" / "Start Menu" / "Programs"
             for x in (appdata, program_data) if x]
    count = 0
    for base in bases:
        try:
            # Traversal remains confined to the two standard Programs trees.
            for shortcut in base.rglob("*.lnk"):
                count += 1
                if count > MAX_SHORTCUTS:
                    return
                target = _shortcut_target(shortcut)
                exe = validate_executable(target)
                if exe and not _is_installer_metadata(shortcut.stem, exe):
                    yield shortcut.stem, str(exe), "start_menu", str(shortcut)
        except (OSError, PermissionError):
            continue


def _registry_entries() -> Iterable[tuple[str, str, str, str | None]]:
    if sys.platform != "win32":
        return
    try:
        import winreg
    except ImportError:
        return
    locations = (
        (winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Uninstall"),
        (winreg.HKEY_LOCAL_MACHINE, r"Software\Microsoft\Windows\CurrentVersion\Uninstall"),
    )
    views = [getattr(winreg, "KEY_WOW64_64KEY", 0), getattr(winreg, "KEY_WOW64_32KEY", 0)]
    seen = set()
    for hive, keypath in locations:
        for view in views:
            try:
                root = winreg.OpenKey(hive, keypath, 0, winreg.KEY_READ | view)
            except OSError:
                continue
            try:
                sub_count = winreg.QueryInfoKey(root)[0]
                for i in range(min(sub_count, 3000)):
                    try:
                        sub = winreg.OpenKey(root, winreg.EnumKey(root, i), 0, winreg.KEY_READ | view)
                        with sub:
                            display, _ = winreg.QueryValueEx(sub, "DisplayName")
                            icon, _ = winreg.QueryValueEx(sub, "DisplayIcon")
                        if not isinstance(display, str) or not isinstance(icon, str):
                            continue
                        match = re.match(r'^\s*"?(.+?\.exe)"?(?:\s*,\s*-?\d+)?\s*$', icon, re.I)
                        exe = validate_executable(match.group(1) if match else None)
                        if exe and not _is_installer_metadata(display, exe) and (normalize(display), str(exe).casefold()) not in seen:
                            seen.add((normalize(display), str(exe).casefold()))
                            yield display.strip(), str(exe), "uninstall_registry", None
                    except (OSError, ValueError, TypeError):
                        continue
            finally:
                winreg.CloseKey(root)


class ApplicationDiscovery:
    """Lazy bounded cache of installed app names and validated targets."""
    def __init__(self, *, discovery_enabled: bool = True, cache_enabled: bool = True, fuzzy_enabled: bool = True,
                 fuzzy_threshold: float = 0.90, max_results: int = 25):
        self.discovery_enabled = discovery_enabled if type(discovery_enabled) is bool else True
        self.cache_enabled = cache_enabled if type(cache_enabled) is bool else True
        self.fuzzy_enabled = fuzzy_enabled if type(fuzzy_enabled) is bool else False
        self.fuzzy_threshold = float(fuzzy_threshold) if isinstance(fuzzy_threshold, (int, float)) and not isinstance(fuzzy_threshold, bool) and 0.9 <= fuzzy_threshold <= 1 else 0.90
        self.max_results = max(1, min(max_results if type(max_results) is int else 25, 100))
        self._apps: tuple[DiscoveredApplication, ...] | None = None

    def refresh(self) -> tuple[DiscoveredApplication, ...]:
        logger.info("Application discovery started")
        found: dict[tuple[str, str], DiscoveredApplication] = {}
        if self.discovery_enabled and sys.platform == "win32":
            for name, executable, source, shortcut_path in chain(_start_menu_entries(), _registry_entries()):
                path = validate_executable(executable)
                normalized = normalize(name)
                if not path or not normalized:
                    continue
                key = (normalized, str(path).casefold())
                aliases = tuple(sorted(alias for alias, canonical in ALIASES.items()
                                       if normalize(canonical) == normalized))
                found.setdefault(key, DiscoveredApplication(
                    name.strip(), normalized, str(path), source, aliases,
                    shortcut_path=shortcut_path,
                ))
                if len(found) >= MAX_INDEX_ENTRIES:
                    break
        apps = tuple(sorted(found.values(), key=lambda item: (item.normalized_name, item.executable.casefold())))
        self._apps = apps if self.cache_enabled else None
        logger.info("Application discovery completed: %d entries", len(apps))
        return apps

    def get_all(self) -> tuple[DiscoveredApplication, ...]:
        return self._apps if self._apps is not None else self.refresh()

    def find_candidates(self, query: str) -> tuple[DiscoveredApplication, ...]:
        return self.get_all()[:self.max_results] if not normalize(query) else tuple(
            app for app in self.get_all() if normalize(query) in app.normalized_name
        )[:self.max_results]

    def resolve(self, query: str) -> Resolution:
        q = normalize(query)
        if (not q or len(q) > 100
                or re.match(r"^[A-Za-z]:[\\/]|^\\\\|^(?:https?|file)://", query.strip(), re.I)):
            return Resolution("UNKNOWN", query)
        if set(q.split()) & {"open", "opens", "launch", "launches", "start", "starts", "close", "then", "and"}:
            return Resolution("UNKNOWN", query)
        apps = self.get_all()
        alias = normalize(ALIASES.get(q, ""))
        exact = [a for a in apps if a.normalized_name == q or (alias and a.normalized_name == alias)]
        # A short exact product name can also be the prefix of a second
        # product (for example Visual Studio / Visual Studio Code). Ask
        # instead of silently choosing the shorter one.
        longer = [a for a in apps if a.normalized_name.startswith(q + " ")]
        if exact and longer:
            unique = {a.executable.casefold(): a for a in (*exact, *longer)}
            return Resolution("AMBIGUOUS", query, candidates=tuple(unique.values())[:self.max_results])
        if exact:
            unique = {a.executable.casefold(): a for a in exact}
            if len(unique) != 1:
                return Resolution("AMBIGUOUS", query, candidates=tuple(unique.values()))
            return Resolution("MATCH", query, next(iter(unique.values())), confidence=1.0 if not alias else .95)
        phrase = [a for a in apps if re.search(
            r"(?:^|\s)" + re.escape(q) + r"(?:$|\s)", a.normalized_name
        )]
        if len(phrase) == 1:
            return Resolution("MATCH", query, phrase[0], confidence=.92)
        if len(phrase) > 1:
            return Resolution("AMBIGUOUS", query, candidates=tuple(phrase[:self.max_results]))
        if self.fuzzy_enabled:
            scored = sorted(((SequenceMatcher(None, q, a.normalized_name).ratio(), a) for a in apps), key=lambda x: (-x[0], x[1].normalized_name))
            if scored and scored[0][0] >= self.fuzzy_threshold:
                close = tuple(a for score, a in scored if score >= self.fuzzy_threshold and scored[0][0] - score <= .02)
                if len({a.executable.casefold() for a in close}) > 1:
                    return Resolution("AMBIGUOUS", query, candidates=close[:self.max_results])
                return Resolution("MATCH", query, scored[0][1], confidence=scored[0][0])
        return Resolution("UNKNOWN", query)


_default: ApplicationDiscovery | None = None


def get_discovery() -> ApplicationDiscovery:
    global _default
    if _default is None:
        try:
            from core.config_manager import get_section
            settings = get_section("applications")
        except Exception:
            settings = {}
        _default = ApplicationDiscovery(
            discovery_enabled=settings.get("discovery_enabled", True),
            cache_enabled=settings.get("cache_enabled", True),
            fuzzy_enabled=settings.get("fuzzy_matching_enabled", True),
            fuzzy_threshold=settings.get("fuzzy_threshold", .90),
            max_results=settings.get("max_results", 25),
        )
    return _default


def refresh_application_index() -> tuple[DiscoveredApplication, ...]:
    return get_discovery().refresh()
