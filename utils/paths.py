"""
Filesystem Paths (Step 23 — packaging / resource handling)

Single source of truth for where TITAN reads and writes files. Two
kinds of location exist:

Read-only application resources — bundle_dir()
    Development : the project directory (TITAN/)
    Packaged    : the PyInstaller bundle (sys._MEIPASS), i.e. the
                  _internal/ folder next to TITAN.exe
    Never written at runtime.

Writable user data — data_root()
    Development : the project directory, which keeps the original
                  layout (config/ and logs/ stay in the source tree)
    Packaged    : %APPDATA%\\TITAN — per-user, no administrator rights
    Override    : the TITAN_DATA_DIR environment variable redirects
                  the data root in either mode (used by tests and by
                  users who want a relocatable data folder)

Derived paths (all absolute, never relative to the working directory):
    settings_path()         data_root()/config/settings.json
    log_dir()               data_root()/logs
    default_settings_path() bundle_dir()/config/settings.json  (template)

Rules
-----
- Nothing reads the current working directory: every path is anchored
  to this file, to sys, or to a documented environment variable. TITAN
  therefore behaves identically when launched from the project folder,
  another terminal directory, a shortcut, the Startup folder, Explorer,
  or Task Scheduler.
- No developer-specific absolute path is ever hardcoded, and user input
  never influences any path returned here — the functions compute from
  the installation layout only.
- This module only computes paths. It executes nothing and reads no
  configuration values.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from core.version import APP_NAME

# Environment variable that relocates the writable data root.
DATA_DIR_ENV_VAR = "TITAN_DATA_DIR"
LEGACY_DATA_DIR_ENV_VAR = "JARVIS_DATA_DIR"

# Source tree root (TITAN/), fixed once from this file's location.
_SOURCE_ROOT = Path(__file__).resolve().parent.parent


def is_frozen() -> bool:
    """Return True when running from a packaged (PyInstaller) build."""
    return bool(getattr(sys, "frozen", False))


def bundle_dir() -> Path:
    """
    Return the read-only application resource directory.

    Development: the project directory (config/, core/ spell model, ...).
    Packaged: the bundle directory (sys._MEIPASS) that holds the data
    files collected by TITAN.spec.
    """
    if is_frozen():
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
        # Defensive fallback: next to the executable.
        return Path(sys.executable).resolve().parent
    return _SOURCE_ROOT


def data_root() -> Path:
    """
    Return the writable data root for this user.

    Order: TITAN_DATA_DIR override, then %APPDATA%\\TITAN when
    packaged, otherwise the project directory (source mode keeps the
    historical in-project layout).
    """
    override = (
        os.environ.get(DATA_DIR_ENV_VAR, "").strip()
        or os.environ.get(LEGACY_DATA_DIR_ENV_VAR, "").strip()
    )
    if override:
        return Path(override).expanduser()

    if is_frozen():
        roaming = os.environ.get("APPDATA")
        if roaming:
            titan_root = Path(roaming) / APP_NAME
            legacy_root = Path(roaming) / "Jarvis"
            return legacy_root if not titan_root.exists() and legacy_root.exists() else titan_root
        # Still a stable per-user location if APPDATA is unavailable.
        titan_root = Path.home() / "AppData" / "Roaming" / APP_NAME
        legacy_root = Path.home() / "AppData" / "Roaming" / "Jarvis"
        return legacy_root if not titan_root.exists() and legacy_root.exists() else titan_root

    return _SOURCE_ROOT


def settings_path() -> Path:
    """Return the writable settings.json location (never cwd-based)."""
    return data_root() / "config" / "settings.json"


def default_settings_path() -> Path:
    """
    Return the read-only default settings template.

    In source mode the project's own settings file doubles as the
    template; when packaged, the copy bundled by TITAN.spec is used
    to seed the user's configuration on first launch.
    """
    if is_frozen():
        return bundle_dir() / "config" / "settings.json"
    return _SOURCE_ROOT / "config" / "settings.json"


def log_dir() -> Path:
    """Return the writable log directory (never cwd-based)."""
    return data_root() / "logs"
