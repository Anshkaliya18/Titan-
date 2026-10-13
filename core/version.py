"""
Version Information (Step 23 — Phase 20)

One small, dependency-free module as the single source of truth for
TITAN's name and version. It is used by:

- the command-line flags (--version), see core/cli.py
- build metadata (TITAN.spec stamps this version into the Windows
  version resource of TITAN.exe, and build.py reports it)
- diagnostics (the session log records the version at startup)
- utils.paths (the writable data folder is named after APP_NAME)

Keep it simple: plain strings, no versioning framework.
"""

APP_NAME = "TITAN"
VERSION = "1.0.0"


def version_string() -> str:
    """Return the user-facing version string, e.g. "TITAN 0.1.0"."""
    return f"{APP_NAME} {VERSION}"
