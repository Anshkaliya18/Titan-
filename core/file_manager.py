"""
File Manager

Reusable, safe file and folder operations. This module receives
structured values (e.g. "report.pdf", "Projects") — it does NOT
parse natural language.

Architecture:
    executor
      -> file_manager
        -> safe path validation
          -> Windows filesystem

Safety rules:
- All operations are restricted to configured safe directories.
- Paths are resolved and checked with proper containment logic
  (never string-prefix checks).
- Deletion requires explicit confirmation and never recurses.
"""

import os
import sys
from pathlib import Path
from typing import Any

from core.config_manager import get_value
from utils import paths
from utils.logger import get_logger

logger = get_logger(__name__)

# Root Jarvis owns: the project directory in source mode, the writable
# per-user data root when packaged (Step 23). Anchored to this file —
# never to the current working directory.
PROJECT_ROOT = paths.data_root()

# Default limit for search results
DEFAULT_SEARCH_LIMIT = 20

# Windows FILE_ATTRIBUTE_REPARSE_POINT flag (junctions / symlinks / mounts)
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400


def _user_profile() -> Path:
    """Return the user profile directory without hardcoding a username."""
    profile = os.environ.get("USERPROFILE")
    if profile:
        return Path(profile)
    return Path.home()


def get_safe_directories() -> list[Path]:
    """
    Return the directories Jarvis is allowed to operate in.

    Uses Windows environment variables (USERPROFILE) — no hardcoded
    usernames. Directories that do not exist are still returned;
    callers handle missing directories gracefully.

    These are derived from the system, never from arbitrary user
    configuration, so a setting cannot widen the allowed locations.
    """
    profile = _user_profile()
    return [
        profile / "Desktop",
        profile / "Documents",
        profile / "Downloads",
        PROJECT_ROOT,
    ]


def _determine_default_workspace() -> Path:
    """Choose the default workspace: Desktop if present, else project root."""
    desktop = _user_profile() / "Desktop"
    if desktop.exists():
        return desktop
    return PROJECT_ROOT


def _workspace_subdir() -> str:
    """
    Return the configured workspace subdirectory name.

    Read from the Configuration Manager. Only a plain name is used —
    it is resolved under the safe base directory and can never escape
    it, because resolve_path() enforces containment.
    """
    name = get_value("filesystem", "workspace")
    if not isinstance(name, str) or not name.strip():
        return "jarvis_workspace"
    # Reject any path separators to keep this a single directory name
    name = name.strip()
    if "/" in name or "\\" in name or name in {".", ".."}:
        logger.warning(f"Invalid workspace name '{name}', using default")
        return "jarvis_workspace"
    return name


# Default location used when resolving relative targets. The
# configured workspace name is placed under the safe base directory.
DEFAULT_WORKSPACE = _determine_default_workspace() / _workspace_subdir()


def resolve_path(target: str) -> Path:
    """
    Resolve a target into an absolute, normalised path.

    Absolute targets are resolved directly. Relative targets are
    joined onto the default workspace. Path traversal sequences
    such as ".." are normalised by Path.resolve().
    """
    candidate = Path(target)
    if candidate.is_absolute():
        return candidate.resolve()
    return (DEFAULT_WORKSPACE / candidate).resolve()


def is_safe_path(path: str | Path) -> bool:
    """
    Check whether a path is inside one of the safe directories.

    Uses Path.resolve() + containment logic rather than string-prefix
    checks, so traversal (e.g. "Desktop/../../Windows") cannot escape.
    """
    try:
        resolved = Path(path).resolve()
    except (OSError, ValueError, RuntimeError):
        return False

    for safe_dir in get_safe_directories():
        try:
            if resolved.is_relative_to(safe_dir.resolve()):
                return True
        except (OSError, ValueError, RuntimeError):
            continue
    return False


def _is_reparse_or_link(path: Path) -> bool:
    """Check if a path is a symlink or Windows reparse point."""
    try:
        if path.is_symlink():
            return True
        attrs = getattr(os.lstat(path), "st_file_attributes", 0)
        return bool(attrs & _FILE_ATTRIBUTE_REPARSE_POINT)
    except OSError:
        # Inaccessible — treat as unsafe to skip
        return True


def _search_files(name: str, limit: int, exact: bool = False) -> list[Path]:
    """
    Search the safe directories for files matching a name.

    Args:
        name: Filename or substring to match (case-insensitive).
        limit: Maximum number of results to return.
        exact: If True, require a full filename match; otherwise
            match as a substring.

    Returns:
        List of matching file paths (may be empty).
    """
    matches: list[Path] = []
    needle = name.strip().lower()
    if not needle:
        return matches

    for safe_dir in get_safe_directories():
        if not safe_dir.is_dir():
            continue
        try:
            for root, dirs, files in os.walk(safe_dir, followlinks=False):
                # Skip symlinked / reparse-point directories, and walk in
                # sorted order so results are deterministic regardless of
                # the filesystem's native enumeration order.
                dirs[:] = sorted(
                    d for d in dirs if not _is_reparse_or_link(Path(root) / d)
                )
                for filename in sorted(files):
                    if len(matches) >= limit:
                        return matches
                    lower = filename.lower()
                    if (lower == needle) if exact else (needle in lower):
                        matches.append(Path(root) / filename)
        except OSError:
            # Inaccessible directory — skip it
            continue

    return matches


def _locate(target: str) -> Path | None:
    """
    Locate a target file within the safe directories.

    First checks the resolved path. For bare filenames (no directory
    component), falls back to an exact filename search of the safe
    directories.
    """
    resolved = resolve_path(target)
    if resolved.exists():
        return resolved

    # Only fall back for bare filenames, not explicit paths
    if Path(target).name != target:
        return None

    matches = _search_files(target, limit=1, exact=True)
    return matches[0] if matches else None


def create_folder(name: str) -> dict[str, Any]:
    """
    Create a folder inside the default workspace.

    Absolute paths and path traversal are rejected. An existing
    folder returns a clean success result instead of crashing.
    """
    logger.info(f"Creating folder: {name}")

    if not name or not name.strip():
        return _result(False, "No folder name provided.", {})

    candidate = Path(name)
    if candidate.is_absolute():
        logger.warning(f"Blocked unsafe file path: {name}")
        return _result(
            False,
            "Absolute paths are not allowed. Provide a folder name.",
            {},
        )

    folder_path = (DEFAULT_WORKSPACE / candidate).resolve()

    # Folders may only be created inside the default workspace — a
    # relative name that climbs out of it (e.g. "../Documents/x") is
    # rejected even when the destination is another safe directory.
    workspace = DEFAULT_WORKSPACE.resolve()
    if folder_path != workspace and not folder_path.is_relative_to(workspace):
        logger.warning(f"Blocked unsafe file path: {name}")
        return _result(
            False,
            "Access denied: folders can only be created in the workspace.",
            {},
        )

    if not is_safe_path(folder_path):
        logger.warning(f"Blocked unsafe file path: {name}")
        return _result(
            False,
            "Access denied: path is outside Jarvis safe directories.",
            {},
        )

    if folder_path.exists():
        if folder_path.is_dir():
            logger.info(f"Folder already exists: {folder_path}")
            return _result(
                True,
                f"Folder '{name}' already exists.",
                {"path": str(folder_path)},
            )
        return _result(
            False,
            f"'{name}' already exists and is not a folder.",
            {},
        )

    try:
        folder_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created folder: {folder_path}")
        return _result(
            True,
            f"Folder '{name}' created successfully.",
            {"path": str(folder_path)},
        )
    except OSError:
        # Traceback to the log; the user message stays generic.
        logger.exception("Failed to create folder")
        return _result(False, "Failed to create the folder.", {})


def open_file(path: str) -> dict[str, Any]:
    """
    Open a file or folder inside the safe directories with its
    associated Windows application.

    Uses os.startfile() — never os.system() or shell=True.
    """
    logger.info(f"Opening file: {path}")

    if not path or not path.strip():
        return _result(False, "No file path provided.", {})

    resolved = resolve_path(path)
    if not is_safe_path(resolved):
        logger.warning(f"Blocked unsafe file path: {path}")
        return _result(
            False,
            "Access denied: path is outside Jarvis safe directories.",
            {},
        )

    located = _locate(path)
    if located is None or not located.exists():
        return _result(False, f"File not found: {path}", {})

    if not (located.is_file() or located.is_dir()):
        return _result(False, f"Not a file or folder: {path}", {})

    if not is_safe_path(located):
        logger.warning(f"Blocked unsafe file path: {path}")
        return _result(
            False,
            "Access denied: path is outside Jarvis safe directories.",
            {},
        )

    if sys.platform != "win32":
        return _result(False, "This command requires Windows.", {})

    try:
        os.startfile(str(located))  # Windows-only shell association
        logger.info(f"Opened file: {located}")
        kind = "Folder" if located.is_dir() else "File"
        return _result(True, f"{kind} opened: {path}", {"path": str(located)})
    except OSError:
        logger.exception("Failed to open file")
        return _result(False, "Failed to open the file.", {})


def delete_file(path: str, confirmed: bool = False) -> dict[str, Any]:
    """
    Delete a single file inside the safe directories.

    Requires explicit confirmation. Directories and recursive
    deletion are never supported.

    Args:
        path: File name or path to delete.
        confirmed: Must be True before any deletion occurs.
    """
    logger.info(f"Deleting file: {path}")

    if not path or not path.strip():
        return _result(False, "No file path provided.", {})

    if not confirmed:
        return _result(
            False,
            f"Confirmation required before deleting '{path}'.",
            {},
        )

    resolved = resolve_path(path)
    if not is_safe_path(resolved):
        logger.warning(f"Blocked unsafe file path: {path}")
        return _result(
            False,
            "Access denied: path is outside Jarvis safe directories.",
            {},
        )

    located = _locate(path)
    if located is None or not located.exists():
        return _result(False, f"File not found: {path}", {})

    if not is_safe_path(located):
        logger.warning(f"Blocked unsafe file path: {path}")
        return _result(
            False,
            "Access denied: path is outside Jarvis safe directories.",
            {},
        )

    if located.is_dir():
        return _result(
            False,
            f"'{path}' is a folder. Directory deletion is not allowed.",
            {},
        )

    try:
        located.unlink()
        logger.info(f"Deleted file: {located}")
        return _result(
            True,
            "File deleted successfully.",
            {"path": str(located)},
        )
    except OSError:
        logger.exception("Failed to delete file")
        return _result(False, "Failed to delete the file.", {})


def search_file(name: str, limit: int = DEFAULT_SEARCH_LIMIT) -> dict[str, Any]:
    """
    Search the safe directories for files matching a name.

    Matching is case-insensitive. Results are limited to a
    reasonable number (default 20).
    """
    logger.info(f"Searching for file: {name}")

    if not name or not name.strip():
        return _result(False, "No search term provided.", {})

    matches = _search_files(name, limit=limit)

    if not matches:
        logger.info(f"No matches for {name}")
        return _result(False, f"No files found matching '{name}'", {})

    files = [str(p) for p in matches]
    logger.info(f"Found {len(files)} matches for {name}")
    suffix = "" if len(files) == 1 else "s"
    return _result(
        True,
        f"Found {len(files)} matching file{suffix}.",
        {"files": files},
    )


def _result(success: bool, message: str, data: dict[str, Any]) -> dict:
    """Create a consistent result structure.

    Only failures are logged here — individual functions emit their
    own INFO logs on success.
    """
    if not success:
        logger.warning(f"File operation failed: {message}")
    return {
        "success": success,
        "message": message,
        "data": data,
    }
