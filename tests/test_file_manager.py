"""
Tests for the file manager module.

All tests run against temporary directories — the user's real
Desktop/Documents/Downloads are never modified. os.startfile is
mocked so no files are actually opened during tests.
"""

import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

# Add parent directory to path so we can import core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.file_manager import (
    DEFAULT_SEARCH_LIMIT,
    create_folder,
    delete_file,
    get_safe_directories,
    is_safe_path,
    open_file,
    resolve_path,
    search_file,
)


@contextmanager
def temp_workspace():
    """
    Create an isolated workspace:

        tmp/
          safe/            <- safe directory AND default workspace
            test.txt
            resume.pdf
            notes.txt
            resume_old.pdf
          outside.txt      <- outside the safe directory

    Patches get_safe_directories and DEFAULT_WORKSPACE so no real
    user directories are touched.
    """
    with tempfile.TemporaryDirectory() as tmp:
        safe = Path(tmp) / "safe"
        safe.mkdir()
        (safe / "test.txt").write_text("hello", encoding="utf-8")
        (safe / "resume.pdf").write_bytes(b"%PDF")
        (safe / "notes.txt").write_text("notes", encoding="utf-8")
        (safe / "resume_old.pdf").write_bytes(b"%PDF")
        outside = Path(tmp) / "outside.txt"
        outside.write_text("outside", encoding="utf-8")

        with patch(
            "core.file_manager.get_safe_directories", return_value=[safe]
        ), patch("core.file_manager.DEFAULT_WORKSPACE", safe):
            yield safe, outside


class TestSafeDirectories(unittest.TestCase):
    """Test the safe directory configuration."""

    def test_returns_expected_directories(self):
        """Should include Desktop, Documents, Downloads, Jarvis workspace."""
        dirs = get_safe_directories()
        names = [d.name for d in dirs]
        self.assertIn("Desktop", names)
        self.assertIn("Documents", names)
        self.assertIn("Downloads", names)

    def test_no_hardcoded_username(self):
        """Directories should derive from USERPROFILE / home, not a literal."""
        profile = os.environ.get("USERPROFILE") or str(Path.home())
        dirs = get_safe_directories()
        # Every directory should live under the user profile or project root
        project_root = Path(__file__).resolve().parent.parent
        for d in dirs:
            resolved = d.resolve()
            under_profile = str(resolved).startswith(str(Path(profile).resolve()))
            under_project = resolved == project_root.resolve()
            self.assertTrue(
                under_profile or under_project,
                f"{d} is not under the user profile or project root",
            )

    def test_missing_directories_are_graceful(self):
        """Non-existent directories should not raise."""
        with patch(
            "core.file_manager._user_profile",
            return_value=Path("C:\\definitely_not_a_real_user_dir_xyz"),
        ):
            dirs = get_safe_directories()
            self.assertEqual(len(dirs), 4)
            # Safety checks with missing dirs should not crash:
            # a path under a configured safe dir is still allowed,
            # while Windows system paths remain blocked.
            self.assertTrue(
                is_safe_path(
                    "C:\\definitely_not_a_real_user_dir_xyz\\Desktop\\file.txt"
                )
            )
            self.assertFalse(is_safe_path("C:\\Windows\\System32"))


class TestSafePath(unittest.TestCase):
    """Test path validation and containment."""

    def test_safe_file_returns_true(self):
        """A file inside a safe directory should be safe."""
        with temp_workspace() as (safe, _):
            self.assertTrue(is_safe_path(safe / "test.txt"))

    def test_outside_file_returns_false(self):
        """A file outside every safe directory should not be safe."""
        with temp_workspace() as (_, outside):
            self.assertFalse(is_safe_path(outside))

    def test_path_traversal_cannot_escape(self):
        """Desktop/../outside must not be considered safe."""
        with temp_workspace() as (safe, outside):
            traversal = safe / ".." / "outside.txt"
            self.assertFalse(is_safe_path(traversal))
            # The resolved target really is the outside file
            self.assertEqual(traversal.resolve(), outside.resolve())

    def test_deep_traversal_blocked(self):
        """Many levels of '..' must not escape the safe directory."""
        with temp_workspace() as (safe, _):
            self.assertFalse(is_safe_path(safe / ".." / ".." / ".." / "Windows"))

    def test_windows_system32_is_unsafe(self):
        """C:\\Windows\\System32 must never be a safe path."""
        with temp_workspace() as _:
            self.assertFalse(is_safe_path("C:\\Windows\\System32"))

    def test_sibling_prefix_not_matched(self):
        """A directory that merely shares a prefix must not match.

        E.g. Desktop2 should not be considered inside Desktop.
        """
        with tempfile.TemporaryDirectory() as tmp:
            safe = Path(tmp) / "Desktop"
            safe.mkdir()
            sibling = Path(tmp) / "Desktop2"
            sibling.mkdir()
            with patch(
                "core.file_manager.get_safe_directories", return_value=[safe]
            ):
                self.assertTrue(is_safe_path(safe / "file.txt"))
                self.assertFalse(is_safe_path(sibling / "file.txt"))


class TestResolvePath(unittest.TestCase):
    """Test target resolution."""

    def test_relative_target_joins_workspace(self):
        """Relative targets should resolve inside the default workspace."""
        with temp_workspace() as (safe, _):
            self.assertEqual(resolve_path("test.txt"), (safe / "test.txt").resolve())

    def test_absolute_target_resolved_directly(self):
        """Absolute targets should resolve as-is."""
        with temp_workspace() as (_, outside):
            self.assertEqual(resolve_path(str(outside)), outside.resolve())

    def test_traversal_is_normalised(self):
        """'..' sequences should be normalised by resolve()."""
        with temp_workspace() as (safe, _):
            resolved = resolve_path("sub/../test.txt")
            self.assertEqual(resolved, (safe / "test.txt").resolve())


class TestCreateFolder(unittest.TestCase):
    """Test folder creation."""

    def test_create_folder_success(self):
        """A simple name should create a folder in the workspace."""
        with temp_workspace() as (safe, _):
            result = create_folder("Projects")
            self.assertTrue(result["success"])
            self.assertTrue((safe / "Projects").is_dir())
            self.assertEqual(result["data"]["path"], str((safe / "Projects").resolve()))
            self.assertIn("created successfully", result["message"])

    def test_create_folder_existing_is_clean(self):
        """Creating an existing folder should return a clean result."""
        with temp_workspace() as (safe, _):
            (safe / "Existing").mkdir()
            result = create_folder("Existing")
            self.assertTrue(result["success"])
            self.assertIn("already exists", result["message"])
            self.assertTrue((safe / "Existing").is_dir())

    def test_create_folder_rejects_absolute_path(self):
        """Absolute paths such as C:\\Windows\\System32 must fail."""
        with temp_workspace() as _:
            result = create_folder("C:\\Windows\\System32")
            self.assertFalse(result["success"])
            self.assertFalse(os.path.exists("C:\\Windows\\System32\\System32"))
            self.assertIn("absolute", result["message"].lower())

    def test_create_folder_rejects_traversal(self):
        """'../evil' must not escape the workspace."""
        with temp_workspace() as (safe, _):
            result = create_folder("../evil")
            self.assertFalse(result["success"])
            self.assertFalse((safe.parent / "evil").exists())

    def test_create_folder_rejects_empty_name(self):
        """Empty names should fail cleanly."""
        result = create_folder("   ")
        self.assertFalse(result["success"])

    def test_create_folder_nested_relative(self):
        """A nested relative name should be created inside the workspace."""
        with temp_workspace() as (safe, _):
            result = create_folder(os.path.join("Parent", "Child"))
            self.assertTrue(result["success"])
            self.assertTrue((safe / "Parent" / "Child").is_dir())


class TestOpenFile(unittest.TestCase):
    """Test opening files (os.startfile is mocked)."""

    @patch("core.file_manager.os.startfile")
    def test_open_existing_safe_file(self, mock_startfile):
        """An existing safe file should open successfully."""
        with temp_workspace() as _:
            result = open_file("test.txt")
            self.assertTrue(result["success"])
            mock_startfile.assert_called_once()
            # The opened path must be the resolved safe file
            opened = mock_startfile.call_args[0][0]
            self.assertTrue(opened.endswith("test.txt"))

    @patch("core.file_manager.os.startfile")
    def test_open_missing_file_fails_cleanly(self, mock_startfile):
        """A missing file should fail cleanly, not crash."""
        with temp_workspace() as _:
            result = open_file("missing.txt")
            self.assertFalse(result["success"])
            self.assertIn("not found", result["message"].lower())
            mock_startfile.assert_not_called()

    @patch("core.file_manager.os.startfile")
    def test_open_unsafe_file_is_denied(self, mock_startfile):
        """A path outside the safe directories must be denied."""
        with temp_workspace() as (_, outside):
            result = open_file(str(outside))
            self.assertFalse(result["success"])
            self.assertIn("access denied", result["message"].lower())
            mock_startfile.assert_not_called()

    @patch("core.file_manager.os.startfile")
    def test_open_traversal_is_denied(self, mock_startfile):
        """A traversal path must be denied."""
        with temp_workspace() as (safe, _):
            result = open_file(str(safe / ".." / "outside.txt"))
            self.assertFalse(result["success"])
            self.assertIn("access denied", result["message"].lower())
            mock_startfile.assert_not_called()

    @patch("core.file_manager.os.startfile")
    def test_open_safe_directory(self, mock_startfile):
        """A safe workspace directory can be opened in Explorer."""
        with temp_workspace() as (safe, _):
            (safe / "some_dir").mkdir()
            result = open_file("some_dir")
            self.assertTrue(result["success"])
            self.assertIn("folder opened", result["message"].lower())
            mock_startfile.assert_called_once()

    @patch("core.file_manager.os.startfile")
    def test_open_empty_target_fails(self, mock_startfile):
        """Empty target should fail cleanly."""
        result = open_file("  ")
        self.assertFalse(result["success"])
        mock_startfile.assert_not_called()


class TestDeleteFile(unittest.TestCase):
    """Test file deletion (never recursive, always confirmed)."""

    def test_delete_requires_confirmation(self):
        """Without confirmed=True the file must remain."""
        with temp_workspace() as (safe, _):
            result = delete_file("test.txt")
            self.assertFalse(result["success"])
            self.assertIn("confirmation required", result["message"].lower())
            self.assertTrue((safe / "test.txt").exists())

    def test_delete_confirmed_removes_file(self):
        """With confirmed=True the file should be deleted."""
        with temp_workspace() as (safe, _):
            result = delete_file("test.txt", confirmed=True)
            self.assertTrue(result["success"])
            self.assertFalse((safe / "test.txt").exists())
            self.assertEqual(
                result["data"]["path"], str((safe / "test.txt").resolve())
            )

    def test_delete_outside_safe_directory_fails(self):
        """Files outside the safe directories must not be deleted."""
        with temp_workspace() as (_, outside):
            result = delete_file(str(outside), confirmed=True)
            self.assertFalse(result["success"])
            self.assertIn("access denied", result["message"].lower())
            self.assertTrue(outside.exists())

    def test_delete_traversal_fails(self):
        """Traversal paths must not delete outside files."""
        with temp_workspace() as (safe, _):
            result = delete_file(str(safe / ".." / "outside.txt"), confirmed=True)
            self.assertFalse(result["success"])
            self.assertTrue((safe.parent / "outside.txt").exists())

    def test_delete_missing_file_fails(self):
        """Deleting a non-existent file should fail cleanly."""
        with temp_workspace() as _:
            result = delete_file("nope.txt", confirmed=True)
            self.assertFalse(result["success"])
            self.assertIn("not found", result["message"].lower())

    def test_delete_directory_is_rejected(self):
        """Directories must never be deleted."""
        with temp_workspace() as (safe, _):
            (safe / "a_dir").mkdir()
            result = delete_file("a_dir", confirmed=True)
            self.assertFalse(result["success"])
            self.assertIn("folder", result["message"].lower())
            self.assertTrue((safe / "a_dir").is_dir())


class TestSearchFile(unittest.TestCase):
    """Test file searching within safe directories."""

    def test_search_returns_matches(self):
        """Searching for resume.pdf should return it."""
        with temp_workspace() as _:
            result = search_file("resume.pdf")
            self.assertTrue(result["success"])
            files = result["data"]["files"]
            self.assertEqual(len(files), 1)
            self.assertTrue(files[0].endswith("resume.pdf"))

    def test_search_substring_matching(self):
        """Searching 'resume' should match both resume files."""
        with temp_workspace() as _:
            result = search_file("resume")
            self.assertTrue(result["success"])
            files = result["data"]["files"]
            self.assertEqual(len(files), 2)
            names = sorted(Path(f).name for f in files)
            self.assertEqual(names, ["resume.pdf", "resume_old.pdf"])

    def test_search_case_insensitive(self):
        """Uppercase queries should match lowercase filenames."""
        with temp_workspace() as _:
            result = search_file("RESUME.PDF")
            self.assertTrue(result["success"])
            files = result["data"]["files"]
            self.assertEqual(len(files), 1)
            self.assertTrue(files[0].endswith("resume.pdf"))

    def test_search_no_matches(self):
        """A query with no matches should fail cleanly."""
        with temp_workspace() as _:
            result = search_file("definitely_missing_xyz.dat")
            self.assertFalse(result["success"])
            self.assertIn("no files found", result["message"].lower())

    def test_search_result_limit(self):
        """Results should respect the limit."""
        with temp_workspace() as (safe, _):
            for i in range(5):
                (safe / f"resume_{i}.pdf").write_bytes(b"%PDF")
            result = search_file("resume", limit=3)
            self.assertTrue(result["success"])
            self.assertEqual(len(result["data"]["files"]), 3)

    def test_search_default_limit_is_20(self):
        """The default limit should be 20."""
        self.assertEqual(DEFAULT_SEARCH_LIMIT, 20)

    def test_search_only_safe_directories(self):
        """Files outside safe directories must not be returned."""
        with temp_workspace() as (safe, outside):
            result = search_file("outside")
            self.assertFalse(result["success"])
            # The outside file exists but must not be found
            self.assertTrue(outside.exists())
            self.assertNotIn(str(outside), str(result.get("data", {})))

    def test_search_empty_query_fails(self):
        """An empty query should fail cleanly."""
        result = search_file("  ")
        self.assertFalse(result["success"])


class TestResultStructure(unittest.TestCase):
    """Test that all file operations return a consistent structure."""

    def test_success_result_shape(self):
        """Success results should have success, message, data."""
        with temp_workspace() as _:
            result = search_file("test")
            self.assertIn("success", result)
            self.assertIn("message", result)
            self.assertIn("data", result)
            self.assertIsInstance(result["success"], bool)
            self.assertIsInstance(result["message"], str)
            self.assertIsInstance(result["data"], dict)

    def test_failure_result_shape(self):
        """Failure results should have success, message, empty data."""
        result = delete_file("x.txt")  # not confirmed
        self.assertEqual(
            set(result.keys()), {"success", "message", "data"}
        )
        self.assertFalse(result["success"])
        self.assertEqual(result["data"], {})


class TestWindowsCompatibility(unittest.TestCase):
    """The module must import and validate safely on any platform."""

    def test_module_imports(self):
        """Importing the file manager must never crash."""
        import importlib

        mod = importlib.import_module("core.file_manager")
        self.assertTrue(hasattr(mod, "create_folder"))
        self.assertTrue(hasattr(mod, "open_file"))
        self.assertTrue(hasattr(mod, "delete_file"))
        self.assertTrue(hasattr(mod, "search_file"))

    @patch("core.file_manager.sys.platform", "linux")
    @patch("core.file_manager.os.startfile", create=True)
    def test_open_file_reports_platform_on_non_windows(self, mock_startfile):
        """On non-Windows, opening a valid file reports a clear failure."""
        with temp_workspace() as _:
            result = open_file("test.txt")
            self.assertFalse(result["success"])
            self.assertIn("windows", result["message"].lower())
            mock_startfile.assert_not_called()


if __name__ == "__main__":
    unittest.main()
