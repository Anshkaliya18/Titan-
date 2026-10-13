"""
Pytest Configuration

Redirects logging to a temporary directory before any test module is
imported, so the real logs/TITAN.log is never written during tests.
"""

import sys
import tempfile
from pathlib import Path

import pytest

# Make project modules importable (mirrors each test file's own hack)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import configure_logging  # noqa: E402

# Session log directory — temporary, never the real project logs/
TEST_LOG_DIR = Path(tempfile.mkdtemp(prefix="TITAN-test-logs-"))
configure_logging(log_dir=TEST_LOG_DIR)


@pytest.fixture(autouse=True)
def _reset_application_resolution_cache():
    """
    Step 22: executable resolutions are memoized between commands, so
    every test starts (and ends) with an empty cache. This keeps each
    test's patched isfile/which discovery behaviour deterministic.
    """
    from core import application_manager

    application_manager.invalidate_application_cache()
    yield
    application_manager.invalidate_application_cache()
