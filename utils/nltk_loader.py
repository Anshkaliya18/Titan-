"""
NLTK Loader (Step 22 — startup import optimization)

Imports NLTK's tokenizer with a small, deterministic import-time shim.

Why this exists
---------------
Importing ``nltk`` executes ``nltk/__init__.py``, which eagerly
star-imports optional subsystems (metrics, parse, classify, ...). Those
in turn pull in the heavy scientific stack from ``try/except
ImportError`` guards that NLTK provides for machines without the
packages installed::

    nltk/metrics/association.py :  try: from scipy.stats import fisher_exact
    nltk/metrics/scores.py      :  try: from scipy.stats.stats import betai
    nltk/parse/transitionparser.py:
        try:
            from numpy import array
            from scipy import sparse
            from sklearn import svm

Measured on this project (Step 22 baseline): with those packages
installed, ``import nltk`` costs ~1.39 s (scipy.stats ~0.6 s, plus the
pandas/pyarrow/requests chains it drags in). TITAN never uses NLTK's
collocation statistics or transition parsers — only
``nltk.tokenize.word_tokenize``.

The shim registers stand-in modules for the optional packages so that
NLTK's guarded imports take exactly the same fallback branch they take
on a machine where scipy/sklearn are not installed. The tokenizer
itself never touches them.

Measured effect: importing the tokenizer drops from ~1.39 s to
~0.31 s with byte-identical tokenization output (verified by the A/B
subprocess test in tests/test_performance.py).

Guarantees
----------
- Deterministic and single-threaded: runs once at import of
  core.tokenizer, before any command is processed.
- Never replaces a real package that is already imported.
- Fallback: if the shimmed import fails for any reason, the shim
  entries and any partially imported ``nltk`` modules are removed and
  the import is retried without shims. If that also fails, the original
  exception propagates — the pre-Step-22 behaviour is the fallback.
- No ML, no network, no downloads: only import wiring is affected.
"""

from __future__ import annotations

import sys
import types
from typing import Callable, List

from utils.paths import bundle_dir, is_frozen

# Optional NLTK dependencies TITAN never uses. Installed as stand-ins
# before `nltk` is imported (root packages plus the exact submodule
# paths NLTK imports by name, so no real package is ever touched).
#
# NOTE (Step 25): TITAN now uses scikit-learn for the ML intent
# classifier, which needs the REAL scipy/sklearn. The shims below are
# still installed first so NLTK takes its fast guarded fallback branch
# (Step 22 startup optimization) instead of loading sklearn.svm itself.
# Before the ML layer imports sklearn, core.ml_intent_classifier purges
# ONLY these shim objects from sys.modules (verified by type name —
# real modules are never removed) and then imports the real packages.
# NLTK modules already imported keep their guarded fallbacks, which
# TITAN never calls, so this is safe. See _ensure_real_ml_stack().
OPTIONAL_DEPENDENCY_SHIMS: tuple = (
    "scipy",            # nltk.metrics.association, nltk.metrics.scores
    "scipy.stats",
    "scipy.stats.stats",
    "scipy.sparse",     # nltk.parse.transitionparser
    "sklearn",          # nltk.parse.transitionparser
    "sklearn.datasets",
)


class MissingOptionalDependency(types.ModuleType):
    """Stand-in for an optional dependency that is not needed.

    Any attribute access raises ``ImportError`` — exactly what NLTK's
    guarded blocks expect when the package is absent. ``__path__`` is
    empty, so attempts to import a submodule fail fast with
    ``ModuleNotFoundError`` instead of loading the real package.
    """

    def __getattr__(self, name: str):
        raise ImportError(
            f"{self.__name__} is unavailable (TITAN import shim)"
        )


def install_optional_dependency_shims() -> List[str]:
    """Register shim modules that are not already present.

    Returns:
        Names actually installed, so the fallback can remove them.
        Real modules already in ``sys.modules`` are never replaced.
    """
    installed: List[str] = []
    for name in OPTIONAL_DEPENDENCY_SHIMS:
        if name in sys.modules:
            continue
        shim = MissingOptionalDependency(name)
        shim.__path__ = []  # package stub: blocks real submodule search
        sys.modules[name] = shim
        installed.append(name)
    return installed


def _purge(names: List[str]) -> None:
    """Remove entries from sys.modules (best effort)."""
    for name in names:
        sys.modules.pop(name, None)


def register_bundled_nltk_data() -> bool:
    """Point NLTK at the ``nltk_data`` folder bundled with the build.

    Packaged mode only (Step 23, Phase 9): TITAN.spec collects the
    tokenizers/punkt_tab resource into the application bundle, and
    this adds that folder to NLTK's search path so tokenization works
    offline on a machine that has never run ``nltk.downloader``.

    In source mode nothing changes — NLTK's own search path already
    finds the user's ``nltk_data`` folder.

    This function never downloads anything. A missing resource stays
    missing and raises NLTK's usual LookupError when it is needed.

    Returns:
        bool: True when the bundled folder was added, False otherwise.
    """
    if not is_frozen():
        return False
    try:
        import nltk.data

        candidate = bundle_dir() / "nltk_data"
        if candidate.is_dir() and str(candidate) not in nltk.data.path:
            nltk.data.path.insert(0, str(candidate))
            return True
    except Exception:
        # Registration must never break tokenization setup; NLTK will
        # simply search its default locations.
        return False
    return False


def import_word_tokenize(use_shims: bool = True) -> Callable[[str], list]:
    """Import and return ``nltk.tokenize.word_tokenize``.

    Args:
        use_shims: When True (production default), install the optional
            dependency shims first — but only if ``nltk`` has not been
            imported yet. On ANY import failure the shims and partially
            imported ``nltk`` modules are purged and the import is
            retried without shims, so a future NLTK that genuinely
            requires scipy still loads (at the original cost).

    Returns:
        The ``word_tokenize`` callable. In a packaged build the bundled
        ``nltk_data`` folder is registered on NLTK's search path first
        (Phase 9) so tokenization works without internet access.

    Raises:
        ImportError: if NLTK cannot be imported even without shims
            (same behaviour as before Step 22).
    """
    installed: List[str] = []
    if use_shims and "nltk" not in sys.modules:
        installed = install_optional_dependency_shims()

    try:
        from nltk.tokenize import word_tokenize

    except Exception:
        if not use_shims:
            raise
        # Fallback: drop shim entries and every (possibly half-imported)
        # nltk module, then import again with the real environment.
        _purge(installed)
        _purge([k for k in sys.modules if k == "nltk" or k.startswith("nltk.")])
        from nltk.tokenize import word_tokenize

    # Packaged builds register their bundled nltk_data folder (a no-op
    # in source mode; see register_bundled_nltk_data — never downloads).
    register_bundled_nltk_data()

    return word_tokenize
