"""
Performance regression tests (Step 22)

Design rules:
- Deterministic structural assertions wherever possible (bounded
  caches, bounded history, stable handler counts, in-memory config,
  memoized resolution) — these cannot flake with machine speed.
- Timing assertions use deliberately generous, order-of-magnitude
  thresholds, documented as approximate; they only catch catastrophic
  regressions (e.g. losing the NLTK import shim entirely), never
  normal machine variance.
- All external effects are mocked: no application, browser, file
  system or network operation ever runs for real.
- Startup and tokenization A/B checks run in fresh subprocesses so
  they measure what a real `python main.py` launch experiences.
"""

import io
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import tracemalloc
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

# Add parent directory to path so we can import core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import main
from core.command_history import CommandHistory, set_command_history
from core.session_state import SessionState

ROOT = str(Path(__file__).resolve().parents[1])

# Approximate thresholds (documented as such). Baseline on this
# machine before Step 22: `import main` ≈ 1630 ms, optimized ≈ 350 ms.
# With ML classifier (Step 25): `import main` ≈ 1500-2000 ms due to
# numpy/scipy/sklearn imports. The threshold sits far above both so it
# only fires on catastrophic regressions (e.g. the shim silently
# disappearing plus more), while the *structural* shim test below
# pins the actual optimization.
GENEROUS_STARTUP_MS = 6000.0

# Tokenization battery for the shim A/B comparison — includes the
# quirky cases the tokenizer is pinned to (contract punctuation,
# unicode, URLs, casing).
_TOKEN_BATTERY = [
    "hello world",
    "open chrome",
    "don't stop believin'",
    "Hello, World!",
    "a b c",
    "what's up?",
    "WAIT really?",
    "semi;colon test",
    "UPPER lower MiXeD",
    "one. two. three.",
    "12 + 34 = 46",
    "  spaced  out  ",
    "path\\to\\file.txt",
    "email@example.com",
    "über naïve résumé",
    "STOP. wait.",
    "line1\nline2",
    "(paren) [bracket] {brace}",
    "multiple&&ampersands",
    "price$100 100%",
    "¿qué pasa?",
    "e-mail x",
    "",
]

_SOAK_CYCLE = [
    "help",
    "history",
    "open chrome",
    "opn chrokme",
    "open chrome and open calculator",
]


def _run_python(code: str, *args: str, timeout: float = 90.0) -> str:
    """Run a snippet in a fresh interpreter; return stdout."""
    proc = subprocess.run(
        [sys.executable, "-c", code, *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"subprocess failed ({proc.returncode}):\n{proc.stderr[-2000:]}"
        )
    return proc.stdout


# Snippet used by every subprocess that imports Jarvis: redirect the
# session logger to a temporary directory first, so the real
# logs/jarvis.log is never written during tests.
_TEMP_LOG_PRELUDE = (
    "import tempfile, pathlib, sys; "
    "sys.path.insert(0, {root!r}); "
    "import utils.logger as _ul; "
    "_ul.DEFAULT_LOG_DIR = pathlib.Path(tempfile.mkdtemp()); "
).format(root=ROOT)


class _PerfTestCase(unittest.TestCase):
    """Fresh session + bounded history per test; executor fully mocked."""

    def setUp(self):
        self._original_session = main.session
        main.session = SessionState()
        self.history = CommandHistory(max_entries=100)
        set_command_history(self.history)
        self._buffer = io.StringIO()
        self._exec_patcher = patch(
            "main.execute_command",
            return_value={"success": True, "message": "ok (mocked)",
                          "data": {}},
        )
        self._multi_patcher = patch(
            "core.multi_command_executor.execute_command",
            return_value={"success": True, "message": "ok (mocked)",
                          "data": {}},
        )
        self._exec_patcher.start()
        self._multi_patcher.start()

    def tearDown(self):
        self._exec_patcher.stop()
        self._multi_patcher.stop()
        main.session = self._original_session
        set_command_history(None)

    def _run_turn(self, text: str) -> str:
        with patch("main.get_input", return_value=text):
            with redirect_stdout(self._buffer):
                main._process_once()
        return self._buffer.getvalue()

    def _run_cycle(self, count: int, start: int = 0) -> None:
        for i in range(count):
            text = _SOAK_CYCLE[(start + i) % len(_SOAK_CYCLE)]
            with patch("main.get_input", return_value=text):
                with redirect_stdout(self._buffer):
                    main._process_once()
            # Reuse the buffer so captured output never accumulates
            # (the memory test measures the app, not the harness).
            self._buffer.seek(0)
            self._buffer.truncate(0)


# ---------------------------------------------------------------------------
# Startup (Phase 22 / Phase 2)
# ---------------------------------------------------------------------------

class TestStartupPerformance(unittest.TestCase):
    """Fresh-subprocess startup measurements and shim structure."""

    def test_import_main_under_generous_threshold(self):
        """Approximate timing guard — see GENEROUS_STARTUP_MS."""
        samples = []
        for _ in range(3):
            t0 = time.perf_counter()
            _run_python(_TEMP_LOG_PRELUDE + "import main")
            samples.append((time.perf_counter() - t0) * 1000.0)
        median = sorted(samples)[1]
        self.assertLess(
            median,
            GENEROUS_STARTUP_MS,
            f"import main took {median:.0f} ms (limit "
            f"{GENEROUS_STARTUP_MS:.0f} ms, approximate) — startup "
            "regressed catastrophically",
        )

    def test_nltk_optional_dependency_shim_is_active(self):
        """The startup shim must actually be in place (structural).

        Step 25 note: the full Step 22 shim set is restored
        (scipy/sklearn roots included) so NLTK never loads sklearn.svm
        itself. The ML layer purges ONLY shim objects before importing
        the real stack (see _ensure_real_ml_stack), so both systems
        coexist: fast NLTK import + working sklearn on demand.
        """
        out = _run_python(
            _TEMP_LOG_PRELUDE
            + "import sys; import core.tokenizer; "
              "m = sys.modules.get('scipy'); "
              "print(type(m).__name__ if m is not None else 'none')"
        )
        self.assertEqual(out.strip(), "MissingOptionalDependency")

    def test_tokenization_identical_with_and_without_shims(self):
        """A/B: NLTK's guarded fallback path tokenizes identically."""
        script = (
            "import json, sys; "
            "sys.path.insert(0, %r); "
            "from utils.nltk_loader import import_word_tokenize; "
            "mode = sys.argv[1]; "
            "wt = import_word_tokenize(use_shims=(mode == 'shim')); "
            "battery = json.loads(sys.argv[2]); "
            "tokens = [wt(t) for t in battery]; "
            "real_scipy = 'scipy' in sys.modules and "
            "type(sys.modules['scipy']).__name__ != "
            "'MissingOptionalDependency'; "
            "print(json.dumps({'tokens': tokens, "
            "'real_scipy': real_scipy}, ensure_ascii=False))"
        ) % ROOT
        args = json.dumps(_TOKEN_BATTERY, ensure_ascii=False)

        shim = json.loads(_run_python(script, "shim", args))
        plain = json.loads(_run_python(script, "plain", args))

        # The plain path really did the heavy import (control).
        self.assertTrue(
            plain["real_scipy"],
            "control run should import real scipy (baseline path)",
        )
        # The optimized path did not.
        self.assertFalse(shim["real_scipy"])
        # Tokenization output is byte-identical either way.
        self.assertEqual(shim["tokens"], plain["tokens"])

    def test_production_tokenizer_matches_plain_import(self):
        """core.tokenizer's output equals the unshimmed NLTK output."""
        script = (
            "import json, sys; "
            "sys.path.insert(0, %r); "
            "from utils.nltk_loader import import_word_tokenize; "
            "wt = import_word_tokenize(use_shims=False); "
            "battery = json.loads(sys.argv[1]); "
            "print(json.dumps([wt(t) for t in battery], ensure_ascii=False))"
        ) % ROOT
        plain = json.loads(_run_python(script, json.dumps(_TOKEN_BATTERY,
                                                          ensure_ascii=False)))
        from core.tokenizer import tokenize

        production = [tokenize(t) for t in _TOKEN_BATTERY]
        self.assertEqual(production, plain)

    def test_loader_falls_back_when_shimmed_import_fails(self):
        """A failing shim-path import retries cleanly without shims."""
        import utils.nltk_loader as loader

        poisoned = object()
        sys.modules["nltk.tokenize"] = poisoned
        try:
            wt = loader.import_word_tokenize(use_shims=True)
        finally:
            if sys.modules.get("nltk.tokenize") is poisoned:
                del sys.modules["nltk.tokenize"]

        self.assertTrue(callable(wt))
        self.assertEqual(wt("hello world"), ["hello", "world"])


# ---------------------------------------------------------------------------
# Spell correction cache (Phase 4)
# ---------------------------------------------------------------------------

class TestSpellCorrectionCache(unittest.TestCase):
    """Bounded FIFO cache: hits, eviction, correctness, isolation."""

    @staticmethod
    def _model() -> "object":
        from core.spell_model import SpellModel

        # Tiny in-memory vocabulary — no disk access.
        return SpellModel({"open": 500, "chrome": 400, "hello": 10})

    def test_known_words_bypass_the_cache(self):
        model = self._model()
        self.assertEqual(model.correct("chrome"), "chrome")
        self.assertEqual(model.correct("chrome"), "chrome")
        self.assertEqual(model.cache_size, 0)
        self.assertEqual(model.cache_hits, 0)
        self.assertEqual(model.cache_misses, 0)

    def test_unknown_word_is_cached_after_first_computation(self):
        model = self._model()
        first = model.correct("opne")     # distance-1 typo of "open"
        self.assertEqual(first, "open")
        self.assertEqual(model.cache_size, 1)
        self.assertEqual(model.cache_misses, 1)

        second = model.correct("opne")
        self.assertEqual(second, "open")
        self.assertEqual(model.cache_hits, 1)
        self.assertEqual(model.cache_misses, 1)

    def test_cached_result_matches_uncached_computation(self):
        model_a = self._model()
        model_b = self._model()  # separate cache, same vocabulary
        result_a = model_a.correct("chrpme")
        result_b = model_b.correct("chrpme")
        self.assertEqual(result_a, result_b)
        # And the cached replay is identical.
        self.assertEqual(model_a.correct("chrpme"), result_a)

    def test_cache_is_bounded_and_evicts_oldest_first(self):
        from core.spell_model import SpellModel

        # Vocabulary of base words; probes are one edit away (d1 hit),
        # so each computation is fast while still exercising the cache.
        limit = SpellModel.CACHE_MAX_SIZE
        vocab = {f"zz{i}": 1 for i in range(limit + 60)}
        model = SpellModel(vocab)

        probes = [f"zz{i}q" for i in range(limit + 50)]
        for probe in probes:
            model.correct(probe)
            self.assertLessEqual(model.cache_size, limit)

        self.assertEqual(model.cache_size, limit)
        self.assertNotIn(
            probes[0], model._correction_cache,
            "oldest entry should be evicted (FIFO)",
        )
        self.assertNotIn(
            probes[49], model._correction_cache,
            "all 50 overflow entries should be evicted",
        )
        self.assertIn(probes[50], model._correction_cache)
        self.assertIn(probes[-1], model._correction_cache)
        self.assertEqual(model.cache_misses, limit + 50)

    def test_fresh_model_instance_starts_with_empty_cache(self):
        from core.spell_model import SpellModel

        model = SpellModel.load()
        try:
            self.assertEqual(model.cache_size, 0)
        finally:
            del model  # 3.9 MB vocabulary released again

    def test_clear_cache_resets_counters(self):
        model = self._model()
        model.correct("opne")
        model.correct("opne")
        self.assertEqual(model.cache_hits, 1)
        model.clear_cache()
        self.assertEqual(model.cache_size, 0)
        self.assertEqual(model.cache_hits, 0)
        self.assertEqual(model.cache_misses, 0)


# ---------------------------------------------------------------------------
# Configuration / registry: constructed once, never per command (Phases 6/12/18)
# ---------------------------------------------------------------------------

class TestNoPerCommandRebuilds(_PerfTestCase):
    """Static structures are not re-read or rebuilt for every command."""

    def test_settings_json_never_opened_during_commands(self):
        import builtins

        real_open = builtins.open
        settings_reads = []

        def spy(file, *args, **kwargs):
            if "settings.json" in str(file):
                settings_reads.append(str(file))
            return real_open(file, *args, **kwargs)

        with patch("builtins.open", side_effect=spy):
            for _ in range(30):
                self._run_turn("open chrome")
                self._run_turn("help")

        self.assertEqual(
            settings_reads, [],
            "configuration must stay in memory during command processing",
        )

    def test_config_not_reloaded_and_registry_not_revalidated(self):
        import core.command_registry as registry
        import core.config_manager as config

        with patch.object(
            config, "_load_config",
            side_effect=AssertionError("config reloaded per command"),
        ), patch.object(
            registry, "validate_registry",
            side_effect=AssertionError("registry revalidated per command"),
        ):
            for _ in range(20):
                self._run_turn("opn chrokme")
                self._run_turn("help")

    def test_config_object_identity_is_stable(self):
        import core.config_manager as config

        first = id(config._CONFIG)
        for _ in range(10):
            self._run_turn("open chrome")
        self.assertEqual(id(config._CONFIG), first)

    def test_session_history_and_registry_objects_are_reused(self):
        import core.command_registry as registry

        session_ids = set()
        history_ids = set()
        registry_ids = set()
        for _ in range(15):
            self._run_turn("open chrome")
            session_ids.add(id(main.session))
            history_ids.add(id(main.get_command_history()))
            registry_ids.add(id(registry._COMMANDS))

        self.assertEqual(len(session_ids), 1, "session rebuilt per command")
        self.assertEqual(len(history_ids), 1, "history rebuilt per command")
        self.assertEqual(len(registry_ids), 1, "registry rebuilt per command")


# ---------------------------------------------------------------------------
# Application resolution cache (Phase 10)
# ---------------------------------------------------------------------------

class TestApplicationResolutionCache(unittest.TestCase):
    """Positive resolutions are memoized; the allowlist still governs."""

    def setUp(self):
        from core import application_manager as am

        self.am = am
        am.invalidate_application_cache()

    def tearDown(self):
        self.am.invalidate_application_cache()

    def test_second_resolution_reuses_the_cached_path(self):
        calls = []

        def fake_isfile(path):
            calls.append(path)
            return True

        with patch.object(self.am.os.path, "isfile", side_effect=fake_isfile):
            first = self.am.resolve_application("chrome")
            calls.clear()
            second = self.am.resolve_application("chrome")

        self.assertIsNotNone(first)
        self.assertEqual(first, second)
        self.assertEqual(calls, [], "second resolve must not re-stat paths")

    def test_failed_resolution_is_not_cached(self):
        calls = []

        def fake_isfile(path):
            calls.append(path)
            return False

        with patch.object(self.am.os.path, "isfile", side_effect=fake_isfile), \
                patch.object(self.am.shutil, "which", return_value=None):
            self.assertIsNone(self.am.resolve_application("chrome"))
            self.assertIsNone(self.am.resolve_application("chrome"))

        # Both attempts performed discovery — a newly installed
        # application is found without any invalidation call.
        self.assertGreaterEqual(len(calls), 2)
        self.assertNotIn("chrome", self.am._RESOLVED_PATHS)

    def test_invalidate_forces_recheck(self):
        calls = []

        def fake_isfile(path):
            calls.append(path)
            return True

        with patch.object(self.am.os.path, "isfile", side_effect=fake_isfile):
            self.am.resolve_application("chrome")
            self.assertIn("chrome", self.am._RESOLVED_PATHS)
            after_first = len(calls)

            self.am.resolve_application("chrome")
            self.assertEqual(len(calls), after_first, "second call cached")

            self.am.invalidate_application_cache()
            self.assertNotIn("chrome", self.am._RESOLVED_PATHS)

            self.am.resolve_application("chrome")
            self.assertGreater(len(calls), after_first,
                               "invalidation must force a recheck")

    def test_unknown_name_never_reaches_the_cache(self):
        result = self.am.resolve_application("totally-not-an-app")
        self.assertIsNone(result)
        self.assertEqual(self.am._RESOLVED_PATHS, {})


# ---------------------------------------------------------------------------
# Memory & bounds under load (Phase 20)
# ---------------------------------------------------------------------------

class TestBoundedMemoryAndHistory(_PerfTestCase):
    """1,000 mocked commands: history, caches and handlers stay bounded."""

    WARMUP = 100
    COMMANDS = 1000
    # Approximate, deliberately generous: the pre-change baseline grew
    # ~0.1 MB over 1,000 commands; 8 MB flags real leaks only.
    MAX_GROWTH_BYTES = 8 * 1024 * 1024

    def test_thousand_command_soak_stays_bounded(self):
        import gc

        handlers_before = len(logging.getLogger().handlers)

        # Warm-up: lazy loads (spell model, help text, punkt) excluded
        # from the growth measurement.
        self._run_cycle(self.WARMUP)
        gc.collect()

        # Suppress log emission during the measured window: pytest's
        # capture handlers retain every emitted record in memory, which
        # would measure the test runner rather than the application.
        # Handlers stay attached, so handler-count checks still apply.
        logging.disable(logging.CRITICAL)
        tracemalloc.start()
        baseline = tracemalloc.take_snapshot()
        try:
            self._run_cycle(self.COMMANDS, start=self.WARMUP)
            gc.collect()
            after = tracemalloc.take_snapshot()
            current, _peak = tracemalloc.get_traced_memory()
        finally:
            logging.disable(logging.NOTSET)
            tracemalloc.stop()

        # History: exactly at its configured maximum, never above.
        self.assertEqual(self.history.count(), self.history.max_entries)
        self.assertEqual(self.history.max_entries, 100)

        # Logger handlers: created once, never duplicated under load.
        self.assertEqual(len(logging.getLogger().handlers),
                         handlers_before)
        configure_calls = len(logging.getLogger().handlers)
        from utils.logger import configure_logging

        configure_logging()
        configure_logging()
        self.assertEqual(len(logging.getLogger().handlers),
                         configure_calls)

        # Spell cache: bounded by its maximum.
        from core.spell_checker import load_model
        from core.spell_model import SpellModel

        model = load_model()
        self.assertLessEqual(model.cache_size, SpellModel.CACHE_MAX_SIZE)

        # Heap growth: bounded (approximate threshold).
        growth = sum(s.size_diff for s in after.compare_to(baseline,
                                                           "filename"))
        self.assertLess(
            growth, self.MAX_GROWTH_BYTES,
            f"heap grew {growth / 1048576:.2f} MB over "
            f"{self.COMMANDS} commands (limit "
            f"{self.MAX_GROWTH_BYTES / 1048576:.0f} MB, approximate)",
        )

    def test_soak_keeps_pipeline_responses_correct(self):
        """Pipeline still completes and responds under load.

        The executor is mocked, so help/history text generation (which
        lives inside the executor) is covered by their own test files;
        here we only verify the pipeline reaches execution and formats
        a response for every turn.
        """
        out = self._run_turn("opn chrokme")
        self.assertIn("ok (mocked)", out)
        out = self._run_turn("help")
        self.assertIn("ok (mocked)", out)
        out = self._run_turn("history")
        self.assertIn("ok (mocked)", out)


# ---------------------------------------------------------------------------
# Rotating log handler fast path (Phase 13)
# ---------------------------------------------------------------------------

class TestRotatingHandlerFastPath(unittest.TestCase):
    """The rollover fast path must not change rotation behaviour."""

    def _write_records(self, handler, count, marker):
        logger = logging.getLogger("test.perf.rotation")
        old_handlers = logger.handlers[:]
        old_level = logger.level
        old_propagate = logger.propagate
        try:
            logger.handlers = [handler]
            logger.setLevel(logging.DEBUG)
            logger.propagate = False
            for i in range(count):
                logger.info("%s record %03d %s", marker, i, "x" * 60)
        finally:
            logger.handlers = old_handlers
            logger.setLevel(old_level)
            logger.propagate = old_propagate

    def test_rotation_still_occurs_when_limit_is_exceeded(self):
        from utils.logger import _SizeRotatingFileHandler

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rot.log"
            handler = _SizeRotatingFileHandler(
                path, maxBytes=600, backupCount=2, encoding="utf-8"
            )
            try:
                self._write_records(handler, 40, "rotation-test")
                handler.close()
            finally:
                handler.close()

            self.assertTrue(path.exists())
            backup = Path(str(path) + ".1")
            self.assertTrue(
                backup.exists(),
                "rotation must still produce a backup file at the limit",
            )
            # Main file never exceeds the limit by much (stock
            # threshold check runs at the decision point).
            self.assertLessEqual(path.stat().st_size, 600 + 512)

    def test_normal_records_are_written_with_expected_format(self):
        from utils.logger import _SizeRotatingFileHandler

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fmt.log"
            handler = _SizeRotatingFileHandler(
                path, maxBytes=10 * 1024 * 1024, backupCount=1,
                encoding="utf-8",
            )
            handler.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            try:
                self._write_records(handler, 5, "format-test")
                handler.flush()
                content = path.read_text(encoding="utf-8")
            finally:
                handler.close()

            self.assertIn("format-test record 000", content)
            lines = [l for l in content.splitlines() if "record 000" in l]
            self.assertEqual(len(lines), 1)
            self.assertRegex(
                lines[0],
                r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| INFO \| "
                r"test\.perf\.rotation \| format-test record 000",
            )

    def test_should_rollover_false_while_well_below_limit(self):
        from utils.logger import _SizeRotatingFileHandler

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "near.log"
            handler = _SizeRotatingFileHandler(
                path, maxBytes=10 * 1024 * 1024, backupCount=1,
                encoding="utf-8",
            )
            try:
                record = logging.LogRecord(
                    "test.near", logging.INFO, __file__, 1,
                    "small message", None, None,
                )
                self.assertFalse(handler.shouldRollover(record))
            finally:
                handler.close()


if __name__ == "__main__":
    unittest.main()
