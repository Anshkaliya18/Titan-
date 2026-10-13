"""
Jarvis Performance Benchmark (Step 22)

A reproducible, side-effect-free benchmark for startup time and
per-command processing latency. Nothing dangerous ever runs:

- every command path has `execute_command` mocked (no applications,
  no browsers, no file/system effects),
- the executor's help/history formatting is measured directly,
- startup is measured in a fresh subprocess (`import main` only).

Usage:
    python tests/benchmark.py            # full run
    python tests/benchmark.py --quick    # fewer iterations (smoke)

Metrics:
    1. Startup (subprocess, median):  bare `python -c pass` vs `import main`
       (`import main` = interpreter start + all imports + module-level
       initialization — everything that happens before the first prompt).
    2. Simple command        : "open chrome"     (executor mocked)
    3. Spell correction      : "opn chrokme"     (executor mocked)
    4. Multi-command         : "open chrome and open calculator"
    5. Help command          : "help"            (pipeline, executor mocked)
    6. History command       : "history"         (pipeline, executor mocked)
    7. Help text generation  : help_manager.get_help() measured directly
    8. History list formatting: get_history_result() direct, full history
    9. Memory                : tracemalloc import peak + growth over 1,000
                               lightweight mocked commands.

Results are medians over N runs and are intended for BEFORE/AFTER
comparison on the same machine — not as absolute guarantees. Broad
regression thresholds live in tests/test_performance.py.
"""

from __future__ import annotations

import argparse
import io
import platform
import statistics
import subprocess
import sys
import time
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Deterministic, side-effect-free result for the mocked executor.
_OK_RESULT = {"success": True, "message": "ok (mocked)", "data": {}}

# Lightweight command cycle used for the memory soak.
_SOAK_CYCLE = [
    "help",
    "history",
    "open chrome",
    "opn chrokme",
    "open chrome and open calculator",
]


# ---------------------------------------------------------------------------
# Startup (subprocess)
# ---------------------------------------------------------------------------

def _subprocess_ms(code: str, runs: int) -> list[float]:
    """Run `python -c code` in a fresh process; return wall times in ms."""
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        samples.append((time.perf_counter() - t0) * 1000.0)
        if proc.returncode != 0:
            raise RuntimeError(
                f"startup probe failed ({proc.returncode}): {proc.stderr[:500]}"
            )
    return samples


def _subprocess_output(code: str) -> str:
    """Run `python -c code` once and return its stdout."""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"subprocess probe failed ({proc.returncode}): {proc.stderr[:500]}"
        )
    return proc.stdout.strip()


def measure_startup(runs: int) -> dict:
    """Measure bare interpreter start and Jarvis import+init."""
    bare = _subprocess_ms("pass", max(3, runs // 2))
    jarvis = _subprocess_ms(
        "import main  # noqa: F401  (full Jarvis import + initialization)",
        runs,
    )
    import_memory = _subprocess_output(
        "import tracemalloc; tracemalloc.start(); import main; "
        "cur, peak = tracemalloc.get_traced_memory(); "
        "print(f'{cur / 1048576:.2f} {peak / 1048576:.2f}')"
    )
    return {"bare": bare, "jarvis": jarvis, "import_memory": import_memory}


# ---------------------------------------------------------------------------
# Command processing (in-process, executor mocked)
# ---------------------------------------------------------------------------

def _bench_one(text: str, runs: int) -> tuple[float, float, float]:
    """Run one pipeline turn `runs` times; return (median, min, max) ms."""
    samples = []
    for _ in range(runs):
        out, err = io.StringIO(), io.StringIO()
        t0 = time.perf_counter()
        with redirect_stdout(out), redirect_stderr(err):
            from main import _process_once  # local import: keeps startup honest

            _process_once()
        samples.append((time.perf_counter() - t0) * 1000.0)
        if not out.getvalue():
            raise RuntimeError(f"no response printed for {text!r}")
    return (statistics.median(samples), min(samples), max(samples))


def benchmark_commands(runs: int) -> tuple[list[tuple], float]:
    """Benchmark the pipeline; returns (rows, in-process import ms)."""
    t0 = time.perf_counter()
    import main
    from core.session_state import SessionState

    import_ms = (time.perf_counter() - t0) * 1000.0

    main.session = SessionState()

    metrics = [
        ("Simple command  (open chrome)", "open chrome"),
        ("Spell correction (opn chrokme)", "opn chrokme"),
        ("Multi-command (open and open)", "open chrome and open calculator"),
        ("Help command (pipeline)", "help"),
        ("History command (pipeline)", "history"),
    ]

    rows = []
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(main, "execute_command", lambda *a, **k: dict(_OK_RESULT))
        )
        stack.enter_context(
            patch(
                "core.multi_command_executor.execute_command",
                lambda *a, **k: dict(_OK_RESULT),
            )
        )

        # Fill history to its configured maximum so the history metric
        # measures formatting of a full list (entries are recorded by the
        # real pipeline; the history command itself is exempt).
        get_history = main.get_command_history
        fill_input = patch.object(main, "get_input", lambda: "open chrome")
        with fill_input, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            while get_history().count() < get_history().max_entries:
                main._process_once()

        for label, text in metrics:
            with patch.object(main, "get_input", lambda t=text: t):
                for _ in range(3):  # warm-up (loads spell model, punkt, caches)
                    out, err = io.StringIO(), io.StringIO()
                    with redirect_stdout(out), redirect_stderr(err):
                        main._process_once()
                med, mn, mx = _bench_one(text, runs)
            rows.append((label, med, mn, mx))

        # Response formatting that lives inside the executor (the
        # executor itself is mocked above): measure the real work.
        from core.help_manager import get_help
        from core.command_history import get_command_history

        def _timed(fn, n):
            samples = []
            for _ in range(n):
                t0 = time.perf_counter()
                fn()
                samples.append((time.perf_counter() - t0) * 1000.0)
            return (statistics.median(samples), min(samples), max(samples))

        get_help()  # warm
        get_command_history().get_history_result()  # warm
        rows.append(("Help text generation", *_timed(get_help, runs)))
        rows.append(
            ("History list formatting", *_timed(
                get_command_history().get_history_result, runs))
        )
    return rows, import_ms


# ---------------------------------------------------------------------------
# Memory soak (tracemalloc)
# ---------------------------------------------------------------------------

def benchmark_memory(commands: int) -> dict:
    """Process `commands` mocked turns and report bounded-memory facts."""
    import tracemalloc

    import main
    from core.session_state import SessionState

    main.session = SessionState()
    tracemalloc.start()

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(main, "execute_command", lambda *a, **k: dict(_OK_RESULT))
        )
        stack.enter_context(
            patch(
                "core.multi_command_executor.execute_command",
                lambda *a, **k: dict(_OK_RESULT),
            )
        )

        # Warm-up so first-time lazy loads (spell model, help text) are
        # excluded from the growth measurement.
        with patch.object(main, "get_input", lambda: _SOAK_CYCLE[0]):
            for _ in range(50):
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    main._process_once()

        baseline = tracemalloc.take_snapshot()

        step = max(1, commands // 10)
        for i in range(commands):
            with patch.object(main, "get_input", lambda: _SOAK_CYCLE[i % len(_SOAK_CYCLE)]):
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    main._process_once()
            if i % step == 0:
                # Periodic progress marker for very slow machines.
                tracemalloc.get_traced_memory()

        after = tracemalloc.take_snapshot()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    diff = after.compare_to(baseline, "filename")
    growth = sum(stat.size_diff for stat in diff)

    history = main.get_command_history()
    import logging

    # configure_logging attaches handlers to the ROOT logger.
    root_logger = logging.getLogger()
    return {
        "commands": commands,
        "growth_bytes": growth,
        "current_mb": current / 1048576.0,
        "peak_mb": peak / 1048576.0,
        "history_count": history.count(),
        "history_max": history.max_entries,
        "logger_handlers": len(root_logger.handlers),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_table(title: str, rows: list[tuple], decimals: int) -> None:
    print(f"\n{title}")
    header = f"{'Metric':<34}{'Median':>11}{'Min':>11}{'Max':>11}   ms"
    print(header)
    print("-" * len(header))
    for label, med, mn, mx in rows:
        fmt = f"{{:>11.{decimals}f}}"
        print(f"{label:<34}{fmt.format(med)}{fmt.format(mn)}{fmt.format(mx)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Jarvis performance benchmark")
    parser.add_argument("--quick", action="store_true", help="fewer iterations")
    args = parser.parse_args()

    startup_runs = 3 if args.quick else 7
    cmd_runs = 15 if args.quick else 60
    soak_commands = 300 if args.quick else 1000

    print(f"Jarvis benchmark — Python {platform.python_version()} "
          f"on {platform.system()} {platform.machine()}")
    print(f"runs: startup={startup_runs}, commands={cmd_runs}, "
          f"memory soak={soak_commands}")

    # -- 1. Startup (fresh subprocesses) -----------------------------------
    startup = measure_startup(startup_runs)
    _print_table(
        "Startup (fresh subprocess, wall time)",
        [
            ("Bare python (reference)", statistics.median(startup["bare"]),
             min(startup["bare"]), max(startup["bare"])),
            ("import main (Jarvis init)", statistics.median(startup["jarvis"]),
             min(startup["jarvis"]), max(startup["jarvis"])),
        ],
        decimals=1,
    )
    if startup["import_memory"]:
        cur, peak = startup["import_memory"].split()
        print(f"  tracemalloc during import: current={cur} MB, peak={peak} MB")

    # -- 2. Command processing (in-process) --------------------------------
    rows, import_ms = benchmark_commands(cmd_runs)
    _print_table(
        "Command processing (in-process, executor mocked, warm)",
        rows,
        decimals=3,
    )
    print(f"  in-process `import main` (warm OS cache): {import_ms:.1f} ms")

    # -- 3. Memory soak ----------------------------------------------------
    mem = benchmark_memory(soak_commands)
    print("\nMemory soak (tracemalloc, mocked execution)")
    print("-" * 60)
    print(f"  commands processed : {mem['commands']}")
    print(f"  heap growth        : {mem['growth_bytes'] / 1048576.0:+.3f} MB")
    print(f"  current / peak     : {mem['current_mb']:.2f} / {mem['peak_mb']:.2f} MB")
    print(f"  history entries    : {mem['history_count']} "
          f"(max {mem['history_max']})")
    print(f"  logger handlers    : {mem['logger_handlers']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
