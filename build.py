#!/usr/bin/env python
"""
Repeatable Windows build for Jarvis (Step 23 — Phase 11).

Usage:
    python build.py               clean, test, build, verify
    python build.py --skip-tests  build only (faster iteration)
    python build.py --help

Steps (in order):
    1. Preflight     confirm Windows + PyInstaller are available
    2. Clean         remove ONLY the build-owned directories (build/,
                     dist/) — nothing else is ever deleted
    3. Test          run the complete pytest suite; a failing test
                     stops the build
    4. Build         run PyInstaller with jarvis.spec
    5. Verify        check the executable and its bundled resources,
                     then run `Jarvis.exe --version`
    6. Report        print the output location, size, and what to do
                     next — or a clear failure message with the exact
                     command to run

Nothing here downloads anything, and no shell interpreter is used:
every step is a plain argument list executed with shell=False.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPEC_FILE = ROOT / "jarvis.spec"
BUILD_DIR = ROOT / "build"
DIST_DIR = ROOT / "dist"
APP_DIR = DIST_DIR / "Jarvis"
EXE_FILE = APP_DIR / "Jarvis.exe"

# Resources jarvis.spec bundles; verified after the build.
REQUIRED_RESOURCES = (
    "_internal/config/settings.json",
    "_internal/config/commands.json",
    "_internal/core/spell_model.json",
    "_internal/nltk_data/tokenizers/punkt_tab/english",
)

BUILD_TIMEOUT = 900   # seconds for the PyInstaller run
VERIFY_TIMEOUT = 60   # seconds for the verification runs


class BuildError(Exception):
    """A build step failed; the message is shown to the user."""


def _print_step(number: int, title: str) -> None:
    print()
    print(f"=== Step {number}: {title} ===", flush=True)


def _run(cmd: list[str], label: str, timeout: int = VERIFY_TIMEOUT) -> subprocess.CompletedProcess:
    """Run one command without a shell and return its result."""
    print(f"  $ {' '.join(str(part) for part in cmd)}", flush=True)
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise BuildError(f"{label}: could not start the command ({exc})") from exc
    except subprocess.TimeoutExpired as exc:
        raise BuildError(f"{label}: timed out after {timeout}s") from exc
    return proc


def step_preflight() -> None:
    """Confirm this machine can produce the Windows build."""
    if sys.platform != "win32":
        raise BuildError(
            "This build script targets Windows (win32); "
            f"current platform is {sys.platform}."
        )
    if not SPEC_FILE.is_file():
        raise BuildError(f"Missing build specification: {SPEC_FILE}")
    try:
        import PyInstaller  # noqa: F401
    except ImportError as exc:
        raise BuildError(
            "PyInstaller is not installed.\n"
            "  Install it with:  python -m pip install \"pyinstaller>=6\"\n"
            "  Then run again:   python build.py"
        ) from exc
    print(f"  Python {sys.version.split()[0]}, PyInstaller available")


def step_clean() -> None:
    """Remove only the directories this build owns."""
    for directory in (BUILD_DIR, DIST_DIR):
        if directory.parent.resolve() != ROOT.resolve():
            raise BuildError(f"Refusing to clean unexpected path: {directory}")
        if directory.exists():
            shutil.rmtree(directory, ignore_errors=True)
            print(f"  removed {directory.relative_to(ROOT)}/")
    print("  build/ and dist/ are clean")


def step_test() -> None:
    """Run the complete test suite; any failure stops the build."""
    proc = _run(
        [sys.executable, "-m", "pytest", "tests/", "-q"],
        "Test suite",
        timeout=1800,
    )
    tail = "\n".join((proc.stdout or "").strip().splitlines()[-3:])
    if tail:
        print(f"  {tail}")
    if proc.returncode != 0:
        tail_err = (proc.stderr or "")[-3000:]
        raise BuildError(
            "Tests failed — the build was stopped.\n"
            f"{tail_err}"
        )


def step_build() -> None:
    """Run PyInstaller with the project spec file."""
    proc = _run(
        [sys.executable, "-m", "PyInstaller", "--noconfirm", str(SPEC_FILE)],
        "PyInstaller",
        timeout=BUILD_TIMEOUT,
    )
    if proc.returncode != 0:
        # Show the most useful part of the PyInstaller log.
        output = (proc.stdout or "") + "\n" + (proc.stderr or "")
        raise BuildError(
            "PyInstaller failed:\n" + output[-6000:]
        )
    warnings = [
        line
        for line in (proc.stdout or "").splitlines()
        if line.startswith("WARNING:")
    ]
    if warnings:
        print(f"  {len(warnings)} PyInstaller warning(s) (see build log)")


def step_verify() -> list[str]:
    """Check the executable, its resources, and --version."""
    problems: list[str] = []

    if not EXE_FILE.is_file():
        raise BuildError(
            f"Expected executable not found: {EXE_FILE}\n"
            "The build did not produce the expected output."
        )

    for relative in REQUIRED_RESOURCES:
        target = APP_DIR / relative
        if not target.exists():
            problems.append(f"missing bundled resource: {relative}")

    proc = _run([str(EXE_FILE), "--version"], "Verification (--version)")
    output = ((proc.stdout or "") + (proc.stderr or "")).strip()
    if proc.returncode != 0:
        problems.append(
            f"--version exited with code {proc.returncode}: {output[-500:]}"
        )
    elif "Jarvis" not in output:
        problems.append(f"--version printed an unexpected value: {output[:200]}")
    else:
        print(f"  {output.splitlines()[0]}")

    if problems:
        raise BuildError("Verification failed:\n  - " + "\n  - ".join(problems))


def step_report(elapsed: float) -> None:
    """Print where everything ended up and how to use it."""
    size = sum(
        f.stat().st_size for f in APP_DIR.rglob("*") if f.is_file()
    )
    print(f"  Output       : {EXE_FILE}")
    print(f"  Size         : {size / (1024 * 1024):.1f} MB")
    print(f"  Build time   : {elapsed:.1f} s")
    print()
    print("Next steps:")
    print(f'  Run packaged : "{EXE_FILE}"')
    print("  Validate     : see docs/VALIDATION.md (repeatable checklist)")
    print("  Cleanup      : deleting the dist/Jarvis folder removes the app")
    print("                 (user data in %APPDATA%\\Jarvis is kept)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the Jarvis Windows executable."
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="skip the test suite (faster iteration; not for releases)",
    )
    args = parser.parse_args(argv)

    started = time.monotonic()
    try:
        _print_step(1, "Preflight")
        step_preflight()

        _print_step(2, "Clean")
        step_clean()

        if args.skip_tests:
            print("  skipped (--skip-tests)")
        else:
            _print_step(3, "Test")
            step_test()

        _print_step(4, "Build")
        step_build()

        _print_step(5, "Verify")
        step_verify()

        _print_step(6, "Report")
        step_report(time.monotonic() - started)
    except BuildError as exc:
        print()
        print("BUILD FAILED")
        print(str(exc))
        return 1

    print()
    print("BUILD SUCCESSFUL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
