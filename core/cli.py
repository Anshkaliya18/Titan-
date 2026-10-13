"""
Command-Line Flags (Step 23 — Phase 21)

A deliberately tiny flag handler — no CLI framework.

Handled before logging and the managers initialize (see main.py), so:

    Jarvis.exe --version    print the version and exit
    Jarvis.exe --help       print short usage and exit

Both print to stdout and return exit code 0 without starting the
interactive loop, configuring logging, or touching the registry, the
filesystem, or NLTK.

This is unrelated to the conversational ``help`` command inside the
Jarvis prompt, which keeps working exactly as before.
"""

from __future__ import annotations

import sys

from core.version import version_string

USAGE = f"""{version_string()}

Usage: Jarvis [options]

Options:
  --version   Print the version and exit
  --help, -h  Show this help and exit

Run Jarvis without options to start the interactive assistant.
Once started, type "help" for command help.
"""

# Flags this handler understands (checked against argv[1:]).
VERSION_FLAGS = ("--version",)
HELP_FLAGS = ("--help", "-h")


def handle_cli_flags(argv: list[str] | None = None) -> int | None:
    """
    Handle lightweight command-line flags.

    Args:
        argv: Argument list to inspect. Defaults to sys.argv. The
            program name (argv[0]) is never interpreted as a flag.

    Returns:
        The process exit code (0) when a flag was handled, or None
        when normal startup should continue.

    Side effects:
        Prints to stdout. Reads nothing, writes nothing, starts no
        manager — the caller exits before the application initializes.
    """
    args = list(sys.argv if argv is None else argv)[1:]

    if any(flag in args for flag in VERSION_FLAGS):
        print(version_string())
        return 0

    if any(flag in args for flag in HELP_FLAGS):
        print(USAGE)
        return 0

    return None
