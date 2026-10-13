"""
Custom Exceptions Module

Lightweight exception hierarchy for Jarvis.

These exceptions are reserved for genuinely exceptional conditions.
Normal failures (unknown command, application not installed, file not
found, ...) stay structured results:

    {"success": False, "message": "...", "data": {}}

Architecture:
    managers return structured results for expected failures
    JarvisError and its subclasses describe unexpected failures
    the Executor logs them and returns a safe user-facing result
"""


class JarvisError(Exception):
    """Base class for all Jarvis-specific exceptions."""


class CommandError(JarvisError):
    """A command could not be executed."""


class ValidationError(JarvisError):
    """Command validation failed in an exceptional way."""


class ApplicationError(JarvisError):
    """An application operation failed unexpectedly."""


class FileOperationError(JarvisError):
    """A file operation failed unexpectedly."""


class BrowserError(JarvisError):
    """A browser operation failed unexpectedly."""


class SystemOperationError(JarvisError):
    """A system operation failed unexpectedly."""


class ConfigurationError(JarvisError):
    """Configuration could not be read or written."""


class MultiCommandError(JarvisError):
    """
    Multi-command input violated a safety limit.

    Raised by the Multi-Command Parser when an input contains too
    many command segments or a segment is too long. The exception
    message is user-safe and is displayed verbatim — it never
    contains internal details.
    """
