"""
Custom Exceptions Module

Lightweight exception hierarchy for TITAN.

These exceptions are reserved for genuinely exceptional conditions.
Normal failures (unknown command, application not installed, file not
found, ...) stay structured results:

    {"success": False, "message": "...", "data": {}}

Architecture:
    managers return structured results for expected failures
    TITANError and its subclasses describe unexpected failures
    the Executor logs them and returns a safe user-facing result
"""


class TITANError(Exception):
    """Base class for all TITAN-specific exceptions."""


class CommandError(TITANError):
    """A command could not be executed."""


class ValidationError(TITANError):
    """Command validation failed in an exceptional way."""


class ApplicationError(TITANError):
    """An application operation failed unexpectedly."""


class FileOperationError(TITANError):
    """A file operation failed unexpectedly."""


class BrowserError(TITANError):
    """A browser operation failed unexpectedly."""


class SystemOperationError(TITANError):
    """A system operation failed unexpectedly."""


class ConfigurationError(TITANError):
    """Configuration could not be read or written."""


class MultiCommandError(TITANError):
    """
    Multi-command input violated a safety limit.

    Raised by the Multi-Command Parser when an input contains too
    many command segments or a segment is too long. The exception
    message is user-safe and is displayed verbatim — it never
    contains internal details.
    """
