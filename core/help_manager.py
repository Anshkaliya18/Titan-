"""
Help Manager Module

Generates help information from the Command Registry.

This module does NOT execute commands. It only displays information.

Architecture:
    Command Registry
           ↓
    Help Manager
           ↓
    Help Output

The Help Manager reads all command metadata from the Command Registry,
so new commands registered there are automatically discoverable.
"""

from core.command_registry import get_all_commands, get_command
from utils.logger import get_logger

logger = get_logger(__name__)

# Category display order and labels for the general help output
_CATEGORY_ORDER = [
    ("application", "Applications"),
    ("file", "Files"),
    ("system", "System"),
    ("browser", "Browser"),
    ("help", "Other"),
]

# Safe example commands shown at the end of general help
_HELP_EXAMPLES = (
    '  "open chrome"',
    '  "search web python tutorials"',
    '  "lock computer"',
)


def get_help() -> dict:
    """
    Generate general help information.

    Groups the registered commands by category and appends a few
    safe example commands.

    Returns:
        dict: {success, message, data} with data.type == "help".
    """
    commands = get_all_commands()

    lines = ["TITAN can help you with:", ""]

    for category_key, category_label in _CATEGORY_ORDER:
        category_commands = [
            cmd for cmd in commands.values() if cmd.category == category_key
        ]
        if not category_commands:
            continue
        lines.append(category_label)
        for cmd in category_commands:
            example = cmd.aliases[0] if cmd.aliases else cmd.intent.lower()
            lines.append(f"  {example}")
        lines.append("")

    lines.append("Try:")
    lines.extend(_HELP_EXAMPLES)

    message = "\n".join(lines)
    logger.info("General help generated")
    return {
        "success": True,
        "message": message,
        "data": {"type": "help"},
    }


def get_command_list() -> dict:
    """
    Generate a concise list of available commands.

    The list is generated from the Command Registry — no separate
    command list is maintained here.

    Returns:
        dict: {success, message, data} with data.type == "command_list".
    """
    commands = get_all_commands()

    lines = ["Available commands:", ""]
    for cmd in commands.values():
        if cmd.intent == "UNKNOWN":
            continue
        lines.append(f"- {cmd.description}")

    message = "\n".join(lines)
    logger.info("Command list generated")
    return {
        "success": True,
        "message": message,
        "data": {"type": "command_list"},
    }


def get_command_help(intent: str) -> dict:
    """
    Generate help for a specific command.

    Uses the registry's intent, description, and aliases to build the
    output. Unknown intents produce a safe fallback — no guessing.

    Args:
        intent: The intent name to look up, e.g. "OPEN_APPLICATION".

    Returns:
        dict: {success, message, data} with data.type == "command_help".
    """
    command = get_command(intent)

    if command is None:
        logger.warning(f"Command help requested for unknown intent: {intent}")
        return {
            "success": False,
            "message": 'Unknown command.\n\nType "commands" to see available commands.',
            "data": {"type": "command_help"},
        }

    title = command.intent.title().replace("_", " ")
    lines = [title, "", command.description, ""]

    if command.aliases:
        lines.append("Aliases:")
        lines.extend(f"  {alias}" for alias in command.aliases)
        lines.append("")

    lines.append("Examples:")
    # Show up to three aliases as safe example invocations
    for alias in command.aliases[:3]:
        lines.append(f"  {alias}")

    message = "\n".join(lines)
    logger.info(f"Command help generated: {command.intent}")
    return {
        "success": True,
        "message": message,
        "data": {"type": "command_help"},
    }
