"""Quick manual pipeline probe for Step 23 (not part of the test suite)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.tokenizer import tokenize
from core.spell_checker import correct_tokens
from core.intent import detect_intent
from core.command_parser import parse_command
from core.command_validator import validate_command
from core.command_registry import get_all_commands
from core.executor import CONFIRMATION_INTENTS
from core.multi_command_parser import split_commands, COMMAND_START_WORDS

PHRASES = [
    "enable startup",
    "start with windows",
    "enable windows startup",
    "disable startup",
    "stop startup",
    "disable windows startup",
    "startup status",
    "check startup",
    "is startup enabled",
    "startup",
    "enable startup c:\\evil.exe",
    "start with windows media player",
    "open chrome",
    "shutdown computer",
]

for phrase in PHRASES:
    tokens = tokenize(phrase)
    corrected = correct_tokens(tokens)
    intent = detect_intent(corrected)
    command = parse_command(corrected, intent)
    validation = validate_command(command)
    print(
        f"{phrase!r:34} -> {corrected} -> {intent.value:16} "
        f"target={command['target']!r} valid={validation['valid']}"
    )

print("commands:", len(get_all_commands()))
print("confirmation intents:", sorted(CONFIRMATION_INTENTS))
print("start words:", sorted(COMMAND_START_WORDS))
print("split:", split_commands("enable startup and open chrome"))
print("split2:", split_commands("help and startup status"))
print("split3:", split_commands("open chrome and start with windows"))
