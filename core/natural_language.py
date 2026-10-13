"""
Natural Language Command Layer (Step 27)

Deterministic natural-language interpretation for Jarvis.

Converts natural-language requests into structured command interpretations
that feed into the existing hybrid intent detection pipeline.

Architecture:
    USER INPUT
        ↓
    Input Processor
        ↓
    Normalize
        ↓
    Tokenizer
        ↓
    Spell Correction
        ↓
    Natural Language Interpretation (this module)
        ↓
    Hybrid Intent Detection
        ↓
    Command Parser
        ↓
    Command Validator
        ↓
    Executor
        ↓
    Manager
        ↓
    Response

This module is strictly an interpretation component. It never executes
commands, calls the executor, launches applications, modifies files,
accesses the shell, or bypasses validation/confirmation.

Design principles:
- Deterministic: same input always produces same output
- Explainable: every interpretation has a confidence score and reason
- Local: no network access, no external APIs
- Security-first: never generates arbitrary commands, only maps to
  registered intents in the Command Registry
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from core.command_registry import is_valid_intent
from core.application_manager import APP_NAMES, is_application
from core.config_manager import get_value
_APPLICATION_ALIASES = tuple(sorted(APP_NAMES, key=len, reverse=True))


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class EntityType(Enum):
    """Types of entities that can be extracted from natural language."""
    APPLICATION = "application"
    FILE = "file"
    FILE_NAME = "file_name"
    FOLDER = "folder"
    URL = "url"
    SEARCH_QUERY = "search_query"
    SYSTEM_TARGET = "system_target"
    UNKNOWN = "unknown"


class TokenType(Enum):
    """Classification of tokens for context-aware spell correction."""
    COMMAND_WORD = "command_word"
    APPLICATION_NAME = "application_name"
    APPLICATION_PHRASE = "application_phrase"
    SYSTEM_WORD = "system_word"
    NORMAL_WORD = "normal_word"
    SEARCH_QUERY = "search_query"
    URL = "url"
    WINDOWS_PATH = "windows_path"
    FILE_EXTENSION = "file_extension"
    FILE_NAME = "file_name"
    TECHNICAL_TERM = "technical_term"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Entity:
    """An extracted entity from natural language."""
    text: str
    entity_type: EntityType
    confidence: float = 1.0
    start_index: int = -1
    end_index: int = -1


@dataclass
class InterpretationResult:
    """Result of natural-language interpretation."""
    success: bool
    intent: str
    target: Optional[str] = None
    arguments: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    reason: str = ""
    entities: List[Entity] = field(default_factory=list)
    original_text: str = ""
    normalized_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/debugging."""
        return {
            "success": self.success,
            "intent": self.intent,
            "target": self.target,
            "arguments": self.arguments,
            "confidence": self.confidence,
            "reason": self.reason,
            "entities": [
                {
                    "text": e.text,
                    "type": e.entity_type.value,
                    "confidence": e.confidence,
                }
                for e in self.entities
            ],
            "original_text": self.original_text,
            "normalized_text": self.normalized_text,
        }


# =============================================================================
# FILLER PHRASE REMOVAL
# =============================================================================

class FillerRemover:
    """Removes conversational filler phrases from natural language input."""

    # Filler patterns (regex)
    FILLER_PATTERNS: List[Tuple[re.Pattern, str]] = [
        (re.compile(r"^(?:please|kindly)\s+", re.IGNORECASE), ""),
        # Leading filler: "Could you please ..." -> "..."
        (re.compile(r"^(?:can|could|would|will) you (?:please )?", re.IGNORECASE), ""),
        (re.compile(r"^(?:can|could|would|will) you kindly ", re.IGNORECASE), ""),
        # "I want you to ..." -> "..."
        (re.compile(r"^i (?:would )?like you to ", re.IGNORECASE), ""),
        (re.compile(r"^i want you to ", re.IGNORECASE), ""),
        (re.compile(r"^i need you to ", re.IGNORECASE), ""),
        # "Do you mind ..."
        (re.compile(r"^do you mind ", re.IGNORECASE), ""),
        # Trailing filler
        (re.compile(r" for me$", re.IGNORECASE), ""),
        (re.compile(r" for us$", re.IGNORECASE), ""),
        (re.compile(r" if you could$", re.IGNORECASE), ""),
        (re.compile(r" if you would$", re.IGNORECASE), ""),
        (re.compile(r" if possible$", re.IGNORECASE), ""),
        (re.compile(r" if you don't mind$", re.IGNORECASE), ""),
    ]

    def remove_fillers(self, text: str) -> str:
        """Remove filler phrases from text."""
        result = text.strip()

        # Apply regex patterns
        for pattern, replacement in self.FILLER_PATTERNS:
            result = pattern.sub(replacement, result)

        # Clean up extra whitespace
        result = " ".join(result.split())

        return result.strip()


# =============================================================================
# SYNONYM MAPPING
# =============================================================================

class SynonymMapper:
    """Maps synonyms to known command concepts."""

    # Synonym mappings: synonym -> canonical concept
    SYNONYMS: Dict[str, str] = {
        # Open synonyms
        "launch": "open",
        "start": "open",
        "run": "open",
        "bring up": "open",
        "fire up": "open",
        # Close synonyms
        "quit": "close",
        "exit": "close",
        "stop": "close",
        "shut": "close",
        # Search synonyms
        "look up": "search",
        "find online": "search",
        "google": "search",
        "search for": "search",
        "search online": "search",
        # Show synonyms
        "display": "show",
        "list": "list",
        "show me": "show",
        # Create synonyms
        "make": "create",
        # Delete synonyms
        "remove": "delete",
        # Restart synonyms
        "reboot": "restart",
        # Shutdown synonyms
        "power off": "shutdown",
        "turn off": "shutdown",
        "shut down": "shutdown",
    }

    def map_synonyms(self, text: str) -> str:
        """Map synonyms to canonical command concepts."""
        result = text.lower()

        # The bare word "google" is a search verb only as a leading
        # action. It remains part of registered application aliases such
        # as "google chrome" when following an open/launch command.
        starts_with_google_query = bool(re.match(r"^google\s+\S", result))

        # Sort by length descending to match longer phrases first
        for synonym in sorted(self.SYNONYMS.keys(), key=len, reverse=True):
            if synonym == "google":
                continue
            canonical = self.SYNONYMS[synonym]
            # Replace whole-word synonyms
            pattern = re.compile(r"\b" + re.escape(synonym) + r"\b", re.IGNORECASE)
            result = pattern.sub(canonical, result)

        if starts_with_google_query:
            result = re.sub(r"^google\b", "search", result)

        return result


# =============================================================================
# ENTITY EXTRACTION
# =============================================================================

class EntityExtractor:
    """Extracts entities (applications, files, URLs, search queries) from text."""

    def __init__(self):
        self.app_names = APP_NAMES

    def extract_url(self, text: str) -> Optional[str]:
        """Extract URL from text."""
        # URL pattern
        url_pattern = re.compile(
            r"https?://[^\s]+",
            re.IGNORECASE,
        )
        match = url_pattern.search(text)
        if match:
            return match.group(0)
        return None

    def extract_application(self, text: str) -> Optional[str]:
        """Extract application name from text."""
        text_lower = text.lower()

        # Check for multi-word application names first
        for app_name in _APPLICATION_ALIASES:
            if app_name in text_lower:
                return app_name

        return None

    def extract_search_query(self, text: str) -> Optional[str]:
        """Extract search query from text."""
        # Remove command words
        query = text.lower()
        command_words = {"search", "web", "for", "find", "look", "up", "online", "google", "the"}
        words = query.split()
        filtered = [w for w in words if w not in command_words]
        return " ".join(filtered) if filtered else None

    def extract_file_name(self, text: str) -> Optional[str]:
        """Extract file name from text."""
        # Look for file extensions
        file_pattern = re.compile(
            r"\b[\w\-.]+\.(?:txt|py|json|yaml|yml|xml|html?|css|js|ts|tsx|jsx|exe|dll|so|dylib|app|docx?|pdf|xlsx?|csv|md|rst|log|ini|cfg|conf|config|toml|bat|ps1|sh|bash|zsh|fish)\b",
            re.IGNORECASE,
        )
        match = file_pattern.search(text)
        if match:
            return match.group(0)
        return None

    def extract_folder_name(self, text: str) -> Optional[str]:
        """Extract folder name from text."""
        # Look for "folder named X" or "folder called X" patterns
        patterns = [
            re.compile(r"folder (?:named|called) ([\w\-]+)", re.IGNORECASE),
            re.compile(r"create (?:a )?folder ([\w\-]+)", re.IGNORECASE),
            re.compile(r"make (?:a )?folder (?:named|called) ([\w\-]+)", re.IGNORECASE),
        ]
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                return match.group(1)
        return None


# =============================================================================
# CONFIDENCE SCORING
# =============================================================================

class ConfidenceScorer:
    """Scores confidence of natural-language interpretations."""

    def __init__(self):
        self.base_threshold = self._threshold("confidence_threshold", 0.75)
        self.dangerous_threshold = self._threshold("dangerous_confidence_threshold", 0.90)

    @staticmethod
    def _threshold(key: str, default: float) -> float:
        value = get_value("natural_language", key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return default
        return float(value) if 0.0 <= value <= 1.0 else default

    def score_intent(self, text: str, intent: str, entities: List[Entity]) -> float:
        """
        Score confidence for an intent interpretation.

        Factors:
        - Presence of clear command keywords
        - Entity extraction success
        - Text length and complexity
        - Pattern match strength
        """
        score = 0.5  # Base score

        # Boost for clear command keywords
        command_keywords = {
            "open": 0.15,
            "close": 0.15,
            "launch": 0.15,
            "start": 0.10,
            "run": 0.10,
            "search": 0.15,
            "find": 0.10,
            "show": 0.10,
            "list": 0.10,
            "create": 0.15,
            "delete": 0.15,
            "remove": 0.10,
            "lock": 0.15,
            "shutdown": 0.15,
            "restart": 0.15,
            "reboot": 0.15,
            "help": 0.15,
            "history": 0.15,
            "clear": 0.15,
        }

        text_lower = text.lower()
        for keyword, boost in command_keywords.items():
            if keyword in text_lower:
                score += boost
                break  # Only count first matching keyword

        # Boost for successful entity extraction
        if entities:
            score += 0.1 * len(entities)

        # Boost for reasonable text length
        word_count = len(text.split())
        if 2 <= word_count <= 10:
            score += 0.05

        # Cap at 1.0
        return min(score, 1.0)

    def get_threshold(self, intent: str) -> float:
        """Get confidence threshold for an intent."""
        dangerous_intents = {
            "SHUTDOWN_COMPUTER",
            "RESTART_COMPUTER",
            "DELETE_FILE",
            "ENABLE_STARTUP",
            "DISABLE_STARTUP",
        }
        if intent in dangerous_intents:
            return self.dangerous_threshold
        return self.base_threshold


# =============================================================================
# NATURAL LANGUAGE INTERPRETER
# =============================================================================

class NaturalLanguageInterpreter:
    """
    Main entry point for natural-language command interpretation.

    Converts natural-language requests into structured interpretations
    that feed into the existing hybrid intent detection pipeline.
    """

    _FILE_RE = re.compile(
        r"(?<![\w])([\w .-]+\.(?:txt|py|json|ya?ml|xml|html?|css|js|ts|"
        r"exe|dll|docx?|pdf|xlsx?|csv|md|log|ini|cfg|toml|bat|ps1|sh))\b",
        re.IGNORECASE,
    )
    _URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)

    def __init__(self):
        self.filler_remover = FillerRemover()
        self.synonym_mapper = SynonymMapper()
        self.entity_extractor = EntityExtractor()
        self.confidence_scorer = ConfidenceScorer()

    @staticmethod
    def _clean_entity(value: str) -> str:
        return value.strip().strip(" \t\r\n.,!?;:")

    @staticmethod
    def _canonical(intent: str, target: Optional[str]) -> str:
        prefixes = {
            "OPEN_APPLICATION": "open", "CLOSE_APPLICATION": "close",
            "OPEN_FILE": "open", "DELETE_FILE": "delete",
            "CREATE_FOLDER": "create folder", "SEARCH_FILE": "find",
            "SEARCH_WEB": "search web", "OPEN_URL": "open",
        }
        fixed = {
            "OPEN_BROWSER": "open browser", "LOCK_COMPUTER": "lock computer",
            "SHUTDOWN_COMPUTER": "shutdown computer", "RESTART_COMPUTER": "restart computer",
            "HELP": "help", "LIST_COMMANDS": "commands", "HISTORY": "history",
            "CLEAR_HISTORY": "clear history", "STARTUP_STATUS": "startup status",
            "FIND_APPLICATION": "find installed apps",
            "REFRESH_APPLICATIONS": "refresh applications",
        }
        if intent in fixed:
            return fixed[intent]
        return f"{prefixes.get(intent, '')} {target or ''}".strip()

    def interpret(self, text: str, tokens: List[str] = None) -> InterpretationResult:
        original = text if isinstance(text, str) else ""
        if not isinstance(text, str):
            return self._unknown(original, "Input must be text.")
        enabled = get_value("natural_language", "enabled", True)
        if enabled is False:
            return self._unknown(original, "Natural-language interpretation is disabled.")

        # Work on a case-preserving string so search and file entities retain
        # their user spelling. Only matching uses a lowercase copy.
        filler_input = text.strip()
        if not re.search(r"https?://[^\s]+$", filler_input, re.IGNORECASE):
            filler_input = filler_input.rstrip("?!.,;:")
        cleaned = self.filler_remover.remove_fillers(filler_input)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        matching = cleaned.rstrip(" \t\r\n?!").strip()
        lower = matching.lower()
        entities: List[Entity] = []

        def result(intent: str, target: Optional[str] = None, *, query: bool = False,
                   entity_type: Optional[EntityType] = None, confidence: float = 0.96):
            if not is_valid_intent(intent):
                return self._unknown(original, "Interpretation is not registered.", matching)
            target = self._clean_entity(target) if target else None
            if target and entity_type:
                entities.append(Entity(target, entity_type, confidence))
            args = {"query": target} if query and target else {}
            if intent in {"OPEN_APPLICATION", "CLOSE_APPLICATION"} and target:
                if not is_application(target):
                    return self._unknown(original, "Application is not registered.", matching)
            if intent in {"OPEN_FILE", "DELETE_FILE", "SEARCH_FILE"} and not target:
                return self._unknown(original, "A file target is required.", matching)
            if intent == "CREATE_FOLDER" and not target:
                return self._unknown(original, "A folder name is required.", matching)
            if intent == "SEARCH_WEB" and not target:
                return self._unknown(original, "A search query is required.", matching)
            threshold = self.confidence_scorer.get_threshold(intent)
            if confidence < threshold:
                return self._unknown(original, "Confidence is below the configured threshold.", matching, confidence)
            canonical = self._canonical(intent, target)
            return InterpretationResult(True, intent, target, args, confidence,
                "Matched a deterministic command pattern.", list(entities), original, canonical)

        if re.fullmatch(r"refresh (?:installed )?(?:apps|applications)(?: list)?", lower):
            return result("REFRESH_APPLICATIONS")
        if lower in {"find installed apps", "find installed applications",
                     "search installed apps", "search installed applications",
                     "what apps do i have", "what applications do i have"}:
            return result("FIND_APPLICATION")
        app_search = re.match(
            r"^(?:find|search)(?: installed)? (?:apps?|applications?)(?: containing)?\s+(.+)$",
            matching, re.IGNORECASE,
        )
        if app_search:
            return result("FIND_APPLICATION", self._clean_entity(app_search.group(1)))
        simple_app_search = re.match(r"^find\s+(.+)$", matching, re.IGNORECASE)
        if (simple_app_search and not self._FILE_RE.search(simple_app_search.group(1))
                and not re.match(r"^(?:online|on the web|on web)\b", simple_app_search.group(1), re.I)):
            return result("FIND_APPLICATION", self._clean_entity(simple_app_search.group(1)))

        # Exact application aliases take priority over the `google` search synonym.
        app_alias = next((name for name in _APPLICATION_ALIASES
                          if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", lower)), None)
        file_match = self._FILE_RE.search(matching)
        url_match = self._URL_RE.search(matching)

        # Clear/list/history/help commands before generic "show" patterns.
        if re.search(r"\b(clear|delete|remove)\b.*\b(history|recent commands)\b", lower):
            return result("CLEAR_HISTORY")
        if re.search(r"\b(history|recent commands|command history)\b", lower) and re.search(r"\b(show|view|what|recent|my|command)\b", lower):
            return result("HISTORY")
        if re.search(r"\b(commands|what commands|list commands)\b", lower) and re.search(r"\b(show|list|what|available|support|perform|do)\b", lower):
            return result("LIST_COMMANDS")
        if re.fullmatch(r"(?:what can you do|how can you help(?: me)?|show me what you can do|help|show help)", lower):
            return result("HELP")

        # Searches require an explicit search phrase or the unambiguous
        # "google <query>" form. Queries are retained verbatim.
        search_patterns = (
            r"^(?:search(?:\s+the)?\s+(?:web|online)(?:\s+for)?|search\s+for|"
            r"look\s+up|find\s+online|google)\s+(.+)$",
        )
        for pattern in search_patterns:
            match = re.match(pattern, matching, re.IGNORECASE)
            if match:
                query = self._clean_entity(match.group(1))
                # A file locator is handled by the file command patterns
                # below, not interpreted as a web query.
                if file_match:
                    file_target = re.sub(
                        r"^(?:(?:open|delete|remove|find|search(?:\s+for)?|look for)\s+)?file\s+|^(?:open|delete|remove|find|search(?:\s+for)?|look for)\s+",
                        "", self._clean_entity(file_match.group(1)), flags=re.IGNORECASE,
                    )
                    filename_token = file_target.split()[-1] if file_target.split() else ""
                    if (
                        filename_token
                        and query.casefold().endswith(filename_token.casefold())
                        and not re.search(r"\b(web|online)\b", lower)
                    ):
                        continue
                if app_alias and lower.startswith(("open ", "launch ", "start ", "run ", "close ", "quit ")):
                    break
                return result("SEARCH_WEB", query, query=True, entity_type=EntityType.SEARCH_QUERY, confidence=0.95)

        # System actions use explicit, bounded patterns. Vague statements
        # never become shutdown/restart/delete commands.
        if re.search(r"\b(lock)\b(?:\s+my|\s+the|\s+this)?\s+(?:computer|pc)\b", lower):
            return result("LOCK_COMPUTER")
        if re.search(r"\b(shutdown|shut down|turn off|power off)\b.*\b(computer|pc|my computer|my pc)\b", lower) or re.search(r"\bshut\s+my\s+(?:computer|pc)\s+down\b", lower):
            return result("SHUTDOWN_COMPUTER", confidence=0.96)
        if re.search(r"\b(restart|reboot)\b.*\b(computer|pc|my computer|my pc)\b", lower):
            return result("RESTART_COMPUTER", confidence=0.96)

        # Folder creation and file actions extract only the entity span.
        folder = re.search(r"\b(?:create|make)\s+(?:a\s+)?folder\s+(?:(?:named|called)\s+)?(.+)$", matching, re.IGNORECASE)
        if folder:
            name = self._clean_entity(folder.group(1))
            name = re.sub(r"\s+(?:for me|please)$", "", name, flags=re.IGNORECASE).strip()
            if name:
                return result("CREATE_FOLDER", name, entity_type=EntityType.FOLDER)

        if file_match:
            filename = self._clean_entity(file_match.group(1))
            filename = re.sub(
                r"^(?:(?:open|delete|remove|find|search(?:\s+for)?|look for)\s+)?file\s+|^(?:open|delete|remove|find|search(?:\s+for)?|look for)\s+",
                "", filename, flags=re.IGNORECASE,
            )
            if re.search(r"\b(delete|remove)\b", lower):
                return result("DELETE_FILE", filename, entity_type=EntityType.FILE, confidence=0.96)
            if re.search(r"\b(find|search|look for)\b", lower):
                return result("SEARCH_FILE", filename, entity_type=EntityType.FILE)
            if re.search(r"\b(open|launch)\b", lower):
                return result("OPEN_FILE", filename, entity_type=EntityType.FILE)
        if re.search(r"\b(find|search for|look for)\b", lower):
            target_match = re.search(r"\b(?:find|search for|look for)\s+(?:my\s+|the\s+)?(?:file\s+(?:called\s+)?)?(.+)$", matching, re.IGNORECASE)
            if target_match and not re.search(r"\b(online|web)\b", lower):
                target = self._clean_entity(target_match.group(1))
                if target:
                    return result("SEARCH_FILE", target, entity_type=EntityType.FILE)

        # Application/browser/URL actions validate app aliases through the
        # Application Manager, preserving multi-word aliases as entities.
        action = re.match(r"^(open|launch|start|run|bring up|fire up|close|quit|exit|stop)\s+(.+)$", matching, re.IGNORECASE)
        if action:
            verb, rest = action.group(1).lower(), self._clean_entity(action.group(2))
            rest_lower = rest.lower()
            if re.fullmatch(r"(?:(?:my|the|default)\s+)?browser", rest_lower):
                if verb in {"close", "quit", "exit", "stop"}:
                    return self._unknown(original, "Closing the default browser is not a registered command.", matching)
                return result("OPEN_BROWSER")
            if url_match and verb not in {"close", "quit", "exit", "stop"}:
                url = self._clean_entity(url_match.group(0))
                entities.append(Entity(url, EntityType.URL, 1.0))
                return result("OPEN_URL", url, entity_type=None)
            if (app_alias and rest_lower == app_alias) or is_application(rest):
                if is_application(rest):
                    intent = "CLOSE_APPLICATION" if verb in {"close", "quit", "exit", "stop"} else "OPEN_APPLICATION"
                    return result(intent, app_alias if app_alias and rest_lower == app_alias else rest,
                                  entity_type=EntityType.APPLICATION)
            if file_match and verb in {"open", "launch"}:
                return result("OPEN_FILE", self._clean_entity(file_match.group(1)), entity_type=EntityType.FILE)

        if tokens:
            corrected_text = " ".join(tokens)
            if corrected_text.casefold() != original.casefold():
                corrected_result = self.interpret(corrected_text)
                if corrected_result.success:
                    corrected_result.original_text = original
                    return corrected_result
        return self._unknown(original, "No supported deterministic pattern matched.", matching)

    def _unknown(self, original: str, reason: str, normalized: str = "", confidence: float = 0.0) -> InterpretationResult:
        return InterpretationResult(False, "UNKNOWN", None, {}, confidence, reason, [], original, normalized)

    # Compatibility methods retained for existing callers/tests.
    def _extract_entities(self, text: str, tokens: List[str]) -> List[Entity]:
        result = self.interpret(text, tokens)
        return result.entities

    def _determine_intent(self, text: str, entities: List[Entity]) -> str:
        return self.interpret(text).intent

    def _extract_target(self, text: str, intent: str, entities: List[Entity]) -> Optional[str]:
        result = self.interpret(text)
        return result.target if result.intent == intent else None

    def _build_arguments(self, text: str, intent: str, entities: List[Entity]) -> Dict[str, Any]:
        result = self.interpret(text)
        return result.arguments if result.intent == intent else {}


# =============================================================================
# PUBLIC API
# =============================================================================


def interpret_natural_language(text: str, tokens: List[str]) -> InterpretationResult:
    """
    Interpret natural-language text into a structured command interpretation.

    This is the main entry point for natural-language interpretation.

    Args:
        text: Original user text
        tokens: Corrected tokens from spell checker

    Returns:
        InterpretationResult with intent, target, arguments, and confidence
    """
    interpreter = NaturalLanguageInterpreter()
    return interpreter.interpret(text, tokens)


def normalize_natural_command(text: str) -> str:
    """
    Normalize a natural-language command to a simpler form.

    Removes filler phrases and maps synonyms to canonical concepts.

    Args:
        text: Natural-language command text

    Returns:
        Normalized command text
    """
    remover = FillerRemover()
    mapper = SynonymMapper()

    cleaned = remover.remove_fillers(text)
    normalized = mapper.map_synonyms(cleaned)

    return normalized


# Backward compatibility
def correct_tokens(tokens: List[str]) -> List[str]:
    """Backward compatible entry point."""
    from core.spell_checker import correct_tokens as spell_correct_tokens
    return spell_correct_tokens(tokens)


def correct_token(token: str) -> str:
    """Correct a single token (backward compatible)."""
    from core.spell_checker import correct_token as spell_correct_token
    return spell_correct_token(token)
