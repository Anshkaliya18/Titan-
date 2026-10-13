"""
Advanced Context-Aware Spell Checker (Step 26)

Enhances the base spell checker with:
- Token classification for context-aware corrections
- Multi-word application phrase protection
- Context-aware candidate ranking
- URL/path/extension/technical term protection
- Confidence thresholds with dangerous command handling
- Configurable via ConfigManager

Architecture:
    RAW TOKENS
        -> TOKEN CLASSIFICATION
        -> PROTECTED TOKEN DETECTION
        -> CANDIDATE GENERATION
        -> CONTEXT ANALYSIS
        -> CANDIDATE RANKING
        -> CONFIDENCE CHECK
        -> CORRECTED TOKENS

This module is strictly an interpretation component. It never executes
commands, calls the executor, launches applications, modifies files,
accesses the shell, or bypasses validation/confirmation.
"""

from __future__ import annotations

import re
import logging
from typing import List, Optional, Set, Tuple, Dict
from dataclasses import dataclass
from enum import Enum

from core.application_manager import APP_NAMES
from core.spell_model import SpellModel
from core.spell_checker import PUNCTUATION
from core.vocabulary import PROTECTED_WORDS, COMMAND_TYPOS
from core.command_registry import get_all_commands
from core.config_manager import get_value
from utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# TOKEN CLASSIFICATION
# =============================================================================

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
class TokenClassification:
    """Classification result for a token."""
    token: str
    token_type: TokenType
    is_protected: bool
    is_multi_word_app: bool = False
    phrase_index: int = -1


class TokenClassifier:
    """Classifies tokens for context-aware spell correction."""

    def __init__(self):
        self._init_vocabulary()

    def _init_vocabulary(self):
        """Initialize vocabulary sets for fast classification."""
        # Command words from registry
        commands = get_all_commands()
        self.command_keywords = set()
        self.command_aliases = set()
        for cmd_def in commands.values():
            for alias in cmd_def.aliases:
                self.command_aliases.add(alias.lower())
                first_word = alias.split()[0].lower()
                self.command_keywords.add(first_word)

        # Application names and their components
        self.app_names = set()
        self.app_words = set()
        self.multi_word_apps = []
        from core.application_manager import APPLICATIONS
        for app in APPLICATIONS:
            self.app_names.add(app.name.lower())
            for alias in app.aliases:
                self.app_names.add(alias.lower())
                words = alias.lower().split()
                for w in words:
                    self.app_words.add(w)
                if len(words) > 1:
                    self.multi_word_apps.append((alias.lower(), words))

        # Sort multi-word apps by length descending for greedy matching
        self.multi_word_apps.sort(key=lambda x: len(x[1]), reverse=True)

        # System words
        self.system_words = {
            "computer", "folder", "file", "browser",
            "web", "website", "http", "https", "url",
            "lock", "shutdown", "restart", "reboot", "power", "off",
            "create", "delete", "search", "open", "close",
            "help", "commands", "history", "clear",
            "enable", "disable", "startup", "status",
        }

        # Technical terms that should not be corrected
        self.technical_terms = {
            "python", "github", "git", "json", "yaml", "api", "http",
            "https", "cli", "cpu", "ram", "windows", "linux",
            "docker", "kubernetes", "aws", "azure", "gcp", "ssh",
            "ssl", "tls", "vpn", "dns", "dhcp", "ip", "tcp", "udp",
            "npm", "pip", "venv", "conda", "k8s",
            "tutorials", "tutorial", "docs", "documentation",
        }

        # File extensions to protect
        self.file_extensions = {
            ".txt", ".py", ".json", ".yaml", ".yml", ".xml",
            ".html", ".htm", ".css", ".js", ".ts", ".tsx", ".jsx",
            ".exe", ".dll", ".so", ".dylib", ".app",
            ".doc", ".docx", ".pdf", ".xls", ".xlsx", ".csv",
            ".md", ".rst", ".log", ".ini", ".cfg", ".conf",
            ".config", ".toml", ".bat", ".ps1", ".sh", ".bash",
            ".zsh", ".fish", ".msi", ".dmg", ".pkg", ".deb", ".rpm",
            ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
        }

        # URL schemes
        self.url_schemes = {"http://", "https://", "ftp://", "ftps://", "file://"}

    def classify_token(
        self, token: str, context: List[str] = None, index: int = 0
    ) -> TokenClassification:
        """Classify a token based on its content and context."""
        if not token or token in PUNCTUATION:
            return TokenClassification(token, TokenType.UNKNOWN, True)

        token_lower = token.lower()

        # Check URL schemes
        if any(token_lower.startswith(scheme) for scheme in
               ["http://", "https://", "ftp://", "ftps://", "file://"]):
            return TokenClassification(token, TokenType.URL, True)

        # Check for Windows paths (C:\...)
        if re.match(r'^[a-zA-Z]:[\\/]', token) or token.startswith("\\\\"):
            return TokenClassification(token, TokenType.WINDOWS_PATH, True)

        # Check for file extensions
        for ext in self.file_extensions:
            if token.lower().endswith(ext):
                return TokenClassification(token, TokenType.FILE_EXTENSION, True)

        # Check for file names with extensions
        if '.' in token and not token.startswith('.') and not token.endswith('.'):
            parts = token.split('.')
            if len(parts) == 2 and parts[1].lower() in [
                ext.lstrip('.') for ext in self.file_extensions
            ]:
                return TokenClassification(token, TokenType.FILE_NAME, True)

        token_lower = token.lower()

        # Check command keywords FIRST (before system words)
        if token_lower in self.command_keywords:
            return TokenClassification(token, TokenType.COMMAND_WORD, True)

        # Check command aliases
        if token_lower in self.command_aliases:
            return TokenClassification(token, TokenType.COMMAND_WORD, True)

        # Check application names
        if token_lower in self.app_names:
            return TokenClassification(token, TokenType.APPLICATION_NAME, True)

        # Check application words
        if token_lower in self.app_words:
            return TokenClassification(token, TokenType.APPLICATION_NAME, True)

        # Check system words (after command keywords)
        if token_lower in self.system_words:
            return TokenClassification(token, TokenType.SYSTEM_WORD, True)

        # Check technical terms
        if token_lower in self.technical_terms:
            return TokenClassification(token, TokenType.TECHNICAL_TERM, True)

        # Check if it's a search query word
        if token_lower in {"search", "web", "for", "find", "look", "google", "bing", "duckduckgo"}:
            return TokenClassification(token, TokenType.SEARCH_QUERY, False)

        # Check for file extensions in token
        for ext in self.file_extensions:
            if token_lower.endswith(ext):
                return TokenClassification(token, TokenType.FILE_EXTENSION, True)

        # Check for Windows paths
        if re.match(r'^[a-zA-Z]:[\\/]', token) or token.startswith('\\\\'):
            return TokenClassification(token, TokenType.WINDOWS_PATH, True)

        # Check for URLs
        if any(token_lower.startswith(scheme) for scheme in
               ["http://", "https://", "ftp://", "ftps://", "file://"]):
            return TokenClassification(token, TokenType.URL, True)

        # Default to normal word
        return TokenClassification(token, TokenType.NORMAL_WORD, False)


# Global classifier instance
_classifier_instance = None


def get_classifier() -> TokenClassifier:
    """Get the global token classifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = TokenClassifier()
    return _classifier_instance


# =============================================================================
# MULTI-WORD APPLICATION PHRASE DETECTION
# =============================================================================

class MultiWordAppDetector:
    """Detects and protects multi-word application phrases."""

    def __init__(self):
        self._init_multi_word_apps()

    def _init_multi_word_apps(self):
        """Load multi-word application phrases from application manager."""
        from core.application_manager import APPLICATIONS
        self.phrases = []
        for app in APPLICATIONS:
            for alias in app.aliases:
                words = alias.lower().split()
                if len(words) > 1:
                    self.phrases.append({
                        "canonical": alias.lower(),
                        "words": words,
                        "canonical_name": app.name,
                    })
        # Sort by length descending for greedy matching
        self.phrases.sort(key=lambda p: len(p["words"]), reverse=True)

    def detect_phrases(self, tokens: List[str]) -> List[Tuple[int, int, Dict]]:
        """
        Detect multi-word application phrases in token sequence.
        Returns list of (start_index, end_index, phrase_info).
        """
        tokens_lower = [t.lower() for t in tokens]
        matches = []
        i = 0
        while i < len(tokens_lower):
            matched = False
            for phrase in self.phrases:
                words = phrase["words"]
                if i + len(words) <= len(tokens_lower):
                    if tokens_lower[i:i + len(words)] == words:
                        matches.append((i, i + len(words), phrase))
                        i += len(words)
                        matched = True
                        break
            if not matched:
                i += 1
        return matches

    def is_part_of_phrase(self, tokens: List[str], index: int) -> Optional[Dict]:
        """Check if token at index is part of a multi-word app phrase."""
        tokens_lower = [t.lower() for t in tokens]
        for phrase in self.phrases:
            words = phrase["words"]
            for start in range(
                max(0, index - len(words) + 1),
                min(index + 1, len(tokens_lower) - len(words) + 1),
            ):
                if tokens_lower[start:start + len(words)] == words:
                    return phrase
        return None


# =============================================================================
# CONTEXT-AWARE CANDIDATE GENERATOR
# =============================================================================

@dataclass
class CorrectionCandidate:
    """A candidate correction with its score components."""
    original: str
    corrected: str
    edit_distance: int
    edit_similarity: float
    frequency: int
    context_score: float
    command_score: float
    application_score: float
    phrase_score: float
    frequency_score: float
    total_score: float
    confidence: float


class ContextAwareCandidateGenerator:
    """Generates correction candidates with context awareness."""

    def __init__(self, spell_model: SpellModel, classifier: TokenClassifier):
        self.spell_model = spell_model
        self.classifier = classifier
        self.multi_word_detector = MultiWordAppDetector()

    def generate_candidates(
        self,
        original: str,
        token_classification: TokenClassification,
        context: List[str],
        token_index: int,
    ) -> List[CorrectionCandidate]:
        """Generate and rank correction candidates for a token."""

        # Skip correction for protected token types
        if token_classification.is_protected:
            return [CorrectionCandidate(
                original=original,
                corrected=original,
                edit_distance=0,
                edit_similarity=1.0,
                frequency=1000000,
                context_score=1.0,
                command_score=1.0 if token_classification.token_type in [
                    TokenType.COMMAND_WORD,
                ] else 0.0,
                application_score=1.0 if token_classification.token_type in [
                    TokenType.APPLICATION_NAME,
                    TokenType.APPLICATION_PHRASE,
                ] else 0.0,
                phrase_score=1.0 if token_classification.is_multi_word_app else 0.0,
                frequency_score=1.0,
                total_score=1.0,
                confidence=1.0,
            )]

        # Check for URL, path, extension, technical term protection
        if token_classification.token_type in [
            TokenType.URL,
            TokenType.WINDOWS_PATH,
            TokenType.FILE_EXTENSION,
            TokenType.FILE_NAME,
            TokenType.TECHNICAL_TERM,
        ]:
            return [CorrectionCandidate(
                original=original,
                corrected=original,
                edit_distance=0,
                edit_similarity=1.0,
                frequency=1000000,
                context_score=1.0,
                command_score=0.0,
                application_score=0.0,
                phrase_score=0.0,
                frequency_score=1.0,
                total_score=1.0,
                confidence=1.0,
            )]

        # Get base candidates from spell model
        candidates = self.spell_model._known(self.spell_model._edits1(original))
        if not candidates:
            candidates = self.spell_model._known(self.spell_model._edits2(original))

        if not candidates:
            return [CorrectionCandidate(
                original=original,
                corrected=original,
                edit_distance=0,
                edit_similarity=1.0,
                frequency=1,
                context_score=0.5,
                command_score=0.0,
                application_score=0.0,
                phrase_score=0.0,
                frequency_score=0.0,
                total_score=0.5,
                confidence=0.0,
            )]

        # Rank candidates
        ranker = CandidateRanker(self.spell_model, get_classifier())
        ranked = ranker.rank_candidates(
            original,
            candidates,
            token_classification,
            context,
            token_index,
            self.spell_model,
        )

        return ranked


# =============================================================================
# CANDIDATE RANKING AND SCORING
# =============================================================================

class CandidateRanker:
    """Ranks correction candidates based on multiple scoring factors."""

    def __init__(self, spell_model: SpellModel, classifier: TokenClassifier):
        self.spell_model = spell_model
        self.classifier = classifier

    def rank_candidates(
        self,
        original: str,
        candidates: Set[str],
        token_classification: TokenClassification,
        context: List[str],
        token_index: int,
        spell_model: SpellModel,
    ) -> List[CorrectionCandidate]:
        """Rank candidates based on multiple scoring factors."""
        candidates_list = []

        for candidate in candidates:
            edit_dist = self._edit_distance(original, candidate)
            edit_similarity = 1.0 - (
                edit_dist / max(len(original), len(candidate))
            )

            freq = self._get_word_frequency(candidate)
            freq_score = min(freq / 1000000.0, 1.0)

            context_score = self._calculate_context_score(
                candidate, context, spell_model
            )

            command_score = self._calculate_command_score(
                context, candidate
            )

            app_score = self._calculate_application_score(candidate)

            phrase_score = self._calculate_phrase_score(
                candidate, context, spell_model
            )

            total_score = (
                edit_similarity * 0.30
                + freq_score * 0.25
                + context_score * 0.20
                + command_score * 0.10
                + app_score * 0.10
                + phrase_score * 0.05
            )

            # Confidence is derived from total score, clamped to [0, 1]
            confidence = max(0.0, min(1.0, total_score))

            candidates_list.append(
                CorrectionCandidate(
                    original=original,
                    corrected=candidate,
                    edit_distance=edit_dist,
                    edit_similarity=edit_similarity,
                    frequency=freq,
                    context_score=context_score,
                    command_score=command_score,
                    application_score=app_score,
                    phrase_score=phrase_score,
                    frequency_score=freq_score,
                    total_score=total_score,
                    confidence=confidence,
                )
            )

        candidates_list.sort(key=lambda c: c.total_score, reverse=True)
        return candidates_list

    def _edit_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein edit distance."""
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    def _get_word_frequency(self, word: str) -> int:
        """Get word frequency from spell model vocabulary."""
        return self.spell_model.vocab.get(word.lower(), 1)

    def _calculate_context_score(
        self, candidate: str, context: List[str], spell_model: SpellModel
    ) -> float:
        """Calculate how well candidate fits in the surrounding context."""
        if not context:
            return 0.5
        return 0.5

    def _calculate_command_score(
        self, context: List[str], candidate: str
    ) -> float:
        """Score based on command vocabulary match."""
        command_words = {
            "open",
            "close",
            "launch",
            "start",
            "run",
            "lock",
            "shutdown",
            "restart",
            "reboot",
            "help",
            "commands",
            "history",
            "clear",
            "enable",
            "disable",
            "startup",
            "status",
        }
        if candidate.lower() in {
            "open",
            "close",
            "launch",
            "start",
            "run",
            "lock",
            "shutdown",
            "restart",
            "reboot",
            "help",
            "commands",
            "history",
            "clear",
            "enable",
            "disable",
            "startup",
            "status",
        }:
            return 1.0
        return 0.0

    def _calculate_application_score(self, candidate: str) -> float:
        """Score based on application vocabulary match."""
        candidate_lower = candidate.lower()
        from core.application_manager import APP_NAMES

        if candidate_lower in APP_NAMES:
            return 1.0
        from core.application_manager import APPLICATIONS

        for app in APPLICATIONS:
            for alias in app.aliases:
                words = alias.lower().split()
                if candidate_lower in words:
                    return 0.8
        return 0.0

    def _calculate_phrase_score(
        self, candidate: str, context: List[str], spell_model
    ) -> float:
        """Score based on multi-word phrase matching."""
        from core.application_manager import APPLICATIONS

        for app in APPLICATIONS:
            for alias in app.aliases:
                words = alias.lower().split()
                if len(words) > 1 and candidate.lower() in words:
                    other_words = [w for w in words if w != candidate.lower()]
                    if any(
                        w in [c.lower() for c in context] for w in other_words
                    ):
                        return 1.0
        return 0.0


# =============================================================================
# ADVANCED SPELL CHECKER MAIN CLASS
# =============================================================================

# Explicit typo corrections for dangerous commands.
# These are checked BEFORE the spell model to ensure dangerous commands
# are corrected conservatively and never to unrelated words.
DANGEROUS_COMMAND_TYPOS = {
    "shutdwon": "shutdown",
    "shutdonw": "shutdown",
    "restat": "restart",
    "restrt": "restart",
    "rebot": "reboot",
    "reboto": "reboot",
    "delet": "delete",
    "delte": "delete",
    "dlete": "delete",
}


class AdvancedSpellChecker:
    """Main entry point for advanced context-aware spell correction."""

    def __init__(self):
        self._model = None
        self.classifier = get_classifier()
        self.candidate_generator = None
        self.multi_word_detector = MultiWordAppDetector()

    def _load_model(self) -> SpellModel:
        """Load spell model lazily."""
        if self._model is None:
            self._model = SpellModel.load()
            from core.spell_checker import _augment_with_applications

            _augment_with_applications(self._model)
        return self._model

    def correct_tokens(self, tokens: List[str]) -> List[str]:
        """
        Correct a list of tokens with context-aware spell correction.

        This is the main entry point that replaces correct_tokens from
        spell_checker.py
        """
        if not tokens:
            return []

        enabled = get_value("spell_correction", "enabled", True)
        if enabled is False:
            return list(tokens)

        # Work from the registries directly. A missing/corrupt optional
        # frequency model must never trigger a less conservative fallback.
        from core.command_registry import get_all_commands
        from core.application_manager import APPLICATIONS

        command_words = {
            word.lower()
            for definition in get_all_commands().values()
            for alias in definition.aliases
            for word in alias.split()
        }
        app_words = {
            word.lower()
            for app in APPLICATIONS
            for alias in app.aliases
            for word in alias.split()
        }
        command_start_words = {
            alias.split()[0].lower()
            for definition in get_all_commands().values()
            for alias in definition.aliases
            if alias.split()
        }

        threshold = self._read_threshold("confidence_threshold", 0.75)
        dangerous_threshold = self._read_threshold(
            "dangerous_confidence_threshold", 0.90
        )
        corrected = list(tokens)
        discovery_words_loaded = False

        # Search text is user content; preserve it in full.
        if tokens and tokens[0].lower().strip(".,!?;:") in {
            "search", "find", "google", "look"
        }:
            return corrected

        protected_indices = {
            index
            for start, end, _ in self.multi_word_detector.detect_phrases(tokens)
            for index in range(start, end)
        }
        for index, token in enumerate(tokens):
            if index in protected_indices:
                continue
            classification = self.classifier.classify_token(token, tokens, index)
            if classification.is_protected or classification.token_type in {
                TokenType.URL, TokenType.WINDOWS_PATH, TokenType.FILE_EXTENSION,
                TokenType.FILE_NAME, TokenType.TECHNICAL_TERM,
            }:
                continue

            lower = token.lower()
            if lower in DANGEROUS_COMMAND_TYPOS:
                corrected[index] = DANGEROUS_COMMAND_TYPOS[lower]
                continue

            # Correct a command only at the beginning of a command segment.
            # Command vocabulary comes from Command Registry.
            if index == 0:
                candidates = self._near_words(lower, command_start_words)
                if candidates:
                    candidate, confidence, margin = candidates[0]
                    is_dangerous = candidate in {"shutdown", "restart", "delete"}
                    needed = dangerous_threshold if is_dangerous else threshold
                    if confidence >= needed and margin >= 0.08:
                        corrected[index] = candidate
                    continue

            # Application typos are eligible only as the target of an
            # application command. This avoids rewriting prose or queries.
            preceding = corrected[0].lower() if corrected else ""
            if preceding in {"open", "launch", "start", "run", "close", "quit", "exit"} or (
                len(tokens) == 1
            ):
                if not discovery_words_loaded and preceding in {
                    "open", "launch", "start", "run", "close", "quit", "exit"
                }:
                    discovery_words_loaded = True
                    try:
                        from core.application_discovery import get_discovery
                        for discovered in get_discovery().get_all()[:500]:
                            app_words.update(word.casefold() for word in discovered.name.split())
                    except Exception:
                        # Discovery is optional and must never break spell correction.
                        pass
                candidates = self._near_words(lower, app_words)
                if candidates:
                    candidate, confidence, margin = candidates[0]
                    if confidence >= threshold and margin >= 0.08:
                        corrected[index] = candidate
        return corrected

    @staticmethod
    def _read_threshold(key: str, default: float) -> float:
        value = get_value("spell_correction", key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return default
        if not 0.0 <= value <= 1.0:
            return default
        return float(value)

    @staticmethod
    def _edit_distance(left: str, right: str) -> int:
        previous = list(range(len(right) + 1))
        before_previous = None
        for i, char_left in enumerate(left, 1):
            current = [i]
            for j, char_right in enumerate(right, 1):
                distance = min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (char_left != char_right),
                )
                if (
                    before_previous is not None
                    and i > 1 and j > 1
                    and char_left == right[j - 2]
                    and left[i - 2] == char_right
                ):
                    distance = min(distance, before_previous[j - 2] + 1)
                current.append(distance)
            before_previous = previous
            previous = current
        return previous[-1]

    @staticmethod
    def _near_words(word: str, vocabulary: Set[str]) -> List[Tuple[str, float, float]]:
        """Return edit-distance candidates with a conservative confidence and margin."""
        if not word or not word.isalpha():
            return []
        distances = []
        for candidate in vocabulary:
            distance = AdvancedSpellChecker._edit_distance(word, candidate)
            if distance <= 2:
                similarity = 1.0 - distance / max(len(word), len(candidate))
                distances.append((candidate, distance, similarity))
        distances.sort(key=lambda item: (item[1], -item[2], item[0]))
        if not distances:
            return []
        best = distances[0]
        # Short tokens and distance-two guesses are inherently ambiguous.
        confidence = best[2] if best[1] == 1 and len(word) >= 3 else 0.0
        runner_up = next(
            (item[2] for item in distances[1:] if item[1] == best[1]), 0.0
        )
        return [(best[0], confidence, confidence - runner_up)]

        # Detect multi-word application phrases first
        phrase_matches = self.multi_word_detector.detect_phrases(tokens)
        phrase_indices = set()
        for start, end, phrase in phrase_matches:
            for i in range(start, end):
                phrase_indices.add(i)

        # Check if this is a search query (starts with "search web")
        is_search_query = (
            len(tokens) >= 2
            and tokens[0].lower() == "search"
            and tokens[1].lower() == "web"
        )

        # Classify all tokens
        token_classifications = []
        for i, token in enumerate(tokens):
            classification = self.classifier.classify_token(
                tokens[i], tokens, i
            )
            # Create new classification with is_multi_word_app flag set
            # (TokenClassification is frozen, so we can't modify it in place)
            if i in phrase_indices:
                classification = TokenClassification(
                    token=classification.token,
                    token_type=classification.token_type,
                    is_protected=classification.is_protected,
                    is_multi_word_app=True,
                    phrase_index=classification.phrase_index,
                )
            token_classifications.append(classification)

        # Generate corrections
        corrected = []
        for i, (token, classification) in enumerate(
            zip(tokens, token_classifications)
        ):
            # For protected tokens, keep original
            if classification.is_protected:
                corrected.append(token)
                continue

            # For URL, path, extension, technical term - keep original
            if classification.token_type in [
                TokenType.URL,
                TokenType.WINDOWS_PATH,
                TokenType.FILE_EXTENSION,
                TokenType.FILE_NAME,
                TokenType.TECHNICAL_TERM,
            ]:
                corrected.append(token)
                continue

            # For tokens part of a multi-word app phrase, keep original
            if classification.is_multi_word_app:
                corrected.append(token)
                continue

            # For search queries, preserve the search content
            if is_search_query and i >= 2:
                corrected.append(token)
                continue

            # Check for dangerous command typos first (explicit fixes)
            token_lower = token.lower()
            if token_lower in DANGEROUS_COMMAND_TYPOS:
                corrected.append(DANGEROUS_COMMAND_TYPOS[token_lower])
                continue

            # Use spell model for correction
            try:
                corrected_token = spell_model.correct(token)
                corrected.append(corrected_token)
            except Exception as exc:
                logger.warning(
                    "Spell correction failed for '%s': %s", token, exc
                )
                corrected.append(token)

        return corrected

    def _get_confidence_threshold(
        self, classification: TokenClassification
    ) -> float:
        """Get confidence threshold based on token classification."""
        base_threshold = 0.75
        dangerous_threshold = 0.90

        # Dangerous commands need higher threshold
        # This is a simplified check - in practice would need intent detection
        return 0.75


# =============================================================================
# PUBLIC API
# =============================================================================


def correct_tokens_advanced(tokens: List[str]) -> List[str]:
    """Main entry point for advanced spell correction."""
    checker = AdvancedSpellChecker()
    return checker.correct_tokens(tokens)


# Backward compatibility
def correct_tokens(tokens: List[str]) -> List[str]:
    """Backward compatible entry point."""
    return correct_tokens_advanced(tokens)


def correct_token(token: str) -> str:
    """Correct a single token (backward compatible)."""
    return correct_tokens_advanced([token])[0] if token else token
