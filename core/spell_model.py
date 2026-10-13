"""
Spell Model

A simple edit-distance based spell-correction model.
Loads a word-frequency dictionary and uses it to correct misspelled words.
"""

import json
import os
from collections import OrderedDict
from typing import Dict, Optional, Set


class SpellModel:
    """Simple spell-correction model using edit distance and word frequency."""

    # Bounded FIFO cache for corrections of *unknown* words (Step 22).
    # The edit search — especially distance 2 — costs ~60 ms for an
    # unmatched word, versus ~0.1 ms for a dictionary probe, so repeated
    # tokens are memoized. The cache never grows past this size: the
    # oldest entry is evicted first.
    CACHE_MAX_SIZE = 1024

    def __init__(self, vocab: Dict[str, int]):
        """
        Initialize the model with a word-frequency dictionary.

        Args:
            vocab: Dictionary mapping words to their frequency counts.
        """
        self.vocab = vocab
        self.vocab_set: Set[str] = set(vocab.keys())
        # Correction cache state (see correct()). The cache lives on the
        # instance: a freshly loaded model starts with an empty cache,
        # so a reloaded vocabulary can never serve stale corrections.
        self._correction_cache: "OrderedDict[str, str]" = OrderedDict()
        self.cache_hits = 0      # unknown words answered from the cache
        self.cache_misses = 0    # unknown words that needed the edit search

    @classmethod
    def load(cls, model_path: Optional[str] = None) -> "SpellModel":
        """
        Load the model from a JSON file.

        Args:
            model_path: Path to the model JSON file. If None, uses default location.

        Returns:
            SpellModel: Loaded model instance.

        Raises:
            FileNotFoundError: If the model file does not exist.
            ValueError: If the model file is invalid.
        """
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "spell_model.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Spell model not found at {model_path}. "
                f"Run 'python train_spell_model.py' first."
            )

        with open(model_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        if not vocab:
            raise ValueError("Spell model is empty.")

        return cls(vocab)

    def correct(self, word: str) -> str:
        """
        Correct a single word using edit distance and frequency.

        Known vocabulary words are answered directly with an O(1) set
        lookup and are deliberately never cached: the vocabulary is far
        larger than the cache, so caching them would only evict the
        expensive results worth keeping. Corrections of unknown words
        are memoized in the bounded FIFO cache (CACHE_MAX_SIZE).

        Args:
            word: The word to correct.

        Returns:
            str: The corrected word, or the original if no correction found.
        """
        word = word.lower()

        # If word is already in vocabulary, return as-is
        if word in self.vocab_set:
            return word

        cached = self._correction_cache.get(word)
        if cached is not None:
            self.cache_hits += 1
            return cached

        self.cache_misses += 1
        result = self._correct_unknown(word)

        # Bounded FIFO insertion: evict the oldest entry when full.
        self._correction_cache[word] = result
        if len(self._correction_cache) > self.CACHE_MAX_SIZE:
            self._correction_cache.popitem(last=False)

        return result

    @property
    def cache_size(self) -> int:
        """Current number of cached unknown-word corrections."""
        return len(self._correction_cache)

    def clear_cache(self) -> None:
        """Empty the correction cache and reset the hit/miss counters."""
        self._correction_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    def _correct_unknown(self, word: str) -> str:
        """Run the edit-distance search for a word missing from the
        vocabulary: distance 1 first, then distance 2, else the word."""
        # Try edit distance 1
        candidates = self._known(self._edits1(word))
        if candidates:
            return self._best_candidate(word, candidates)

        # Try edit distance 2
        candidates = self._known(self._edits2(word))
        if candidates:
            return self._best_candidate(word, candidates)

        # No correction found, return original
        return word

    def _known(self, words: Set[str]) -> Set[str]:
        """Filter a set of words to only those in the vocabulary."""
        return words & self.vocab_set

    def _best_candidate(self, original: str, candidates: Set[str]) -> str:
        """
        Select the best correction from a set of candidates.

        Prefers words with:
        1. Higher frequency (most important)
        2. Longer length (insertions are common in typos)
        3. Longer common prefix with original
        """
        def common_prefix_len(a: str, b: str) -> int:
            """Length of common prefix between two strings."""
            i = 0
            while i < len(a) and i < len(b) and a[i] == b[i]:
                i += 1
            return i

        def score(w: str) -> tuple:
            # Priority 1: prefer higher frequency (dominant factor)
            freq = self.vocab[w]
            # Priority 2: prefer longer words (insertions more common)
            length_group = 0 if len(w) > len(original) else (1 if len(w) == len(original) else 2)
            # Priority 3: prefer longer common prefix
            prefix_len = common_prefix_len(w, original)
            # Frequency is multiplied by 1000 to dominate prefix length
            # Max prefix_len is ~20, max freq is large, so freq*1000 >> prefix_len
            return (freq * 1000, -length_group, prefix_len)

        return max(candidates, key=score)

    def _edits1(self, word: str) -> set:
        """Generate all strings at edit distance 1 from the word."""
        letters = "abcdefghijklmnopqrstuvwxyz"
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]

        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]

        return set(deletes + transposes + replaces + inserts)

    def _edits2(self, word: str) -> set:
        """Generate all strings at edit distance 2 from the word."""
        return set(e2 for e1 in self._edits1(word) for e2 in self._edits1(e1))
