"""
Train a simple spell-correction model.

Builds a word-frequency dictionary from NLTK's words corpus
and saves it as JSON for the spell checker to use.
"""

import json
import os
from collections import Counter

import nltk


def train():
    """Train and save the spell-correction model."""
    # Download required NLTK data
    nltk.download("words", quiet=True)

    from nltk.corpus import words

    # Build word frequency dictionary
    word_list = [w.lower() for w in words.words() if w.isalpha()]
    freq = Counter(word_list)

    # Save model
    model_path = os.path.join(os.path.dirname(__file__), "core", "spell_model.json")
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(dict(freq), f)

    print(f"Model saved to {model_path}")
    print(f"Vocabulary size: {len(freq)} words")


if __name__ == "__main__":
    train()
