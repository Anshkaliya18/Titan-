"""
Tokenizer Module

Splits the user's query into individual tokens (words).
Uses NLTK's word_tokenize for now.

Startup note (Step 22): the NLTK import is wrapped by
utils.nltk_loader, which installs a small shim for NLTK's optional
scientific dependencies (scipy/sklearn) that TITAN never uses, so the
import takes NLTK's own guarded "package not installed" branch. This
cuts ~1.1 s from startup with byte-identical tokenization output (see
utils/nltk_loader.py and the A/B test in tests/test_performance.py).
The import still happens eagerly at module load, so the first command
never pays an unexpected one-time cost.
"""

from utils.logger import get_logger
from utils.nltk_loader import import_word_tokenize

logger = get_logger(__name__)

# Imported once at module load (see the startup note above).
word_tokenize = import_word_tokenize()


def tokenize(query: str) -> list[str]:
    """
    Tokenize the query string into individual words.

    Args:
        query (str): The normalized input string.

    Returns:
        list[str]: A list of tokens (words) from the query.
    """
    tokens = word_tokenize(query)
    # Only the token count is logged: token contents may include
    # user-typed text, and a token list cannot be reliably redacted.
    logger.debug(f"Tokenization completed ({len(tokens)} tokens)")
    return tokens
