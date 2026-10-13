import sys, os
sys.path.insert(0, os.getcwd())

from utils.logger import configure_logging
configure_logging()
import logging
logging.getLogger().setLevel(logging.WARNING)

from core.input import normalize_input
from core.tokenizer import tokenize
from core.spell_checker import correct_tokens
from core.intent import detect_intent as detect_intent_rules
from core.hybrid_intent import detect_hybrid_intent
from core.ml_intent_classifier import MLIntentClassifier, initialize_classifier

# Initialize the real classifier
initialize_classifier()

from core.hybrid_intent import _get_ml_config
print("ML config:", _get_ml_config())

cases = [
    # Deterministic should catch these — ML should NOT override
    "open chrome",
    "open google chrome",
    "open calc",
    "open calculator",
    "open notepad",
    "open file explorer",
    "open browser",
    "open https://github.com",
    "open url",
    "lock computer",
    "shutdown computer",
    "restart computer",
    "create folder",
    "delete file test.txt",
    "help",
    "commands",
    "show history",
    "clear history",
    "enable startup",
    "disable startup",
    "startup status",
    "open cmd",
    "open explorer",
    "search web cats and dogs",
    "search web python tutorials",
    # Natural language — deterministic likely UNKNOWN, ML fallback
    "Could you please open Chrome for me?",
    "Would you mind launching Chrome?",
    "Can you start my browser?",
    "Please search the web for Python tutorials.",
    "I want to see my recent commands.",
    "Can you show me what commands are available?",
    "Please lock my computer.",
    "Could you restart the computer?",
    "please open chrome please",
    "I would like to open chrome",
    # Adversarial
    "computer",
    "chrome maybe",
    "shutdown perhaps",
    "delete something",
    "nothing here",
    "hello how are you",
    "what is the weather today",
]

print(f"\n{'INPUT':45s} {'RULE':22s} {'HYBRID':22s} {'SRC':6s} {'CONF':6s}  {'ML_INTENT':22s} {'ML_CONF':6s}")
print("-" * 140)
for c in cases:
    norm = normalize_input(c)
    toks = tokenize(norm)
    corrected = correct_tokens(toks)
    rule_intent = detect_intent_rules(corrected)
    # tokens passed to hybrid: corrected; text: corrected text joined
    # (matches main.py / multi_command_executor.py after the Step 25
    # debugging pass — ML must see spell-corrected text).
    hr = detect_hybrid_intent(tokens=corrected, text=" ".join(corrected))
    print(f"{c:45s} {rule_intent.value:22s} {hr.intent.value:22s} {hr.source:6s} {hr.confidence:6.3f}  {str(hr.ml_intent):22s} {hr.ml_confidence:6.3f}")
