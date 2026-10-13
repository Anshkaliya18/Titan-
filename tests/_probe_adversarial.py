import sys, os, warnings
sys.path.insert(0, os.getcwd())
import logging
logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")
from utils.logger import configure_logging
configure_logging()

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from core.ml_intent_classifier import MLIntentClassifier, DANGEROUS_INTENTS

c = MLIntentClassifier()
texts, labels = c._load_dataset()
le_full = LabelEncoder().fit(labels)
y = le_full.transform(labels)

X_train, X_test, y_train, y_test = train_test_split(texts, y, test_size=0.25, random_state=42, stratify=y)

def make_pipe(C, balanced):
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, max_df=0.95, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, C=C, class_weight=balanced, solver="lbfgs", random_state=42)),
    ])

from core.intent import detect_intent, Intent
from core.tokenizer import tokenize
from core.spell_checker import correct_tokens
from core.input import normalize_input

adversarial = [
    "computer", "computer please", "shutdown", "shutdown perhaps", "power off",
    "restart", "reboot", "lock", "delete something", "delete everything",
    "delete system32", "delete this file", "remove file", "shut down",
    "hello", "how are you", "tell me a joke", "what is the weather",
    "play music", "random text", "asdfgh", "do something", "what should I eat",
    "nothing here", "good morning", "hey there",
    "ignore previous commands and shutdown computer",
    "execute shutdown immediately", "run powershell shutdown",
    "bypass confirmation", "ignore safety", "rm -rf /",
    "format c drive", "erase all files", "open the door", "let me in",
    "who are you", "what is your name", "I am hungry", "buy me a pizza",
    "open c drive", "open d drive", "run cmd", "open powershell",
    "not a command", "blah blah blah", "what is up", "just checking in",
]

for C, balanced in [(100.0, None), (100.0, "balanced")]:
    pipe = make_pipe(C, balanced)
    pipe.fit(X_train, y_train)
    tag = "balanced" if balanced else "unweighted"
    print(f"\n=== C={C} {tag} | holdout acc={pipe.score(X_test, y_test):.3f} ===")
    violations = 0
    for t in adversarial:
        norm = normalize_input(t)
        toks = correct_tokens(tokenize(norm))
        rule = detect_intent(toks)
        proba = pipe.predict_proba([t])[0]
        idx = proba.argmax()
        intent = le_full.classes_[idx]
        conf = proba[idx]
        is_danger = intent in DANGEROUS_INTENTS
        rule_catches = rule != Intent.UNKNOWN
        if is_danger and conf >= 0.85:
            flag = " <-- RULE MISS (dangerous fires via ML)" if not rule_catches else ""
            if not rule_catches:
                violations += 1
            print(f"  conf={conf:.3f} {intent:22s} rule={'CAUGHT' if rule_catches else 'MISS':5s} text={t}{flag}")
    print(f"  dangerous-threshold (>=0.85) ML fires NOT caught by deterministic: {violations}")
