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
from core.intent import detect_intent, Intent
from core.tokenizer import tokenize
from core.spell_checker import correct_tokens
from core.input import normalize_input

c = MLIntentClassifier()
texts, labels = c._load_dataset()
le = LabelEncoder().fit(labels)
y = le.transform(labels)
X_train, X_test, y_train, y_test = train_test_split(texts, y, test_size=0.25, random_state=42, stratify=y)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, max_df=0.95, sublinear_tf=True)),
    ("clf", LogisticRegression(max_iter=2000, C=100.0, class_weight="balanced", solver="lbfgs", random_state=42)),
])
pipe.fit(X_train, y_train)

# Natural language for every intent, including dangerous ones phrased politely
nl = {
    "OPEN_APPLICATION": ["open chrome", "please open chrome", "can you launch chrome",
        "could you please open google chrome for me", "would you mind opening chrome",
        "i would like to open chrome", "start chrome please", "open calculator",
        "open notepad", "open file explorer"],
    "CLOSE_APPLICATION": ["close chrome", "please close chrome", "quit chrome",
        "exit firefox", "please close google chrome"],
    "LOCK_COMPUTER": ["lock computer", "please lock computer", "lock my computer",
        "could you lock the computer"],
    "SHUTDOWN_COMPUTER": ["shutdown computer", "please shut down computer",
        "shut down my computer", "power off", "please shutdown computer",
        "turn off the computer"],
    "RESTART_COMPUTER": ["restart computer", "please restart the computer",
        "reboot my computer", "would you reboot the computer"],
    "ENABLE_STARTUP": ["enable startup", "please enable startup",
        "start with windows", "make jarvis start with windows"],
    "DISABLE_STARTUP": ["disable startup", "please disable startup",
        "stop windows startup", "disable autostart"],
    "STARTUP_STATUS": ["startup status", "check startup", "is startup enabled",
        "am i starting with windows"],
    "CREATE_FOLDER": ["create folder", "make folder", "please create folder",
        "create a folder", "new folder"],
    "DELETE_FILE": ["delete file", "please delete file", "delete file test.txt",
        "remove file report.pdf", "delete the file"],
    "OPEN_FILE": ["open file", "open file test.txt", "open report.pdf",
        "open the file"],
    "SEARCH_FILE": ["search file", "find file", "search file test.txt",
        "find the file"],
    "OPEN_BROWSER": ["open browser", "launch browser", "open my browser",
        "launch my browser"],
    "SEARCH_WEB": ["search web python tutorials", "search the web for python",
        "google python tutorials", "search cats and dogs"],
    "OPEN_URL": ["open url https://github.com", "open https://example.com",
        "go to https://github.com", "please open https://google.com"],
    "HELP": ["help", "what can you do", "how can you help me", "show help",
        "please help"],
    "LIST_COMMANDS": ["commands", "list commands", "show commands",
        "available commands", "what commands"],
    "HISTORY": ["history", "show history", "command history",
        "recent commands", "show me history"],
    "CLEAR_HISTORY": ["clear history", "delete history", "remove history",
        "wipe history"],
}

THRESH = 0.50
DANGER = 0.85
print(f"{'INTENT':22s} {'INPUT':40s} {'PRED':22s} {'CONF':6s} {'RULE':22s} {'STATUS'}")
print("-" * 135)
problems = []
for true_intent, phrases in nl.items():
    for phrase in phrases:
        norm = normalize_input(phrase)
        toks = correct_tokens(tokenize(norm))
        rule = detect_intent(toks)
        proba = pipe.predict_proba([phrase])[0]
        idx = proba.argmax()
        pred = le.classes_[idx]
        conf = proba[idx]
        # In the hybrid pipeline: if rule != UNKNOWN, rule wins. So ML only matters when rule==UNKNOWN.
        rule_val = rule.value
        status = ""
        if rule_val != "unknown":
            # deterministic caught it; ML not used. Mark OK if rule matches.
            status = "OK(rule)" if rule_val == true_intent.lower() else "RULE-MISMATCH"
        else:
            # ML used. Check pred matches and threshold.
            if pred == true_intent and conf >= THRESH:
                if true_intent in DANGEROUS_INTENTS:
                    status = "OK(ml,danger<%.0f)" % (DANGER*100) if conf >= DANGER else f"OK(ml,danger-thr)"
                    if conf >= DANGER:
                        status = "OK(ml,>=danger-thr->confirm)"
                else:
                    status = "OK(ml)"
            elif pred == true_intent and conf < THRESH:
                status = f"!! below-thresh(conf={conf:.2f})"
                problems.append((phrase, true_intent, pred, conf, "below-thresh"))
            elif pred != true_intent and conf >= THRESH:
                status = f"!! wrong-pred({pred})"
                problems.append((phrase, true_intent, pred, conf, "wrong-pred"))
            else:
                status = "unknown(ok)"
        print(f"{true_intent:22s} {phrase:40s} {pred:22s} {conf:6.3f} {rule_val:22s} {status}")

print("\n=== Problems (ML used, legitimate input not correctly classified above threshold) ===")
for p in problems:
    print(f"  {p}")
print("total problems:", len(problems))
