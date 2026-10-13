import sys, os
sys.path.insert(0, os.getcwd())

from utils.logger import configure_logging
configure_logging()
import logging
logging.getLogger().setLevel(logging.WARNING)

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from core.ml_intent_classifier import MLIntentClassifier

c = MLIntentClassifier()
texts, labels = c._load_dataset()
print("Dataset size after dedup:", len(texts))
print("Unique classes:", sorted(set(labels)))

le = LabelEncoder()
y = le.fit_transform(labels)
classes = list(le.classes_)

# Stratify if possible; if a class has only 1 sample, fall back to no stratify
try:
    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=0.25, random_state=42, stratify=y
    )
    stratified = True
except Exception as e:
    print("stratify failed:", e)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=0.25, random_state=42
    )
    stratified = False
print("stratified:", stratified)
print("train:", len(X_train), "test:", len(X_test))

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, max_df=0.95, sublinear_tf=True)),
    ("clf", LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", solver="lbfgs", multi_class="multinomial", random_state=42)),
])
pipe.fit(X_train, y_train)

acc = pipe.score(X_test, y_test)
print("Test accuracy:", round(acc, 3))

y_pred = pipe.predict(X_test)
print("\n--- Classification report (true intents, predicted) ---")
report = classification_report(y_test, y_pred, target_names=classes, zero_division=0)
print(report)

# Confidence distribution analysis
proba = pipe.predict_proba(X_test)
max_conf = proba.max(axis=1)
print("--- Confidence (max prob) distribution on TEST set ---")
print("min:", round(float(max_conf.min()), 3), "max:", round(float(max_conf.max()), 3),
      "mean:", round(float(max_conf.mean()), 3), "median:", round(float(np.median(max_conf)), 3))
for threshold in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85]:
    covered = (max_conf >= threshold).mean()
    print(f"  threshold {threshold}: {covered:.1%} of test samples covered, accuracy among covered: ", end="")
    mask = max_conf >= threshold
    if mask.sum() > 0:
        covered_acc = (y_pred[mask] == y_test[mask]).mean()
        print(round(float(covered_acc), 3))
    else:
        print("N/A")

# Per-class test counts
from collections import Counter
print("\n--- per-class test sample counts ---")
for i, cls in enumerate(classes):
    n = (y_test == i).sum()
    print(f"  {n:3d}  {cls}")

# What do the lowest-confidence predictions look like?
print("\n--- lowest-confidence test predictions (most risky) ---")
order = np.argsort(max_conf)
shown = 0
for idx in order:
    if shown >= 25:
        break
    true = classes[y_test[idx]]
    pred = classes[y_pred[idx]]
    if true == pred and max_conf[idx] < 0.6:
        shown += 1
        print(f"  conf={max_conf[idx]:.3f}  true={true:22s} pred={pred:22s}  text={X_test[idx]}")
