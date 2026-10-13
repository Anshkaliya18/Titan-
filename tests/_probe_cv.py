import sys, os
sys.path.insert(0, os.getcwd())

from utils.logger import configure_logging
configure_logging()
import logging
logging.getLogger().setLevel(logging.WARNING)

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from core.ml_intent_classifier import MLIntentClassifier

c = MLIntentClassifier()
texts, labels = c._load_dataset()

le = LabelEncoder()
y = le.fit_transform(labels)
classes = list(le.classes_)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def make_pipe(C, balanced):
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, max_df=0.95, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, C=C, class_weight=balanced, solver="lbfgs", multi_class="multinomial", random_state=42)),
    ])

for C in [1.0, 10.0, 100.0]:
    for balanced in [None, "balanced"]:
        scores = cross_val_score(make_pipe(C, balanced), texts, y, cv=cv, scoring="accuracy")
        tag = "balanced" if balanced else "unweighted"
        print(f"C={C:>6} {tag:9}: 5-fold CV acc={scores.mean():.3f} +/- {scores.std():.3f}")

# Detailed per-class for the chosen config on a fixed split
X_train, X_test, y_train, y_test = train_test_split(texts, y, test_size=0.25, random_state=42, stratify=y)
pipe = make_pipe(100.0, None)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print("\n--- Per-class report (unweighted C=100) ---")
print(classification_report(y_test, y_pred, target_names=classes, zero_division=0))

cm = confusion_matrix(y_test, y_pred)
print("--- Off-diagonal confusion ---")
for i, tc in enumerate(classes):
    for j, pc in enumerate(classes):
        if cm[i][j] > 0 and i != j:
            print(f"  true={tc:22s} pred={pc:22s} n={cm[i][j]}")

# Does OPEN_APPLICATION dominate? Count predicted vs true for top class
from collections import Counter
print("\n--- predicted class distribution (test) ---")
for cls, cnt in Counter(y_pred).most_common():
    print(f"  {cnt:3d}  {classes[cls]}")
print("--- true class distribution (test) ---")
for cls, cnt in Counter(y_test).most_common():
    print(f"  {cnt:3d}  {classes[cls]}")
