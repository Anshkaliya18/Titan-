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
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from core.ml_intent_classifier import MLIntentClassifier

c = MLIntentClassifier()
texts, labels = c._load_dataset()
y = LabelEncoder().fit_transform(labels)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def make_pipe(C, balanced):
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, max_df=0.95, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, C=C, class_weight=balanced, solver="lbfgs", random_state=42)),
    ])

print("=== CV accuracy comparison ===")
for C in [1.0, 10.0, 100.0]:
    for balanced in [None, "balanced"]:
        scores = cross_val_score(make_pipe(C, balanced), texts, y, cv=cv, scoring="accuracy")
        tag = "balanced" if balanced else "unweighted"
        print(f"  C={C:>6} {tag:9}: CV acc={scores.mean():.3f} +/- {scores.std():.3f}  folds={np.round(scores,3)}")

# Confidence spread comparison
print("\n=== Confidence spread (mean over full dataset via 5-fold) ===")
from sklearn.model_selection import cross_val_predict
for C in [1.0, 100.0]:
    for balanced in [None, "balanced"]:
        pipe = make_pipe(C, balanced)
        proba = cross_val_predict(pipe, texts, y, cv=cv, method="predict_proba")
        maxp = proba.max(axis=1)
        tag = "balanced" if balanced else "unweighted"
        print(f"  C={C:>6} {tag:9}: mean={maxp.mean():.3f} median={np.median(maxp):.3f} min={maxp.min():.3f} max={maxp.max():.3f}")
