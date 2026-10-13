import sys, os, warnings, io, contextlib
sys.path.insert(0, os.getcwd())
import logging
logging.disable(logging.WARNING)
from utils.logger import configure_logging
configure_logging()

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import warnings as _w

from core.ml_intent_classifier import MLIntentClassifier

c = MLIntentClassifier()
texts, labels = c._load_dataset()
y = LabelEncoder().fit_transform(labels)
print("Full dataset size:", len(texts))

for mi in [1000, 2000]:
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf), _w.catch_warnings(record=True) as wlist:
        _w.simplefilter("always")
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, max_df=0.95, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=mi, C=100.0, class_weight="balanced", solver="lbfgs", random_state=42)),
        ])
        pipe.fit(texts, y)
        conv_warnings = [str(w.message) for w in wlist if "Converge" in str(w.message)]
    print(f"max_iter={mi}: convergence warnings={conv_warnings}, n_iter={pipe.named_steps['clf'].n_iter_}")
