import sys, os
sys.path.insert(0, os.getcwd())

from utils.logger import configure_logging
configure_logging()
import logging
logging.getLogger().setLevel(logging.WARNING)

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

from core.ml_intent_classifier import MLIntentClassifier

c = MLIntentClassifier()
texts, labels = c._load_dataset()
print("Dataset size:", len(texts))

le = LabelEncoder()
y = le.fit_transform(labels)
classes = list(le.classes_)

X_train, X_test, y_train, y_test = train_test_split(texts, y, test_size=0.25, random_state=42, stratify=y)

def evaluate(name, clf_kwargs, vec_kwargs=None):
    vec_kwargs = vec_kwargs or {}
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=vec_kwargs.get("ngram_range", (1, 2)),
                                  min_df=vec_kwargs.get("min_df", 1),
                                  max_df=vec_kwargs.get("max_df", 0.95),
                                  sublinear_tf=vec_kwargs.get("sublinear_tf", True))),
        ("clf", LogisticRegression(**clf_kwargs)),
    ])
    pipe.fit(X_train, y_train)
    acc = pipe.score(X_test, y_test)
    proba = pipe.predict_proba(X_test)
    max_conf = proba.max(axis=1)
    print(f"{name}: acc={acc:.3f}  conf[min={max_conf.min():.3f} max={max_conf.max():.3f} mean={max_conf.mean():.3f} median={np.median(max_conf):.3f}]")
    for th in [0.3, 0.4, 0.5]:
        mask = max_conf >= th
        if mask.sum() > 0:
            covered_acc = (pipe.predict(X_test)[mask] == y_test[mask]).mean()
            print(f"    th={th}: {mask.mean():.1%} covered, acc={covered_acc:.3f}")
        else:
            print(f"    th={th}: 0% covered")
    return pipe

print("\n=== Experiment: compare configurations ===")
evaluate("balanced C1.0 (current)", {"max_iter":1000,"C":1.0,"class_weight":"balanced","solver":"lbfgs","multi_class":"multinomial","random_state":42})
evaluate("balanced C10", {"max_iter":1000,"C":10.0,"class_weight":"balanced","solver":"lbfgs","multi_class":"multinomial","random_state":42})
evaluate("balanced C100", {"max_iter":1000,"C":100.0,"class_weight":"balanced","solver":"lbfgs","multi_class":"multinomial","random_state":42})
evaluate("unweighted C1.0", {"max_iter":1000,"C":1.0,"class_weight":None,"solver":"lbfgs","multi_class":"multinomial","random_state":42})
evaluate("unweighted C10", {"max_iter":1000,"C":10.0,"class_weight":None,"solver":"lbfgs","multi_class":"multinomial","random_state":42})
evaluate("unweighted C100", {"max_iter":1000,"C":100.0,"class_weight":None,"solver":"lbfgs","multi_class":"multinomial","random_state":42})
evaluate("balanced C100 unigrams only", {"max_iter":1000,"C":100.0,"class_weight":"balanced","solver":"lbfgs","multi_class":"multinomial","random_state":42}, {"ngram_range":(1,1)})

# Test on UNSEEN natural-language inputs with the best config
print("\n=== Natural-language inputs (unseen) with unweighted C100 ===")
best = evaluate.__wrapped__ if hasattr(evaluate,'__wrapped__') else None
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, max_df=0.95, sublinear_tf=True)),
    ("clf", LogisticRegression(max_iter=1000, C=100.0, class_weight=None, solver="lbfgs", multi_class="multinomial", random_state=42)),
])
pipe.fit(X_train, y_train)
le2 = LabelEncoder(); le2.fit(labels)
nl = [
    "could you please open chrome",
    "would you mind launching chrome",
    "can you start my browser",
    "please open google chrome for me",
    "i want to open chrome",
    "I would like to open calculator",
    "please launch notepad",
    "open calculator please",
    "hello",
    "how are you",
    "tell me a joke",
    "what is the weather",
    "computer",
    "shutdown perhaps",
    "delete something",
    "nothing here",
    "play music",
]
for t in nl:
    proba = pipe.predict_proba([t])[0]
    idx = proba.argmax()
    intent = le2.inverse_transform([idx])[0]
    conf = proba[idx]
    print(f"  conf={conf:.3f}  {intent:22s}  {t}")
