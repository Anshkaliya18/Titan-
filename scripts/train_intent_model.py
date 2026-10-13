#!/usr/bin/env python
"""Train and evaluate the ML intent classifier (Step 25).

Usage:
    python scripts/train_intent_model.py              train + evaluate + save
    python scripts/train_intent_model.py --no-save    train + evaluate only
    python scripts/train_intent_model.py --force      retrain even if a saved
                                                      model already exists

Steps:
    1. Load dataset (data/intent_dataset.csv).
    2. Validate structure (columns, empty rows, duplicates reported).
    3. Validate every label against the Command Registry.
    4. Stratified train/validation split (80/20, fixed seed).
    5. Train TF-IDF + Logistic Regression.
    6. Evaluate: accuracy + per-class precision/recall/F1.
    7. Save model to the writable data location.
    8. Report the model path and metrics.

Fails safely (non-zero exit, clear message) on invalid data. Never
downloads anything; never touches the network.

NOTE on honest evaluation: the dataset is small (~250 examples across
20 classes), so the validation split is also small (~50 examples).
Metrics are reported as measured — with a dataset this size they are
indicative, not statistically robust. This script says so when the
validation set is small.
"""

from __future__ import annotations

import argparse
import collections
import csv
import sys
import time
from pathlib import Path

# Project root on sys.path so `core` / `utils` import anywhere.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.command_registry import is_valid_intent  # noqa: E402
from utils import paths  # noqa: E402


def load_dataset(path: Path) -> tuple[list[str], list[str]]:
    """Load and validate the dataset. Raises ValueError on any problem."""
    if not path.exists():
        raise ValueError(f"Dataset not found: {path}")

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["text", "intent"]:
            raise ValueError(
                f"Expected columns ['text', 'intent'], "
                f"got {reader.fieldnames}"
            )
        rows = [(r.get("text", ""), r.get("intent", "")) for r in reader]

    texts: list[str] = []
    labels: list[str] = []
    seen: set[tuple[str, str]] = set()
    duplicates = 0
    for i, (text, intent) in enumerate(rows, start=2):
        text, intent = text.strip(), intent.strip()
        if not text:
            raise ValueError(f"Line {i}: empty text")
        if not intent:
            raise ValueError(f"Line {i}: empty intent")
        if not is_valid_intent(intent):
            raise ValueError(
                f"Line {i}: invalid intent '{intent}' (not in registry)"
            )
        key = (text.lower(), intent)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        texts.append(text)
        labels.append(intent)

    if not texts:
        raise ValueError("No valid training examples after validation")

    print(f"Loaded {len(texts)} examples "
          f"({duplicates} exact duplicates skipped)")
    counts = collections.Counter(labels)
    print(f"Classes: {len(counts)}")
    for intent in sorted(counts):
        print(f"  {intent:22} {counts[intent]:3}")
    thinnest = min(counts.values())
    if thinnest < 5:
        print(f"WARNING: thinnest class has only {thinnest} examples; "
              f"predictions for it will be fragile.")
    return texts, labels


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the Jarvis ML intent classifier."
    )
    parser.add_argument("--no-save", action="store_true",
                        help="train and evaluate without saving the model")
    parser.add_argument("--force", action="store_true",
                        help="retrain even if a saved model already exists")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="fraction held out for validation (default 0.2)")
    args = parser.parse_args(argv)

    # Lazy sklearn import: fast --help, clear error when ML stack missing.
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report
        from sklearn.model_selection import StratifiedShuffleSplit
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import LabelEncoder
    except ImportError as exc:
        print(f"ERROR: scikit-learn is required to train: {exc}")
        print("Install it with: python -m pip install scikit-learn joblib")
        return 2

    from core.ml_intent_classifier import (
        MLIntentClassifier,
        MODEL_DIR_NAME,
        MODEL_FILENAME,
    )

    dataset_path = paths.bundle_dir() / "data" / "intent_dataset.csv"
    model_path = paths.data_root() / MODEL_DIR_NAME / MODEL_FILENAME

    if model_path.exists() and not args.force and not args.no_save:
        print(f"A saved model already exists at {model_path}.")
        print("Use --force to retrain or --no-save to evaluate only.")

    try:
        texts, labels = load_dataset(dataset_path)
    except ValueError as exc:
        print(f"ERROR: invalid dataset: {exc}")
        return 1

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=args.test_size, random_state=42
    )
    train_idx, test_idx = next(splitter.split(texts, y))
    X_train = [texts[i] for i in train_idx]
    y_train = [labels[i] for i in train_idx]
    X_test = [texts[i] for i in test_idx]
    y_test = [labels[i] for i in test_idx]
    print(f"\nSplit: {len(X_train)} train / {len(X_test)} validation "
          f"(stratified, seed 42)")
    if len(X_test) < 50:
        print("NOTE: validation set is small (< 50 examples), so the "
              "metrics below are indicative, not statistically robust.")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2),
            min_df=1, max_df=0.95, sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1000, C=100.0, class_weight="balanced",
            solver="lbfgs", random_state=42,
        )),
    ])

    t0 = time.perf_counter()
    pipeline.fit(X_train, encoder.transform(y_train))
    train_ms = (time.perf_counter() - t0) * 1000

    y_pred = pipeline.predict(X_test)
    pred_labels = list(encoder.inverse_transform(y_pred))
    accuracy = sum(p == t for p, t in zip(pred_labels, y_test)) / len(y_test)

    print(f"\nTraining took {train_ms:.0f} ms")
    print(f"Validation accuracy: {accuracy:.3f} "
          f"({sum(p == t for p, t in zip(pred_labels, y_test))}/{len(y_test)})")
    print("\nPer-class precision / recall / F1:")
    print(classification_report(y_test, pred_labels, digits=3))

    # Confidence calibration summary on the validation set.
    import numpy as np
    proba = pipeline.predict_proba(X_test)
    conf = proba.max(axis=1)
    correct = np.array([p == t for p, t in zip(pred_labels, y_test)])
    print(f"Mean confidence (correct):   {conf[correct].mean():.3f} "
          f"(n={int(correct.sum())})")
    if (~correct).sum():
        print(f"Mean confidence (wrong):     {conf[~correct].mean():.3f} "
              f"(n={int((~correct).sum())})")
    else:
        print("No misclassifications on the validation set.")

    if args.no_save:
        print("\n--no-save: model not written.")
        return 0

    # Train the production model on ALL data and save it.
    full = MLIntentClassifier()
    full.train(texts, labels)
    full.save(model_path)
    print(f"\nFull model trained on {len(texts)} examples "
          f"and saved to:\n  {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
