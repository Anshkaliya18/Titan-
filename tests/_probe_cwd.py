import sys, os, warnings, tempfile, shutil
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import logging
logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")

from utils.logger import configure_logging
configure_logging()

# Capture the resolved dataset path
import core.ml_intent_classifier as mlc_mod
print("DATASET_PATH (module attr):", mlc_mod.DATASET_PATH)
print("  is_absolute:", mlc_mod.DATASET_PATH.is_absolute())
print("  exists from project dir:", mlc_mod.DATASET_PATH.exists())

# Now test from a DIFFERENT working directory
other = tempfile.mkdtemp(prefix="TITAN-cwd-")
os.chdir(other)
print("\nChanged CWD to:", other)
print("DATASET_PATH from other dir exists:", mlc_mod.DATASET_PATH.exists())

# Try training from other dir with a fresh classifier (no saved model fallback)
from core.ml_intent_classifier import MLIntentClassifier
c = MLIntentClassifier()
try:
    c.train()
    print("train() from other dir: SUCCESS (unexpected)")
except FileNotFoundError as e:
    print(f"train() from other dir: FileNotFoundError -> {e}")
except Exception as e:
    print(f"train() from other dir: {type(e).__name__} -> {e}")

# But load() uses paths.data_root() (absolute) — does it still work?
print("\nload() from other dir:")
ok = c.load()
print("  load returned:", ok, "| is_trained:", c.is_trained)

shutil.rmtree(other, ignore_errors=True)

# Convergence test at C=100
print("\n=== Convergence check C=100 ===")
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
mc = MLIntentClassifier()
texts, labels = mc._load_dataset()
y = LabelEncoder().fit_transform(labels)
Xtr, Xte, ytr, yte = train_test_split(texts, y, test_size=0.25, random_state=42, stratify=y)
for mi in [1000, 2000, 5000]:
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, max_df=0.95, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=mi, C=100.0, class_weight="balanced", solver="lbfgs", random_state=42)),
    ])
    import io, contextlib
    buf = io.StringIO()
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        pipe.fit(Xtr, ytr)
        conv = any("Converge" in str(w.message) for w in wlist)
    print(f"  max_iter={mi}: converged={not conv}  acc={pipe.score(Xte, yte):.3f}")
