import sys, os, csv
from collections import Counter
sys.path.insert(0, os.getcwd())

from core.ml_intent_classifier import DATASET_PATH
print("DATASET_PATH:", DATASET_PATH, "exists:", DATASET_PATH.exists())

texts = []
labels = []
with open(DATASET_PATH, encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    print("header:", header)
    for row in reader:
        if not row or (len(row) == 1 and not row[0].strip()):
            continue
        texts.append(row[0])
        labels.append(row[1])

print("total rows:", len(texts))

# Distribution
counter = Counter(labels)
print("--- label distribution ---")
for intent, count in sorted(counter.items()):
    print(f"{count:4d}  {intent}")

# Exact duplicate (text, label) pairs
pairs = list(zip(texts, labels))
dup = Counter(pairs)
dups = {k: v for k, v in dup.items() if v > 1}
print("--- duplicate (text,label) pairs:", len(dups), "---")
for k, v in sorted(dups.items(), key=lambda x: -x[1])[:20]:
    print(f"{v}x  {k}")

# Duplicate texts with different labels (conflicts)
from collections import defaultdict
text_to_labels = defaultdict(set)
for t, l in pairs:
    text_to_labels[t].add(l)
conflicts = {t: ls for t, ls in text_to_labels.items() if len(ls) > 1}
print("--- texts with conflicting labels:", len(conflicts), "---")
for t, ls in list(conflicts.items())[:20]:
    print(f"{ls}  {t}")

# Blank/empty rows
print("--- empty text rows:", sum(1 for t in texts if not t.strip()), "---")
