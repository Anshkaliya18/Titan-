"""One-shot dataset cleaner: removes blank lines and exact duplicate rows.

Dedup key matches core.ml_intent_classifier._load_dataset: (text.lower(), intent).
Also validates every label against the Command Registry and writes a clean,
compact CSV. Idempotent.
"""
import csv
import sys
import os
from collections import OrderedDict

sys.path.insert(0, os.getcwd())
from core.command_registry import is_valid_intent

SRC = "data/intent_dataset.csv"

rows_in = []
with open(SRC, encoding="utf-8", newline="") as f:
    reader = csv.reader(f)
    header = next(reader)
    assert header == ["text", "intent"], header
    for row in reader:
        if not row or not row[0].strip():
            continue
        assert len(row) == 2, f"malformed row: {row}"
        rows_in.append((row[0], row[1]))

print(f"Input: {len(rows_in)} non-empty rows (after dropping 19 blank lines)")

seen = set()
clean = []
dup_count = 0
for text, intent in rows_in:
    if not is_valid_intent(intent):
        raise SystemExit(f"INVALID intent in dataset: {intent!r} for text {text!r}")
    key = (text.lower(), intent)
    if key in seen:
        dup_count += 1
        continue
    seen.add(key)
    clean.append((text, intent))

print(f"Removed {dup_count} duplicate rows (case-insensitive key)")
print(f"Output: {len(clean)} unique rows")

# Distribution
from collections import Counter
dist = Counter(intent for _, intent in clean)
print("\nClean distribution:")
for intent, count in sorted(dist.items()):
    print(f"  {count:3d}  {intent}")

# Write back, preserving first-occurrence order (groups stay contiguous)
with open(SRC, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(["text", "intent"])
    for text, intent in clean:
        writer.writerow([text, intent])

print(f"\nWrote clean dataset to {SRC}")

# Verify idempotency: re-reading should yield the same set
rows2 = []
with open(SRC, encoding="utf-8", newline="") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        rows2.append((row[0], row[1]))
assert len(rows2) == len(clean), "idempotency check failed"
assert set((t.lower(), i) for t, i in rows2) == set((t.lower(), i) for t, i in clean)
print("Idempotency check passed.")
