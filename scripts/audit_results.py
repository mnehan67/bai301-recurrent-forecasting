"""Validate simple claims against the included full-run CSV (stdlib only)."""
import csv
from collections import Counter
from pathlib import Path

source = Path(__file__).resolve().parents[1] / "results" / "all_runs.csv"
with source.open(newline="", encoding="utf-8") as handle:
    records = list(csv.DictReader(handle))
counts = Counter(row["architecture"] for row in records)
statuses = Counter(row["status"] for row in records)
seeds = {row["seed"] for row in records}
print("Recorded runs:", len(records))
print("Architectures:", dict(counts))
print("Statuses:", dict(statuses))
print("Seeds:", sorted(seeds))
assert len(records) == 342
assert counts == {"RNN":108, "LSTM":108, "GRU":108, "ESN":18}
assert statuses == {"ok":342}
assert seeds == {"42", "123", "2026"}
print("AUDIT PASS: full factorial grid and ESN sweep logged")
