"""Check that this package contains the user's actual executed notebook and runs."""

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
notebook = ROOT / "notebooks" / "BAI301_Kaggle_T4x2_EXECUTED.ipynb"
original = ROOT / "local_only" / "Original_Kaggle_Results.zip"
report = ROOT / "results" / "all_runs.csv"
raw_logs = ROOT / "results" / "reproduction_archive" / "exp_b8f83540bd67fab5" / "runs"

data = json.loads(notebook.read_text(encoding="utf-8"))
executed = sum(c.get("execution_count") is not None for c in data["cells"])
with report.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

assert len(data["cells"]) == 115
assert executed == 54, executed
assert len(rows) == 342, len(rows)
assert len(list(raw_logs.glob("*.json"))) == 342
assert all(r["status"] == "ok" for r in rows)

expected_nb = "c1734556b76227bb16e81eb0866d8ff204cc3fcbc0434ed447b8265d33bc7ef8"
assert hashlib.sha256(notebook.read_bytes()).hexdigest() == expected_nb
if original.exists():
    expected_zip = "96930d48cf3f07947ba35b137e50fd94a0bf8b3fff81f426844a0b4cb4e3dd91"
    assert hashlib.sha256(original.read_bytes()).hexdigest() == expected_zip
    print("Original 29 MB Kaggle export: verified and excluded from Git")
else:
    print("Original raw ZIP not present (expected in a GitHub clone)")

print(f"PASS: {len(data['cells'])} notebook cells, {executed} executed cells, {len(rows)} completed runs")
