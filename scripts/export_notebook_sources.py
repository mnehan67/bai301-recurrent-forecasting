"""Mechanically extract the exact engine files embedded in the Kaggle notebook.

Run from the repository root:
    python scripts/export_notebook_sources.py

The notebook creates these files with %%writefile magics. This utility
concatenates those same cells in original order, without rewriting the logic.
"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "BAI301_Kaggle_T4x2_FULL.ipynb"
DESTINATION = ROOT / "src"

data = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
pieces = {"experiment_engine.py": [], "analysis_report.py": []}

for cell in data["cells"]:
    if cell.get("cell_type") != "code":
        continue
    contents = "".join(cell.get("source", []))
    if not contents.startswith("%%writefile"):
        continue
    directive, _, body = contents.partition("\n")
    name = directive.split()[-1]
    if name in pieces:
        pieces[name].append(body)

DESTINATION.mkdir(parents=True, exist_ok=True)
for name, chunks in pieces.items():
    if not chunks:
        raise ValueError(f"Notebook did not contain the source for {name}")
    target = DESTINATION / name
    target.write_text("\n".join(chunks), encoding="utf-8")
    print(f"Extracted {len(chunks)} cells to {target.relative_to(ROOT)}")
