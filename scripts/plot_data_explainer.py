"""Plot actual 100-input/100-target windows after running --mode prepare.

Usage: python scripts/plot_data_explainer.py --config configs/full.local.json
Requires both datasets to be prepared. This script does not invent observations.
"""
import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from experiment_engine import run_directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full.local.json")
    parser.add_argument("--output", default="outputs/data_examples_measured.png")
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    data_dir = run_directory(cfg) / "data"
    a, b = data_dir / "mg.npy", data_dir / "electricity.npy"
    if not (a.exists() and b.exists()):
        raise FileNotFoundError("Prepare both datasets first: python src/experiment_engine.py --config configs/full.local.json --mode prepare")
    panels = [
        ("A. Mackey–Glass | simulated chaotic signal", np.load(a), 38_000,
         "Signal value (unitless)", "100 samples = 100 simulation-time units"),
        ("B. Electricity | sum of 321 customers", np.load(b), 20_000,
         "Aggregated load (dataset units)", "100 samples = 100 hours"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(12.8, 7.7), constrained_layout=True)
    fig.patch.set_facecolor("white")
    for ax, (title, series, origin, ylabel, footnote) in zip(axes, panels):
        x = np.arange(200)
        y = series[origin:origin+200]
        ax.axvspan(0, 99, color="#E7F5F1")
        ax.axvspan(100, 199, color="#FFF2DE")
        ax.plot(x[:100], y[:100], color="#08766C", lw=2, label="Observed history")
        ax.plot(x[99:], y[99:], color="#B66C12", lw=2, label="Held-out future")
        ax.axvline(99.5, color="#243349", ls="--", lw=1)
        ax.set(title=title, xlabel=f"Consecutive observations · {footnote}", ylabel=ylabel, xlim=(0,199))
        ax.grid(axis="y", alpha=.2)
        ax.legend()
    fig.suptitle("100 past measurements → 100 future predictions", fontsize=17, weight="bold")
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
