"""Create portable, local configurations from archived parameters.

Examples: python scripts/make_config.py smoke
          python scripts/make_config.py full
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    preset = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if preset not in ("smoke", "full"):
        raise SystemExit("Choose 'smoke' or 'full'")
    cfg = json.loads((ROOT / "configs" / "full_original.json").read_text())
    cfg["OUTPUT_ROOT"] = str(ROOT / "outputs")
    cfg["ELECTRICITY_ZIP"] = None
    if preset == "smoke":
        cfg.update(PRESET="smoke", DATASETS=["mg"], EPOCHS=2, BATCH_SIZE=8,
                   RNN_HIDDEN=16, MAX_TRAIN_ORIGINS=16, MAX_EVAL_ORIGINS=4,
                   VALIDATE_EVERY=1, MG_N=2500, MG_TRANSIENT=500,
                   ESN_FIT_STATES=800, RUN_TUNING=False)
    path = ROOT / "configs" / f"{preset}.local.json"
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
