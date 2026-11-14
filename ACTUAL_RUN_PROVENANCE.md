# Verification against the actual Kaggle uploads (14 November 2026)

You uploaded:

1. `notebook603521a0f2.ipynb` — the **actually executed** 115-cell notebook, including 54 cells with execution counts and 54 with recorded output objects.
2. `results (11).zip` — the exact Kaggle results export, containing the experiment engine, analysis script, 342-run records, parameter tuning, plots, derived data, and an embedded full-results archive.

The original clean notebook contained the **same cell source text** in the same order. The uploaded executed notebook adds the output records. Its exact bytes are preserved as `notebooks/BAI301_Kaggle_T4x2_EXECUTED.ipynb`.

The extracted source scripts in `src/` are the **original `experiment_engine.py` and `analysis_report.py` from your uploaded results ZIP**, not reconstructions. The archived complete-run CSV in the earlier prepared repository has the same SHA-256 as your newly uploaded CSV.

| Item | SHA-256 |
|---|---|
| Uploaded executed notebook | `c1734556b76227bb16e81eb0866d8ff204cc3fcbc0434ed447b8265d33bc7ef8` |
| Uploaded `results (11).zip` | `96930d48cf3f07947ba35b137e50fd94a0bf8b3fff81f426844a0b4cb4e3dd91` |
| Embedded `BAI301_full_results.zip` | `062906f7867580007aa286da7f679a8fcc939cd71fcdd9fedf2d3aa06ac67b9a` |
| `results/all_runs.csv` | `d2bffd36b31efcdfa8d56f0bea7930d5d1787bf44e710e0d4269a34d6e2603a6` |

The 29 MB original export is preserved inside this downloadable ZIP at `local_only/Original_Kaggle_Results.zip` with unchanged bytes. `local_only/` is excluded from Git commits so the archive's bundled source electricity data is not inadvertently redistributed. All tracked `results/` CSVs, log JSONs, worker logs, and exported figures are original experiment outputs, not regenerated numbers.

Run `python scripts/check_run_provenance.py` from this repository root to verify the linked evidence. The original ZIP is optional once the repository has been pushed; keep it locally as your backup.
