# Reproducibility and audit status

**Verified in the packaging environment**

- The **uploaded executed** Kaggle notebook contains 115 cells, 54 recorded execution counts and 54 cells with output objects. Its cell source text matches the earlier clean 115-cell notebook exactly.
- `src/experiment_engine.py` and `src/analysis_report.py` are now the **exact code files from the user-uploaded Kaggle results ZIP**, not the mechanically reassembled source. `scripts/export_notebook_sources.py` remains available for regenerating a fresh source extract if desired.
- Both extracted Python modules pass syntax compilation (`python -m py_compile`).
- `scripts/audit_results.py` passes: **342 rows**, including **108 RNN, 108 LSTM, 108 GRU and 18 ESN**; every row's completion status is `ok`; the three logged seeds are 42, 123 and 2026.
- The dataset example plot was regenerated from the experiment's saved numerical `.npy` arrays in the user-supplied results archive. Its Mackey–Glass time-unit label was corrected to match the numerical generator's subsampling.
- Source README, original config, full run-level CSVs, validation selection records, ANOVA outputs, original environment information, individual run/epoch/tuning JSON logs and 13 extra exported Kaggle plots are packaged. The original 29 MB upload is also supplied in `local_only/` but Git ignores it.

**Not independently verified here**

- Re-executing the PyTorch models or their unit tests; `torch` was not available in the packaging runtime. The notebook contains authentic previous Kaggle outputs, which are a record of the previous run rather than an independent rerun.
- Overleaf/pdflatex compilation of `paper/main.tex` in this runtime (local TeX format files unavailable).
- Bitwise repeatability across CUDA/Python/PyTorch versions. `requirements.txt` is intentionally a broad installation specification, not a frozen lockfile.
- The originality, suitability for publication and completeness of the manuscript's claims; this requires the human authors and supervisor.
- A public GitHub URL, dataset redistribution permission or a journal submission. None exists merely because this ZIP exists.

**Recommended before handing in / publishing**

1. Run the Kaggle notebook end to end or at minimum the smoke config and self-tests in a PyTorch environment.
2. Compile `paper/main.tex` in Overleaf and inspect bibliography, float positions and any overflowing tables.
3. Verify exact author names, title, figure disclosures and dataset attribution.
4. Check dataset source licensing and choose a repository license with the coauthor.
5. Use `WINDOWS_GITHUB_PUSH.md` to create a repository, verify that the source electricity archive stays ignored, and insert the resulting GitHub permalink and, if applicable, commit hash into the manuscript.
