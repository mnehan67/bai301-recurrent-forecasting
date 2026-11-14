# When Training Protocols Change the Winner

**BAI301 Option B | Ajman University | Research/reproducibility package**

**Actual run included:** `notebooks/BAI301_Kaggle_T4x2_EXECUTED.ipynb` is the exact notebook exported after the 342-run Kaggle session, with recorded outputs. The earlier no-output notebook is also preserved as `notebooks/BAI301_Kaggle_T4x2_FULL.ipynb` for a cleaner rerun. The run's **original** 29 MB export is retained as `local_only/Original_Kaggle_Results.zip` in the *downloadable* package but is deliberately ignored by Git so the source electricity archive is not accidentally uploaded. All original result CSVs, run/epoch/tuning logs, four worker logs and 13 exported visualization images are available in the tracked `results/` directory.

**Windows users:** Follow [WINDOWS_GITHUB_PUSH.md](WINDOWS_GITHUB_PUSH.md) for step-by-step commands from `C:\Users\muham\Downloads\xxxx`.

This is a controlled experiment asking whether the apparent best recurrent forecasting model changes when **TBPTT**, **gradient clipping** and **teacher forcing** change. The original full experiment recorded **342 completed runs** over two *univariate* time series, not 342 different datasets.

| Task | Source | Data shown to model | Forecasted future |
|---|---|---|---|
| Mackey–Glass | Simulated chaotic delay equation; 50,000 observations | 100 samples = 100 simulated time units | Next 100 samples |
| Electricity | Real Monash hourly data, 321 customers summed into one 26,304-hour signal | Last 100 hours | Next 100 hours |

The experiment compares vanilla **RNN, LSTM, GRU** (approximately matched parameter counts), plus a distinct **echo state network (ESN)** with fixed reservoirs and ridge readouts. The ESN is not optimized or timed in the same way as the other three models.

## Recorded results, not invented predictions

- **324** factorial RNN/LSTM/GRU runs: 2 datasets × 3 architectures × 3 TBPTT lengths × 2 clipping choices × 3 feedback choices × 3 seeds.
- **18** ESN runs: 2 datasets × 3 reservoir sizes × 3 seeds.
- Validation-selected Mackey–Glass ESN: **test NRMSE 0.0208 ± 0.0075** across seeds.
- Validation-selected electricity GRU: **test NRMSE 0.3318 ± 0.0387** across seeds.
- **4 of 9 electricity ESN runs** exhibited extremely large *finite* rollout errors. Completion is not stability.
- Architecture and feedback policy interacted in an exploratory factorial analysis on both tasks. Three seeds and overlapping windows make the statistical evidence provisional.

These numbers come from `results/all_runs.csv` and companion selection/analysis CSVs. **No diffusion model or U-Net was trained.** ImageGen created two *conceptual illustrations for the paper*, not the plotted numerical results.

## Inspect the completed experiment (no GPU required)

From the repository root:

```bash
python scripts/audit_results.py
```

This checks the 342 recorded runs and their architecture/seed/status counts using only Python's standard library. `python scripts/check_run_provenance.py` additionally checks the executed notebook and log counts (and, if present, the original Kaggle ZIP checksum). Read `results/SUMMARY.md`, `results/validation_selected_models.csv`, `results/esn_selected_runs.csv`, `results/anova_interactions.csv`, `results/figures/`, and `results/kaggle_output_images/` to inspect detailed findings. Each recorded run's JSON and epoch log is included under `results/reproduction_archive/exp_b8f83540bd67fab5/`.

## Reproduce a small run locally

Use Python 3.10+ (a separate environment is recommended). On Kaggle, PyTorch is preinstalled; locally, install from `requirements.txt`:

```bash
python -m pip install -r requirements.txt
python scripts/make_config.py smoke
python src/experiment_engine.py --config configs/smoke.local.json --mode test
python src/experiment_engine.py --config configs/smoke.local.json --mode prepare
python src/experiment_engine.py --config configs/smoke.local.json --mode run --device cpu
```

The **smoke** preset uses one *synthetic* dataset, fewer training windows, 2 epochs and 4 configurations so it is **not** a reproduction of the reported full-grid scores. It is a basic check that the system runs. Generated files go into `outputs/`.

## Reproduce the original experiment on Kaggle

Start by viewing `notebooks/BAI301_Kaggle_T4x2_EXECUTED.ipynb`, which contains the actual executed Kaggle outputs. To reproduce cleanly, use `notebooks/BAI301_Kaggle_T4x2_FULL.ipynb` (same cell source without outputs); its companion `notebooks/BAI301_Kaggle_QUICKSTART.ipynb` provides setup instructions. Select **2×T4 GPUs**, enable Internet for the Monash/Zenodo electricity download, run all cells, and download results. The notebook contains both experiment and analysis scripts.

The equivalent **full** CLI steps are below. This may take considerable GPU time and must complete before creating the report:

```bash
python scripts/make_config.py full
python src/experiment_engine.py --config configs/full.local.json --mode test
python src/experiment_engine.py --config configs/full.local.json --mode prepare
python src/experiment_engine.py --config configs/full.local.json --mode tune --device cuda:0
python src/experiment_engine.py --config configs/full.local.json --mode run --device cuda:0
python src/analysis_report.py --config configs/full.local.json
python scripts/plot_data_explainer.py --config configs/full.local.json
```

The CLI sequence above is a **single-worker** option; the supplied notebook is the authoritative dual-T4 workflow. If working offline, supply a manually downloaded `electricity_hourly_dataset.zip` path by setting `ELECTRICITY_ZIP` in the generated full config. `configs/full_original.json` records the settings of the archived run (including its original Kaggle output path); the helper rewrites only paths and, for smoke mode, shrinks the workload.

## Repository guide

| Path | Contents |
|---|---|
| `paper/` | Elsevier-style two-column research working draft, BibTeX and three measured plots plus two conceptual images |
| `notebooks/` | Exact executed notebook **with outputs**, matching clean-source notebook and quickstart |
| `src/experiment_engine.py` | **Exact file from the uploaded Kaggle `results (11).zip`** |
| `src/analysis_report.py` | **Exact analysis file from the uploaded Kaggle export** |
| `scripts/` | Config creator, results audit, numerical data visualizer, notebook source extractor |
| `configs/full_original.json` | Archived experimental configuration |
| `results/` | Original 342-run CSVs, selected results, ANOVA, figures, 13 Kaggle output images, four worker `.log` files, environment info and JSON logs |
| `local_only/` | Original 29 MB Kaggle ZIP retained in downloadable archive; excluded from `git add .` by `.gitignore` |

See [DATA_PROVENANCE.md](DATA_PROVENANCE.md) for data sourcing, [ACTUAL_RUN_PROVENANCE.md](ACTUAL_RUN_PROVENANCE.md) for byte-level matching and [REPRODUCIBILITY_NOTES.md](REPRODUCIBILITY_NOTES.md) for verified vs unverified checks.

## Important research limitations

1. Mackey–Glass is a **simulated benchmark**; the electricity task is only **one real aggregated signal**. The study does not establish general superiority across real-world forecasting datasets.
2. ESNs use different training targets and CPU fitting, so direct ESN-vs-GPU fitting speed/memory rankings are not defensible.
3. Selection uses validation, but some analysis explores many models/policies; three seeds are not independent experimental datasets.
4. `paper/` is a **journal-styled working draft**, not yet journal-submission ready. Coauthors must approve it, and more independent real datasets and stronger analysis would be needed for a credible journal claim.

## Publish on GitHub

This ZIP is **GitHub-ready but not yet published**. After checking author details, licensing and private data:

```bash
git init
git add .
git commit -m "Release reproducible BAI301 recurrent forecasting study"
```

Then create a repository under your account and follow [WINDOWS_GITHUB_PUSH.md](WINDOWS_GITHUB_PUSH.md) to add a remote and push. **No explicit repository license has been granted here**; choose one together with your coauthor before making it public. Do not force-add `local_only/` or redistribute the downloaded electricity archive until its dataset terms have been checked.
