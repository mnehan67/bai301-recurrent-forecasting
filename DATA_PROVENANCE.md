# Dataset provenance and exact task definitions

## Mackey–Glass (synthetic benchmark)

- Generator: documented forward-Euler integration of the classical delayed differential equation (see `paper/main.tex` and `src/experiment_engine.py`).
- Integration step: **0.1** simulated time units.
- Recorded observation spacing: **1.0** simulated time unit (one observation per 10 Euler updates).
- Delay parameter: **17** time units; transient dropped: **2,000** time units; retained: **50,000** observations.
- Context/horizon: **100/100 recorded observations**, representing 100/100 simulated time units.
- This is *not* a set of real sensor readings; its purpose is to stress recursive prediction under nonlinear dynamics.

## Electricity (real-world aggregate; not the original panel task)

- Source: [Monash Time Series Forecasting Archive, Zenodo record 4656140](https://zenodo.org/records/4656140).
- Requested archive in the notebook: `electricity_hourly_dataset.zip`, with source checksum stored by the engine.
- Original dataset: hourly customer electricity use. The experiment aggregates **321 customers into one univariate time series of 26,304 hours**, with per-series forward fill and zero for initial unavailable values.
- Context/horizon: **100/100 hours**.
- The aggregated modeling task is different from forecasting the original 321 series separately. Do not compare its NRMSE with published original-panel scores.

## Leakage prevention and source redistribution

- Both tasks are chronologically split **70/15/15**.
- Z-score mean and standard deviation are computed on training data only.
- Forecast windows are restricted to their partition; test evaluation never feeds its own future truth back into the model.
- The original electricity archive and derived `.npy` sequences **are intentionally excluded from the tracked public GitHub content** pending confirmation of redistribution rights. The entire original Kaggle ZIP is supplied in this downloadable bundle under `local_only/`, which is excluded by `.gitignore`; keep it locally, not on GitHub. The reproducible download and transformation code is included.
- The full results tables, figures and run logs are included; authors should review any public redistribution restrictions that may apply to source-derived content.
