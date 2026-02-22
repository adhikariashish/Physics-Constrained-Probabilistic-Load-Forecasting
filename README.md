## Project Spec (Milestone 0)

This repository implements an industry-grade, reproducible pipeline for:

**Physics-Constrained Probabilistic Load Forecasting**
- Resolution: **Hourly**
- Context window: **168 hours**
- Forecast horizon: **24 hours**
- Outputs: per-horizon **mean (μ)** and **uncertainty (σ)**

### Configuration System (Single Source of Truth)

All experiments are driven by YAML configs in `configs/`:

- `configs/data.yaml` — dataset, schema, splits, feature switches
- `configs/model.yaml` — model backbone + probabilistic head settings
- `configs/train.yaml` — optimization, reproducibility, physics constraints, logging
- `configs/eval.yaml` — metrics, calibration, plotting and reporting outputs

Configs are validated by `src/config/schema.py` and loaded via `src/config/load.py`.
Each run saves a resolved snapshot to guarantee reproducibility.

### Run Artifact Contract

Every training run writes to:

`runs/<run_id>__<train.run.name>/`

with the following structure:

- `config_resolved.yaml` — merged config snapshot used for the run
- `checkpoints/`
  - `best.pt`
  - `last.pt`
- `logs/`
  - `train.log`
  - `metrics_epoch.csv`
- `metrics/`
  - `metrics_summary.json`
  - `metrics_by_horizon.csv`
  - `calibration.json`
- `predictions/`
  - `val_predictions.parquet`
  - `test_predictions.parquet`
- `plots/`
  - `forecast_examples.png`
  - `coverage_by_horizon.png`
  - `reliability_diagram.png`
  - `interval_width_by_horizon.png`

The project is designed so downstream modules (e.g., dashboards or future RL controllers)
can consume standardized run artifacts without coupling to training internals.

### Probabilistic Objective + Physics Constraints (Concept)

Training optimizes a probabilistic loss (Gaussian NLL) plus soft physics/structure penalties:

- Non-negativity and upper bounds
- Ramp-rate realism across multi-horizon trajectories
- (Optional) smoothness regularization

These are controlled via `train.physics.*` in `configs/train.yaml`.