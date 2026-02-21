# ERCOT Load Forecasting — Project Specification

---

## Dataset Plan (Phase 1 → Phase 2)

### Phase 1 Dataset (Fast + Credible)
- ERCOT historical hourly load as the primary signal.
- Minimal external features: calendar + lag features (no weather yet).

### Phase 2 Enhancement (Serious Realism)
- Add weather (temperature / humidity / wind) from a public source (NOAA or similar).
- Optional: bring in other ISO signals if available later.

> This keeps Milestone 1–3 clean and makes Milestone 4+ more impressive.

---

## Data Contract (Schema + Rules)

### Required Columns (Processed Dataset)

| Column | Type / Notes |
|--------|-------------|
| `timestamp` | Timezone-consistent, hourly, no duplicates |
| `load_mw` | `float` |
| `split` | `train` / `val` / `test` label saved once |

### Feature Columns (Phase 1)

**Calendar:**
- `hour_sin`, `hour_cos`
- `dow_sin`, `dow_cos`
- `is_weekend`
- `is_holiday` *(optional in Phase 1; can add Phase 2)*

**Lag / Rolling:**
- Lags: $y_{t-1}, y_{t-2}, \ldots, y_{t-24}, y_{t-168}$ *(can be implicit via sequence window)*
- Rolling mean/std: 24h and 168h windows *(computed from past only)*

### Integrity Constraints

- Hourly frequency after processing.
- Missing hours handled deterministically (drop or impute with a documented rule).
- **No leakage:** scalers and rolling computations must be train-fitted / past-only.

---

## Baselines (Non-Negotiable for Credibility)

All models are reported against:

1. **Seasonal naïve** — "same hour last week":
   $$\hat{y}_{t+h} = y_{t+h-168}$$

2. **Persistence** — last observed value:
   $$\hat{y}_{t+h} = y_t$$

3. **LSTM probabilistic (Gaussian head)** — ML baseline

4. **Transformer probabilistic (Gaussian head) + physics loss** — primary model

---

## Model Design (Conceptual)

### Input / Output Shapes

| Tensor | Shape |
|--------|-------|
| Input | `[batch, context_len=168, num_features]` |
| Output mean | `[batch, horizon=24]` |
| Output scale | `[batch, horizon=24]` where `scale = softplus(raw_scale) + eps` |

### Why Transformer as Primary Model

- Better long-context handling (weekly seasonality patterns).
- Parallel training.
- Cleaner multi-horizon decoding without recurrence limitations.

### LSTM Baseline Purpose

- Provides a strong classical sequence model baseline.
- Helps prove the gain is not "just deep learning," but architecture + constraints + probabilistic evaluation.

---

## Physics / Structure-Informed Loss Design (Phase 1)

### Primary Probabilistic Loss (Gaussian NLL)

Optimize likelihood (mean + uncertainty) rather than MAE-only.

### Constraint 1: Non-Negativity + Soft Upper Bound

- Penalize mean predictions < 0.
- Optional upper bound $U$ derived from training max + margin.

### Constraint 2: Ramp-Rate Realism (Multi-Horizon Trajectory)

Let $\Delta_h = \mu_h - \mu_{h-1}$. Penalize ramps beyond learned threshold $R_{\max}$ estimated from training data ramp quantiles (e.g., 99th percentile).

### Optional Constraint 3: Smoothness (Second Differences)

Stabilizes wild oscillations across the 24-step forecast.

### Total Loss

$$L = L_{\text{NLL}} + \lambda_{\text{bound}} L_{\text{bound}} + \lambda_{\text{ramp}} L_{\text{ramp}} \; (+\lambda_{\text{smooth}} L_{\text{smooth}})$$

> **Note:** Physics penalties apply primarily to $\mu$. Later, gentle regularization on $\sigma$ can prevent pathological uncertainty (e.g., exploding variance).

---

## Evaluation Design

### Point Metrics

- MAE (overall and per horizon)
- RMSE (overall and per horizon)

### Probabilistic Metrics

- NLL
- Coverage at 50 / 80 / 90 / 95% intervals
- Sharpness (interval width)
- *(Phase 2+)* CRPS as flagship probabilistic score

### Portfolio-Ready Plots

- Forecast fan chart (actual vs mean + intervals)
- Reliability diagram (nominal vs empirical coverage)
- Coverage-by-horizon curve
- Interval width-by-horizon curve
- Error heatmap by hour-of-day / day-of-week

### Ablations (Required)

- Transformer without physics vs. Transformer with physics
- Ramp only vs. Ramp + Bound (+ Smooth)

---

## Training Pipeline Design

### Splits

Time-based split:

| Split | Data |
|-------|------|
| Train | Earliest chunk |
| Val | Next chunk |
| Test | Most recent chunk |

*(Optional: rolling backtest in Milestone 4+)*

### Training Loop

- Early stopping on val NLL
- Checkpoint best model
- Log metrics per epoch
- Save run artifacts: resolved config snapshot, `metrics.json`, `predictions.parquet`, plots

### Reproducibility

- Seed control
- Deterministic settings documented (with caveats for GPU)

---

## Clean File Structure (Final Target)

```
configs/          # data / model / train / eval configs
src/
  config/
  data/
  models/
  losses/
  physics/
  training/
  evaluation/
  utils/
scripts/          # download / preprocess / train / eval / backtest
tests/            # dataset integrity + loss/constraint shape tests
reports/          # figures + metrics summaries
runs/             # per-experiment artifacts
```

---

## Milestone Roadmap

### Milestone 0 — Repo + Spec Lock
- Write project spec into README.
- Lock config schema and run artifact layout.

### Milestone 1 — Data Ingestion + Processed Dataset
- ERCOT hourly load ingestion.
- Deterministic cleaning + hourly index enforcement.
- Train/val/test split artifact saved.

### Milestone 2 — Baselines + LSTM Gaussian
- Seasonal naïve + persistence baselines.
- LSTM probabilistic baseline.
- First full evaluation report (accuracy + calibration).

### Milestone 3 — Transformer Gaussian (No Physics Yet)
- Transformer probabilistic model.
- Compare vs. LSTM + baselines.
- Confirm stability, remove training quirks.

### Milestone 4 — Physics Constraints + Ablations *(Core Novelty)*
- Add bound + ramp penalties.
- Ablation report.
- Show improvements in realism + calibration (not just MAE).

### Milestone 5 — Calibration + Polish
- Post-hoc calibration if needed (variance scaling).
- Reliability plots polished.
- CI tests and packaging polish.

### Milestone 6 — Upgrade Pathway to Variant B
- Swap dataset resolution & horizon config.
- Reuse same pipeline + same evaluation suite.
- Document migration clearly.