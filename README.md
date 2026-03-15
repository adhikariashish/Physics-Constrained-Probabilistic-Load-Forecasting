# ⚡ Physics-Constrained Probabilistic Load Forecasting

A deep learning system for **24-hour ahead probabilistic power demand forecasting** with uncertainty estimation and **physics-based constraints**, built using PyTorch Transformers and deployed through an interactive Streamlit Dashboard.

The system produces **probabilistic forecasts**, peak-risk alerts, and uncertainty intervals to support **power grid planning and operational decision-making.**

The project compares:

- LSTM probabilistic forecasting
- Transformer probabilistic forecasting
- Transformer with physics-informed constraints

and deploys the best model in an **interactive Streamlit dashboard**.

---

## Project Overview

Electric grid operators must forecast electricity demand accurately to:

- maintain grid stability
- plan generation dispatch
- avoid over/under supply

This project builds a **probabilistic load forecasting pipeline** that predicts:

- mean load forecast
- uncertainty (variance)
- 80% prediction interval
- 95% prediction interval

for the **next 24 hours**.

The system incorporates **physical constraints of power systems** such as:

- ramp rate limits
- operational bounds
- smooth temporal transitions

---
## System Key Features

### Probabilistic Load Forecasting

The system predicts electricity demand for the next 24 hours, providing a comprehensive view of potential outcomes rather than a single data point:

- **Mean Load Forecast:** The expected average demand.
- **Standard Deviation:** A measure of forecast uncertainty.
- **80% Prediction Intervals:** The range where load is likely to fall with 80% confidence.
- **95% Prediction Intervals:** The range for high-certainty risk planning.

---
## Model Architecture

**Transformer backbone:**
```
Historical Load (168 hrs)
        ↓
Temporal Feature Encoding
  (hour sin/cos, day-of-week)
        ↓
  Transformer Encoder
        ↓
   Distribution Head
     (mean, sigma)
        ↓
 Probabilistic Forecast
      (24 hours)
```
### Transformer-Based Forecast Model

The engine utilizes a Transformer architecture specifically optimized for time-series data, capturing complex patterns that traditional models might miss.

**Input Features:**

- **Historical Load:** Past performance data.
- **Temporal Encoding:** Hour of day and day of week (using `sin/cos` encoding to maintain cyclical relationships).
- **Calendrical Context:** Weekend and holiday indicators.

> **Note:** The model analyzes a 168-hour historical window to learn long-term temporal dependencies and weekly cycles.

---

### Physics-Constrained Learning

To ensure the model doesn't produce mathematically "correct" but physically impossible results, physics-inspired penalties are incorporated during training:

| Constraint | Purpose |
|---|---|
| Ramp Penalty | Prevents unrealistic, instantaneous jumps in electrical load. |
| Bound Constraint | Ensures forecasts stay within the physical limits of the grid. |
| Smoothness Penalty | Encourages realistic, continuous load trajectories. |

These constraints significantly improve the **stability and realism** of generated predictions.

**Physics-informed loss:**
```
Total Loss = Forecast Loss + λ * Physics Penalty
```
---

### Probabilistic Risk Detection

Forecast outputs are automatically screened to identify potential grid stress. The dashboard highlights risk hours directly on the forecast chart based on the following tiers:

| Risk Level | Description |
|---|---|
| Normal | Load is within the safe operating range. |
| Moderate | Load is approaching system capacity limits. |
| High | Critical potential for grid stress or peak demand failure. |
---


## Dataset

**Source:** U.S. electricity system load data

**Features used:**

- `load_mw`
- `hour_sin`
- `hour_cos`
- `dow_sin`
- `dow_cos`
- `is_weekend`
- `is_holiday`

**Context window:** 168 hours (7 days)

**Forecast horizon:** 24 hours

---

## Experiments

Three models were trained and evaluated.

| Model | MAE (MW) | RMSE (MW) | NLL |
|---|---|---|---|
| LSTM Gaussian | ~2100 | ~2750 | baseline |
| Transformer Gaussian | ~2000 | ~2600 | improved |
| Transformer + Physics | ~1991 | ~2594 | best |

Physics constraints slightly improve **stability and generalization**.

---

## Forecast Output Example

The model produces a probabilistic forecast:
```json
{
  "timestamp": "2026-01-01T00:00:00",
  "mean_mw": 45583,
  "sigma_mw": 1246,
  "lower_80_mw": 43985,
  "upper_80_mw": 47181,
  "lower_95_mw": 43139,
  "upper_95_mw": 48027
}
```

This allows grid operators to evaluate **forecast uncertainty and risk**.

---

## Results

### Forecast Error by Horizon

![MAE comparison](reports/figures/comparison/exp_transformer_physics_v0_vs_exp_transformer_physics_lam005_v0_vs_exp_transformer_physics_lam001_v0_mae_mw_test.png)

### RMSE by Horizon

![RMSE comparison](reports/figures/comparison/exp_transformer_physics_v0_vs_exp_transformer_physics_lam005_v0_vs_exp_transformer_physics_lam001_v0_rmse_mw_test.png)

### NLL by Horizon

![NLL comparison](reports/figures/comparison/exp_transformer_physics_v0_vs_exp_transformer_physics_lam005_v0_vs_exp_transformer_physics_lam001_v0_nll_test.png)

---

## Streamlit Dashboard

The dashboard visualizes the 24-hour probabilistic forecast with uncertainty bands and risk detection.

**Features:**

- Mean Forecast trajectory
- 80% and 95% confidence intervals
- peak load marker
- High-risk and moderate-risk hour detection
- Forecast metadata and peak summary

**Run:**
```bash
streamlit run app/dashboard.py
```

![Dashboard](app/dashboard.png)

---

## Running the Project

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python -m src.training.trainer --config configs/train.yaml
```

### 3. Generate forecast
```bash
python scripts/forecast_next_24h.py --run runs/exp_transformer_physics_v0
```

### 4. Launch dashboard
```bash
streamlit run app/dashboard.py
```

---

## Project Structure
```
physics-constrained-prob-load-forecast/
│
├── app/
│   └── dashboard.py
│
├── configs/
│   └── train.yaml
│
├── scripts/
│   └── forecast_next_24h.py
│
├── src/
│   ├── data/
│   ├── models/
│   ├── losses/
│   ├── physics/
│   ├── training/
│   └── evaluation/
│
├── reports/
│   ├── figures/
│   ├── metrics/
│   └── latest_forecast/
│
├── requirements.txt
└── README.md
```

---

## Key Techniques Demonstrated

- probabilistic forecasting
- transformer time-series models
- physics-informed learning
- uncertainty estimation
- horizon-wise evaluation
- ML experiment tracking
- deployment with Streamlit

---

## Future Work

Possible improvements:

- multi-region forecasting
- weather feature integration
- reinforcement learning for grid control
- online model updating
- real-time streaming pipeline

---

## Author

**Ashish Adhikari**


## License

MIT License