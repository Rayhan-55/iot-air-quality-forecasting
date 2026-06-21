# IoT Sensor Data Trend Prediction — Air Quality Forecasting

An end-to-end machine-learning pipeline that forecasts the **next-hour CO
concentration** from a real, field-deployed gas multi-sensor device, handling the
noise and missingness typical of production IoT hardware. It compares a classical
model (**XGBoost**) against a deep sequential model (**LSTM**).

**Deliverable:** a single Colab-ready notebook,
[`IoT_Air_Quality_Forecasting.ipynb`](IoT_Air_Quality_Forecasting.ipynb), runnable
top-to-bottom with no manual downloads.

![Forecast comparison](assets/forecast_comparison.png)

---

## Results (held-out test set)

| Model    | Test RMSE | Test MAE |
|----------|-----------|----------|
| XGBoost  | **0.66**  | **0.45** |
| LSTM     | 0.89      | 0.66     |

Both models track the diurnal CO cycle; the regularised XGBoost follows the rush-hour
spikes more tightly and wins on both metrics. (Numbers are reproducible with the fixed
seed; minor variation is normal across runs/hardware.)

---

## How to run

**Google Colab (recommended):** open the notebook in Colab → *Runtime → Run all*.
The dataset is pulled automatically from a public mirror; `tensorflow`, `pandas`,
`scikit-learn` and `matplotlib` are preinstalled, and the first cell installs
`xgboost`.

**Locally:**
```bash
pip install -r requirements.txt
jupyter notebook IoT_Air_Quality_Forecasting.ipynb
```

---

## Dataset

**UCI Air Quality** (De Vito et al., 2008): 9,357 hourly records, March 2004 – April
2005, from 5 metal-oxide chemical sensors in an air-quality device deployed at road
level in an Italian city, with co-located certified-analyser ground truth for CO, NMHC,
Benzene, NOx and NO₂. The data shows real sensor drift, cross-sensitivities and
dropouts. A copy is included in [`data/`](data/) and is also auto-downloaded by the
notebook. License: CC BY 4.0.

---

## Must Explain

### 1. Industrial context and target variable
The sensors form a low-cost IoT air-quality monitor; the certified analyser is the
expensive reference instrument. We forecast the reference **`CO(GT)`** (true CO in
mg/m³) **one hour ahead** — an environmental-trend forecasting task. CO is the headline
traffic-related pollutant here, has a strong diurnal (rush-hour) cycle, and ~18% of its
readings are missing, which makes it a realistic data-engineering target.

### 2. Data-cleaning and feature-engineering justification
*Cleaning.* The device writes **`-200`** whenever a reading fails, so the first step is
converting every `-200` to `NaN` (leaving it numeric would wreck the model). The
`NMHC(GT)` channel is **dropped** (>90% missing — imputing it would be fabrication). We
**reindex onto a complete hourly grid** so missing timestamps become explicit `NaN`
rows; this matters because lag/rolling features assume a uniform step and silent gaps
would corrupt them. Remaining physically impossible negatives are nulled (out-of-bounds
guard). Imputation is deliberately conservative: **time-based linear interpolation for
short gaps (≤ 6 h)** only, then ffill/bfill for the residue — we never interpolate
across long outages.

*Feature engineering.* Every feature uses information up to hour *t* only (the label is
the future value, so there is no leakage):
- **Lag features** (`t-1…t-24`) — pollution is strongly autocorrelated; the 24-h lag
  captures the daily cycle.
- **Trailing rolling mean/std** (3/6/24 h, shifted by 1 so the current value can't leak)
  — local level and recent volatility.
- **Co-located sensors + weather** (`PT08.*`, `C6H6`, `T`, `RH`, `AH`) — informative
  exogenous signals.
- **Cyclical calendar** (`sin/cos` of hour and weekday) — rush-hour / weekend structure
  without a 23→0 discontinuity.

### 3. Model architecture and overfitting control on time-series
We compare two families. **XGBoost**: shallow trees (`max_depth=5`), row/column
subsampling, L2 regularisation, and **early stopping** on a validation RMSE. **LSTM**: a
single 48-unit layer over a 24-hour look-back window, **dropout 0.2**, **early stopping**
with best-weight restore, and feature scalers fit on the **training set only**.

The central anti-leakage measure is a **strictly chronological 70/15/15 split** — the
test period is entirely future relative to training, so evaluation mimics real
deployment (no random shuffling, which would let the model see the future). The small
train-vs-test RMSE gap for XGBoost and the non-diverging LSTM validation curve (both
shown in the notebook) indicate neither model is badly overfit.

---

## Repo layout
```
iot-air-quality-forecasting/
├── IoT_Air_Quality_Forecasting.ipynb   # main deliverable (executed, with outputs)
├── data/
│   └── AirQualityUCI.csv                # UCI Air Quality (mirror copy)
├── assets/
│   └── forecast_comparison.png          # ground truth vs predictions
├── requirements.txt
└── README.md
```


