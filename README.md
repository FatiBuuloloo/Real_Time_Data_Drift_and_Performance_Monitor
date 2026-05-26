# MLOps Data Drift Monitoring for Calorie Prediction

## Overview

This project implements a full **MLOps monitoring** loop around a trained machine learning model. The core idea is straightforward: once a model was deployed, its inputs can shift over time. If the live data distribution diverges from the training distribution, model performance silently degrading.
This system catches that degradation before it becomes a problem, by:
- Logging every real-time user input to a SQLite database
- Transforming live data to match the training scale (Box-Cox via `PowerTransformer`)
- Computing KL Divergence between the live and baseline distributions for all features
- Surfacing drift signals through a visual, color-coded Streamlit dashboard

Dataset: [Kaggle]([https://www.salicon.net/](https://www.kaggle.com/datasets/ruchikakumbhar/calories-burnt-prediction))

Live Demo: [Click Here](https://viewww-mlops-calorie-prediction-dashboard.hf.space)

Repository: View on [Hugging Face](https://huggingface.co/spaces/Viewww/MLOps_Calorie_Prediction_Dashboard/tree/main)

## System Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                            STREAMLIT APPLICATION                            │
│                                                                             │
│  ┌─────────────────────────────┐         ┌────────────────────────────────┐  │
│  │       USER PREDICTION       │         │        MLOPS DASHBOARD         │  │
│  ├─────────────────────────────┤         ├────────────────────────────────┤  │
│  │ Input → Label Encoding      │         │ SQLite → Fetch Logs            │  │
│  │ Box-Cox → StandardScaler    │         │ Compute KL Divergence          │  │
│  │ XGBoost → Predict           │         │ Feature Drift Analysis         │  │
│  │ Log Result to SQLite        │         │ Visual Distribution Alerts     │  │
│  └──────────────┬──────────────┘         └────────────────┬───────────────┘  │
│                 │                                         │                   │
└─────────────────┼─────────────────────────────────────────┼───────────────────┘
                  │                                         │
                  │        ┌────────────────────────┐       │
                  └────────►    SQLITE (user_logs)  ◄───────┘
                           │ id • timestamp • raw   │
                           │ features • calories    │
                           └───────────┬────────────┘
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            │               ARTIFACTS & PRESETS                   │
            ├─────────────────────────────────────────────────────┤
            │ • xgb_model.pkl (XGBoost Regressor)                 │
            │ • baseline_distribution.pkl (PT, Scaler, LE, Bins)  │
            └─────────────────────────────────────────────────────┘
```

## MLOps Core: Drift Detection
### Why KL Divergence?

Kullback-Leibler (KL) Divergence measures the information lost when distribution Q (live data) is used to approximate distribution P (training baseline):
```
KL(P || Q) = Σ P(x) · log(P(x) / Q(x))
```
A score of `0` indicates identical distributions. As scores increase, so does the likelihood that model predictions are degrading.

### Thresholds
KL Score	Status	Action
`< 0.1`	🟢 Stable	No action needed
`0.1 – 0.5`	🟡 Moderate Drift	Investigate; consider retraining
`> 0.5`	🔴 Critical Drift	Immediate retraining recommended

### Preprocessing Alignment
A key engineering challenge in drift detection is scale alignment: the baseline histograms are computed on Box-Cox transformed data. When comparing against live inputs, the same `PowerTransformer` (fitted on training data) is applied to the live data first to ensure the KL comparison is done on the same scale.
```python
# Load fitted transformer from baseline pickle
pt = baseline["transformers"]["powertransformer"]

# Transform live data before comparison
live_transformed = pt.transform(live_data[NUM_COLS])

# Bin onto baseline histogram edges and compute KL
counts, _ = np.histogram(live_transformed[:, i], bins=baseline_bins)
q_prob = (counts + 1e-10) / sum(counts + 1e-10)
kl = sum(p * log(p / q) for p, q in zip(baseline_prob, q_prob))
```

## Dashboard Features
### Page 1 — User Prediction (Data Generator)

| Component | Description |
|---|---|
| Input Form | Biometric fields: Age, Height, Weight, Duration, Heart Rate, Body Temp, Gender |
| Quick Stats | Live BMI calculator, MHR estimate, HR zone percentage |
| Prediction Result | Calorie output with burn rate and intensity classification |
| SQLite Logging | Transformed raw inputs + prediction automatically written to `user_logs`|

### Page 2 — MLOps Dashboard (Drift Monitoring)

| Component | Description |
|---|---|
| Summary Metrics | Total logs, avg KL score, most-drifted feature, system health |
| Drift Cards | Per-feature KL score with color-coded badge (Green/Yellow/Red) |
| KL Bar Chart | Comparative bar chart with warning/critical threshold lines |
| Distribution Overlay | Histogram overlay: Baseline (P) vs Live (Q) for all 6 features |
| Prediction Trend | Time-series line chart of last 50 predictions |
| Live Log Table | Last 10 rows ingested into SQLite |


## Model Background
For this project, the model was trained using data from [Kaggle](https://www.kaggle.com/datasets/ruchikakumbhar/calories-burnt-prediction).
While the primary focus is on the MLOps layer, the underlying XGBoost model demonstrates strong performance:

| Model | RMSE | MAE |
|---|---|---|
| Linear Regression | 0.101 | 0.079 |
| Random Forest | 0.043 | 0.030 |
| SVR | 0.046 | 0.035 |
| **XGBoost** | **0.031** | **0.024** |
| Gradient Boosting | 0.046 | 0.034 |

XGBoost was selected for deployment due to its lowest error across both metrics. All numeric features were Box-Cox transformed via `PowerTransformer` prior to training; Gender was label-encoded.

## Project Structure
```
MLOps_Data_Drift_Monitoring_for_Calorie_Prediction/
├── app.py                          # Streamlit entry point + navigation
├── pages/
│   ├── prediction.py               # Page 1: User Prediction & Logging
│   └── dashboard.py                # Page 2: MLOps Drift Dashboard
├── calorie_prediction/
│   ├── xgb_model-calorie_prediction.pkl   # Trained XGBoost model
│   ├── baseline_distribution.pkl          # Baseline distributions + preprocessors + scaler
│   └── SQLite_calorie.db                  # Real-time user log database
├── requirements.txt
├── calorie_prediction.ipynb
└── README.md
```

## Technical Stack
| Layer | Technology |
|---|---|
| **Model** | XGBoost Regressor |
| **Preprocessing** | PowerTransformer (Box-Cox), LabelEncoder, StandardScaler |
| **Drift Metric** | KL Divergence |
| **Database** | SQLite (real-time logging) |
| **Frontend** | Streamlit (multi-page) |
| **Visualization** | Plotly (interactive charts) |
| **Deployment** | HuggingFace Spaces |
