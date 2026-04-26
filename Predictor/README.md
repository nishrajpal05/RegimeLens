# NSE Regime Predictor

A real-time machine learning web app that predicts short-term market signals on Nifty 50 by adapting to changing market regimes.

This project is based on my 6th place solution in the iRage Kaggle competition and extends it to work on live financial data with a deployable dashboard.

---

## What this project does

* Fetches recent Nifty 50 market data (price + volume)
* Creates meaningful features like trends, volatility, and time-based signals
* Detects market regime shifts (i.e., when market behavior changes)
* Selects only stable features that work across different market conditions
* Trains an ensemble of ML models and combines them intelligently
* Generates next-day market signals
* Displays everything in a clean real-time dashboard

---

## Key Ideas 

* **Adversarial Validation**
  Checks how different recent market data is from historical data

* **Regime-Stable Features**
  Keeps only features that perform consistently across time

* **Ensemble Learning**
  Uses multiple LightGBM models and combines them using Ridge regression

* **Group DRO (Robust Training)**
  Focuses more on difficult market periods to improve stability

* **Custom Optimization Metric**
  Uses a competition-specific score to balance consistency and performance

---

## Tech Stack

* **Backend:** FastAPI (Python)
* **ML Models:** LightGBM, Ridge Regression
* **Frontend:** HTML, CSS, JavaScript (Chart.js)
* **Data Source:** Yahoo Finance (yfinance)

---

##  Project Structure

## Folder structure

```
Predictor/
├── backend/
│   ├── main.py          # FastAPI app + all routes
│   ├── data.py          # yfinance data fetcher
│   ├── features.py      # feature engineering, SO3_T norm, regime stability, adversarial AUC
│   ├── model.py         # LGB ensemble, DRO weights, proxy score, Ridge meta
│   └── cache/           # saved pipeline (auto-created after first train)
├── frontend/
│   ├── index.html       # single page dashboard
│   ├── style.css        # minimal classy styling
│   └── app.js           # API calls + Chart.js rendering
├── requirements.txt
└── README.md
```

---

##  How to Run

# Create virtual environment

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies

pip install -r requirements.txt

# Run backend

cd backend
uvicorn main:app --reload --port 8000

---

##  How to Use

1. Open: http://localhost:8000
2. Click **Train Pipeline** (first run takes ~2–4 minutes)
3. View:

   * Predicted signal
   * Market regime health
   * Model performance
   * Feature stability

After the first run, everything loads instantly from cache.

---

## API endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/api/status` | Pipeline status |
| POST | `/api/train` | Trigger training (background) |
| GET | `/api/predict` | Next-day signal + proxy stats |
| GET | `/api/history` | 120-day price + prediction overlay |
| GET | `/api/regime` | Adversarial AUC + feature stability |
| GET | `/api/blend` | Blend weights + per-model R² |

---

##  From Competition → Real World

This project takes a competition-grade ML pipeline and adapts it to:

* Work on live financial data
* Handle changing market conditions
* Provide real-time insights via a dashboard

---

##  Why this project matters

Most ML projects stop at training models.
This one goes further:

* Handles real-world noisy financial data
* Adapts to regime shifts (critical in trading)
* Deploys a full pipeline with UI + API

---
