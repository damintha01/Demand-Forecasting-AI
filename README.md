# Demand Forecasting AI

A demand forecasting application that predicts future monthly sales using an
XGBoost regression model. The project trains on historical monthly car sales
data, engineers time-series features (calendar, lag, and rolling-window
statistics), and serves recursive multi-step forecasts through a Flask web
app with an interactive chart UI.

## Overview

- **Model**: `XGBRegressor` trained on engineered time-series features
- **Data**: [Monthly car sales dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-car-sales.csv) (Jason Brownlee's dataset repo)
- **Serving**: Flask REST API + a simple HTML/JS front end for exploring history and generating forecasts
- **Development**: Model design and experimentation captured in `Demand_Forecasting.ipynb`; `train_model.py` reproduces that pipeline as a standalone, repeatable script

## Features

- Recursive multi-step forecasting (predict up to 24 months ahead)
- Feature engineering: month, year, quarter, time index, lags (1/2/3/6/12 months), and rolling mean/std (3/6 months)
- Hyperparameters tuned via `GridSearchCV` (see `BEST_PARAMS` in `train_model.py`)
- Hold-out evaluation with MAE, RMSE, and R² metrics saved to `model/metrics.json`
- Lightweight web UI showing historical sales alongside forecasted values

## Tech Stack

| Component        | Technology       |
|-------------------|------------------|
| Modeling           | XGBoost, scikit-learn |
| Data processing    | pandas, NumPy    |
| Web framework      | Flask            |
| Model persistence  | joblib           |

## Project Structure

```
Demand-Forecasting-AI/
├── app.py                     # Flask app: serves UI, history, and forecast endpoints
├── train_model.py             # Trains the model and saves artifacts to model/
├── Demand_Forecasting.ipynb   # Exploratory analysis, feature engineering, and model tuning
├── requirements.txt           # Python dependencies
├── templates/
│   └── index.html             # Front-end UI (chart + forecast form)
└── model/                     # Generated at train time (not tracked in git)
    ├── xgb_demand_model.pkl   # Trained XGBRegressor
    ├── sales_history.csv      # Full date/sales/time_index series
    └── metrics.json           # Hold-out MAE / RMSE / R²
```

## Getting Started

### Prerequisites

- Python 3.10+
- Internet access on first run (`train_model.py` downloads the source dataset)

### Installation

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### Train the model

```bash
python train_model.py
```

This downloads the dataset, engineers features, performs an 80/20 chronological
train/test split, fits the model, prints hold-out metrics, refits on the full
dataset, and writes the following to `model/`:

- `xgb_demand_model.pkl` — the trained model
- `sales_history.csv` — full historical series used to build features for future forecasts
- `metrics.json` — MAE, RMSE, and R² on the hold-out set

### Run the app

```bash
python app.py
```

The app starts at `http://localhost:5000`.

## API Reference

### `GET /`
Renders the UI: a chart of historical sales plus a form to request a forecast.

### `GET /api/history`
Returns the historical monthly sales series.

```json
[
  { "date": "1960-01-01", "sales": 6550 },
  { "date": "1960-02-01", "sales": 8728 }
]
```

### `POST /api/predict`
Returns a recursive multi-step forecast for the requested number of months
(1–24).

**Request**
```json
{ "months": 3 }
```

**Response**
```json
{
  "forecast": [
    { "date": "1968-01-01", "predicted_sales": 15234.12 },
    { "date": "1968-02-01", "predicted_sales": 15890.47 },
    { "date": "1968-03-01", "predicted_sales": 16110.05 }
  ]
}
```

Each forecasted value is fed back into the feature pipeline (lags and rolling
statistics) to generate the next step's prediction.

## Modeling Notes

### Feature set

```
month, year, quarter, time_index,
lag1, lag2, lag3, lag6, lag12,
rolling_mean_3, rolling_mean_6, rolling_std_3, rolling_std_6
```

### Rolling feature caveat

In the notebook, `rolling_mean_*` / `rolling_std_*` are computed on the
*same row's* sales value, which is a mild look-ahead during training and
evaluation (the window includes the month being predicted). For genuine
future forecasting this information isn't available, so `app.py`'s
`build_next_month_features` approximates these features using only the most
recently known months instead (a window ending at the last observed month).

As a result, real-world forecast accuracy will likely be somewhat lower than
the notebook's reported test R² (~0.75–0.79). The correct long-term fix is to
retrain with the rolling window shifted by one period
(`.shift(1).rolling(...)`) so training matches inference.

## License

No license specified.
