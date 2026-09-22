"""
Flask app serving the demand forecasting model trained by train_model.py.

Run:
    python train_model.py      # once, to create model/xgb_demand_model.pkl
    python app.py               # starts the server on http://localhost:5000

Endpoints:
    GET  /                 -> simple UI (history chart + forecast form)
    GET  /api/history       -> JSON of the historical monthly sales series
    POST /api/predict       -> JSON forecast for the next N months
        body: {"months": 3}
"""
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

MODEL_DIR = Path(__file__).parent / "model"
MODEL_PATH = MODEL_DIR / "xgb_demand_model.pkl"
HISTORY_PATH = MODEL_DIR / "sales_history.csv"

FEATURE_COLUMNS = [
    "month",
    "year",
    "quarter",
    "time_index",
    "lag1",
    "lag2",
    "lag3",
    "lag6",
    "lag12",
    "rolling_mean_3",
    "rolling_mean_6",
    "rolling_std_3",
    "rolling_std_6",
]

app = Flask(__name__)

if not MODEL_PATH.exists() or not HISTORY_PATH.exists():
    raise FileNotFoundError(
        "Model artifacts not found. Run `python train_model.py` first "
        f"(expected {MODEL_PATH} and {HISTORY_PATH})."
    )

model = joblib.load(MODEL_PATH)
history_df = pd.read_csv(HISTORY_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)


def build_next_month_features(sales_series: pd.Series, next_date: pd.Timestamp, next_time_index: int) -> pd.DataFrame:
    """
    Build the feature row for `next_date`, given all sales known so far
    (sales_series, chronologically ordered, most recent last).

    Note: the notebook's rolling_mean/rolling_std features are computed on the
    CURRENT row's sales value (a look-ahead feature during training). For a
    genuinely unknown future month we can't do that, so we approximate them
    using only the most recent known months (i.e. the window ending at the
    last observed month rather than the forecast month itself).
    """
    def lag(n):
        return sales_series.iloc[-n] if len(sales_series) >= n else None

    recent_3 = sales_series.iloc[-3:]
    recent_6 = sales_series.iloc[-6:]

    return pd.DataFrame([{
        "month": next_date.month,
        "year": next_date.year,
        "quarter": next_date.quarter,
        "time_index": next_time_index,
        "lag1": lag(1),
        "lag2": lag(2),
        "lag3": lag(3),
        "lag6": lag(6),
        "lag12": lag(12),
        "rolling_mean_3": recent_3.mean(),
        "rolling_mean_6": recent_6.mean(),
        "rolling_std_3": recent_3.std(),
        "rolling_std_6": recent_6.std(),
    }])[FEATURE_COLUMNS]


def forecast(months: int) -> list[dict]:
    sales_series = history_df["sales"].copy()
    last_date = history_df["date"].iloc[-1]
    last_time_index = int(history_df["time_index"].iloc[-1])

    results = []
    for step in range(1, months + 1):
        next_date = (last_date + pd.DateOffset(months=step)).replace(day=1)
        next_time_index = last_time_index + step

        features = build_next_month_features(sales_series, next_date, next_time_index)
        predicted_sales = float(model.predict(features)[0])

        results.append({"date": next_date.strftime("%Y-%m-%d"), "predicted_sales": round(predicted_sales, 2)})

        # feed the prediction back in so lag/rolling features for the
        # following step can be computed (recursive multi-step forecasting)
        sales_series = pd.concat([sales_series, pd.Series([predicted_sales])], ignore_index=True)

    return results


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/history")
def api_history():
    records = history_df[["date", "sales"]].copy()
    records["date"] = records["date"].dt.strftime("%Y-%m-%d")
    return jsonify(records.to_dict(orient="records"))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True) or {}
    try:
        months = int(payload.get("months", 1))
    except (TypeError, ValueError):
        return jsonify({"error": "months must be an integer"}), 400

    if months < 1 or months > 24:
        return jsonify({"error": "months must be between 1 and 24"}), 400

    return jsonify({"forecast": forecast(months)})


if __name__ == "__main__":
    app.run(debug=True)
