"""
Trains the demand forecasting model, reproducing the pipeline from
Demand_Forecasting.ipynb, and saves the artifacts the Flask app needs:

  model/xgb_demand_model.pkl   -> the trained XGBRegressor
  model/sales_history.csv      -> date, sales, time_index for the full series
  model/metrics.json           -> hold-out MAE / RMSE / R2

Run:
    python train_model.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-car-sales.csv"

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

# Best params found via GridSearchCV in the notebook.
BEST_PARAMS = {
    "learning_rate": 0.01,
    "max_depth": 3,
    "n_estimators": 700,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}

MODEL_DIR = Path(__file__).parent / "model"


def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_URL)
    df.columns = ["date", "sales"]
    df["date"] = pd.to_datetime(df["date"])
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter

    df["lag1"] = df["sales"].shift(1)
    df["lag2"] = df["sales"].shift(2)
    df["lag3"] = df["sales"].shift(3)
    df["lag6"] = df["sales"].shift(6)
    df["lag12"] = df["sales"].shift(12)

    df["time_index"] = range(len(df))

    df["rolling_mean_3"] = df["sales"].rolling(3).mean()
    df["rolling_mean_6"] = df["sales"].rolling(6).mean()
    df["rolling_std_3"] = df["sales"].rolling(3).std()
    df["rolling_std_6"] = df["sales"].rolling(6).std()

    return df


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)

    raw = load_raw_data()
    featured = add_features(raw).dropna().reset_index(drop=True)

    X = featured[FEATURE_COLUMNS]
    y = featured["sales"]

    split = int(len(featured) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = XGBRegressor(**BEST_PARAMS)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    print(f"MAE: {mae:.2f}  RMSE: {rmse:.2f}  R2: {r2:.4f}")

    # Refit on the full dataset so the deployed model uses all available history.
    model.fit(X, y)

    joblib.dump(model, MODEL_DIR / "xgb_demand_model.pkl")

    # Save the full (unfiltered) date/sales/time_index series so the Flask app
    # can recompute lag/rolling features for arbitrary future forecast dates.
    history = raw.copy()
    history["time_index"] = range(len(history))
    history[["date", "sales", "time_index"]].to_csv(
        MODEL_DIR / "sales_history.csv", index=False
    )

    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump({"mae": mae, "rmse": rmse, "r2": r2}, f, indent=2)

    print(f"Saved model + history to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
