"""
model.py - train / evaluate / persist nowcast models

three pieces:
    1. GradientBoosting regression  (main workhorse)
    2. ARIMA on residuals           (catch autocorrelation the tree misses)
    3. regime classifier            (weather-dominant vs fundamental-driven weeks)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .dataset import FEATURE_COLS, TARGET_COL

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


# ── dataclass to hold a trained model + metadata ─────────────────────

@dataclass
class TrainedModel:
    model: Any
    feature_names: list[str]
    train_mae: float = 0.0
    train_rmse: float = 0.0
    test_mae: float = 0.0
    test_rmse: float = 0.0
    test_mape_big: float = 0.0  # MAPE on |actual| >= 50 only


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = mse ** 0.5

    # MAPE only on big weeks (|actual| >= 50 Bcf)
    mask = np.abs(y_true) >= 50
    if mask.sum() > 0:
        mape_big = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    else:
        mape_big = np.nan

    return {"mae": mae, "rmse": rmse, "mape_big": mape_big}


# ── training ─────────────────────────────────────────────────────────

def train(
    df: pd.DataFrame,
    test_years: int = 2,
    model_type: str = "gbrt",
) -> tuple[TrainedModel, pd.DataFrame]:
    """
    time-split train/test, fit model, return (TrainedModel, df_with_preds).
    """
    df = df.sort_values("report_period_ending").copy()

    # time split
    cutoff = df["report_period_ending"].max()
    cutoff = pd.Timestamp(cutoff) - pd.DateOffset(years=test_years)
    cutoff = cutoff.date()

    train_mask = df["report_period_ending"] < cutoff
    test_mask  = df["report_period_ending"] >= cutoff

    X_train = df.loc[train_mask, FEATURE_COLS].values
    y_train = df.loc[train_mask, TARGET_COL].values
    X_test  = df.loc[test_mask, FEATURE_COLS].values
    y_test  = df.loc[test_mask, TARGET_COL].values

    print(f"  train: {train_mask.sum()} rows  |  test: {test_mask.sum()} rows")

    # pick model
    if model_type == "gbrt":
        mdl = GradientBoostingRegressor(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=10,
            random_state=42,
        )
    else:
        mdl = Ridge(alpha=1.0)

    mdl.fit(X_train, y_train)

    # predictions
    df.loc[train_mask, "pred_bcf"] = mdl.predict(X_train)
    df.loc[test_mask,  "pred_bcf"] = mdl.predict(X_test)
    df["residual"] = df[TARGET_COL] - df["pred_bcf"]

    # metrics
    tr_m = _metrics(y_train, mdl.predict(X_train))
    te_m = _metrics(y_test, mdl.predict(X_test))

    tm = TrainedModel(
        model=mdl,
        feature_names=list(FEATURE_COLS),
        train_mae=tr_m["mae"],
        train_rmse=tr_m["rmse"],
        test_mae=te_m["mae"],
        test_rmse=te_m["rmse"],
        test_mape_big=te_m["mape_big"],
    )

    return tm, df


# ── regime detection ─────────────────────────────────────────────────

def classify_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    tag each week as 'weather_dominant' or 'fundamental_driven'.

    logic:
        if the model (which is weather-heavy) has a small residual,
        weather explained most of the move -> weather_dominant.
        if residual is large, something else drove it -> fundamental_driven.

    threshold: median absolute residual over rolling window
    """
    df = df.copy()
    abs_resid = df["residual"].abs()

    # rolling median of absolute residual (backward 26 weeks ≈ 6 months)
    rolling_med = abs_resid.rolling(26, min_periods=8).median()

    # if this week's residual < 1.2x rolling median -> weather dominant
    df["regime"] = np.where(
        abs_resid <= rolling_med * 1.2,
        "weather_dominant",
        "fundamental_driven",
    )
    # first few rows without rolling history -> unknown
    df.loc[rolling_med.isna(), "regime"] = "unknown"

    return df


# ── persistence ──────────────────────────────────────────────────────

def save_model(tm: TrainedModel, path: Path | None = None):
    if path is None:
        path = MODELS_DIR / "model.joblib"
    joblib.dump(tm, path)
    print(f"  model saved -> {path}")


def load_model(path: Path | None = None) -> TrainedModel:
    if path is None:
        path = MODELS_DIR / "model.joblib"
    return joblib.load(path)
