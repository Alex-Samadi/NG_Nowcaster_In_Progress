"""
model.py - train / evaluate / persist nowcast models

pieces:
    1. ensemble: average of GradientBoosting + Ridge   (the nowcast)
    2. naive benchmarks + walk-forward test   (is it actually better?)
    3. regime tagging                         (weather-dominant vs fundamental-driven weeks)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor
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
    test_start: Any = None
    baseline_mae: dict = field(default_factory=dict)  # test MAE of naive benchmarks


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

def _make_model(model_type: str):
    gbrt = GradientBoostingRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42,
    )
    ridge = Ridge(alpha=1.0)
    if model_type == "gbrt":
        return gbrt
    if model_type == "ridge":
        return ridge
    # ensemble: plain 50/50 average of both (weights deliberately not tuned on test data)
    return VotingRegressor([("gbrt", gbrt), ("ridge", ridge)])


def train(
    df: pd.DataFrame,
    test_years: int = 2,
    model_type: str = "ensemble",
) -> tuple[TrainedModel, pd.DataFrame]:
    """
    time-split train/test, fit model, return (TrainedModel, df_with_preds).
    """
    df = add_five_year_avg(df.sort_values("report_period_ending"))

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

    mdl = _make_model(model_type)
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
        test_start=cutoff,
        baseline_mae={
            name: float((df.loc[test_mask, TARGET_COL] - df.loc[test_mask, col]).abs().mean())
            for name, col in [("last_week", "storage_chg_lag1"), ("five_year_avg", "five_yr_avg")]
        },
    )

    return tm, df


def walk_forward(df: pd.DataFrame, years: range) -> pd.DataFrame:
    """train on everything before each year, test on that year. one row per year,
    MAE (Bcf) for each model + the 5-year-average benchmark."""
    df = add_five_year_avg(df)
    d = pd.to_datetime(df["report_period_ending"])
    rows = []
    for yr in years:
        tr, te = d.dt.year < yr, d.dt.year == yr
        if not te.any():
            continue
        y = df.loc[te, TARGET_COL]
        row = {"year": yr, "weeks": int(te.sum())}
        for name in ["gbrt", "ridge", "ensemble"]:
            m = _make_model(name).fit(df.loc[tr, FEATURE_COLS].values, df.loc[tr, TARGET_COL].values)
            row[name] = float(np.abs(m.predict(df.loc[te, FEATURE_COLS].values) - y).mean())
        row["five_year_avg"] = float((df.loc[te, "five_yr_avg"] - y).abs().mean())
        rows.append(row)
    return pd.DataFrame(rows).round(1)


def add_five_year_avg(df: pd.DataFrame) -> pd.DataFrame:
    """same-week average of the prior 5 years: the benchmark EIA itself reports against"""
    df = df.copy()
    woy = pd.to_datetime(df["report_period_ending"]).dt.isocalendar().week
    df["five_yr_avg"] = (
        df.groupby(woy)[TARGET_COL]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
    )
    return df


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
