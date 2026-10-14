"""
dataset.py - build the weekly training dataset
    EIA storage changes + CPC degree-days + engineered features
"""
from __future__ import annotations
from datetime import date, timedelta
import numpy as np
import pandas as pd

from .eia import get_storage
from .weather import fetch_hdd_actuals, fetch_cdd_actuals, build_weekly_dd


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    engineer features that actually matter for storage prediction.
    everything is constructed with ONLY backward-looking data (no leakage).
    """
    df = df.sort_values("report_period_ending").copy()

    # ── week-of-year seasonality (continuous encoding) ───────────────
    woy = df["report_period_ending"].apply(lambda d: d.isocalendar()[1])
    df["woy_sin"] = np.sin(2 * np.pi * woy / 52)
    df["woy_cos"] = np.cos(2 * np.pi * woy / 52)

    # ── month (for regime splitting later) ───────────────────────────
    df["month"] = df["report_period_ending"].apply(lambda d: d.month)

    # ── lags ─────────────────────────────────────────────────────────
    for lag in [1, 2, 3]:
        df[f"storage_chg_lag{lag}"] = df["storage_change_bcf"].shift(lag)
        df[f"hdd_lag{lag}"] = df["week_hdd"].shift(lag)
    df["cdd_lag1"] = df["week_cdd"].shift(1)

    # ── rolling stats (backward only) ────────────────────────────────
    df["hdd_roll4_mean"] = df["week_hdd"].rolling(4, min_periods=1).mean()
    df["hdd_roll4_std"]  = df["week_hdd"].rolling(4, min_periods=1).std().fillna(0)
    # shift(1): this week's storage change is the target, so it can't be in its own feature
    df["chg_roll4_mean"] = df["storage_change_bcf"].shift(1).rolling(4, min_periods=1).mean()

    # ── HDD deviation from seasonal norm ─────────────────────────────
    # expanding mean of HDD for same week-of-year (uses only past data)
    df["_woy"] = woy
    seasonal_hdd = df.groupby("_woy")["week_hdd"].expanding().mean()
    seasonal_hdd = seasonal_hdd.droplevel(0)
    df["hdd_seasonal_mean"] = seasonal_hdd
    df["hdd_deviation"] = df["week_hdd"] - df["hdd_seasonal_mean"]
    df = df.drop(columns=["_woy"])

    # ── storage level context ────────────────────────────────────────
    if "working_gas_bcf" in df.columns:
        df["storage_level_lag1"] = df["working_gas_bcf"].shift(1)

    # ── interaction: HDD * season ────────────────────────────────────
    df["hdd_x_sin"] = df["week_hdd"] * df["woy_sin"]
    df["hdd_x_cos"] = df["week_hdd"] * df["woy_cos"]

    return df


# features used by the model (order matters for consistency)
FEATURE_COLS = [
    "week_hdd", "week_cdd",
    "woy_sin", "woy_cos",
    "storage_chg_lag1", "storage_chg_lag2", "storage_chg_lag3",
    "hdd_lag1", "hdd_lag2", "hdd_lag3", "cdd_lag1",
    "hdd_roll4_mean", "hdd_roll4_std",
    "chg_roll4_mean",
    "hdd_deviation",
    "hdd_x_sin", "hdd_x_cos",
]

TARGET_COL = "storage_change_bcf"


def build_dataset(start: str = "2014-01-01") -> pd.DataFrame:
    """
    end-to-end: pull EIA + CPC, merge, feature-engineer.
    returns a clean DataFrame ready for train/test split.
    """
    print("[1/3] pulling EIA storage data...")
    storage = get_storage(start=start)

    week_endings = sorted(storage["report_period_ending"].unique())

    print("[2/3] pulling NOAA CPC degree-day actuals...")
    years = list(range(int(start[:4]), date.today().year + 1))
    hdd = fetch_hdd_actuals(years)
    cdd = fetch_cdd_actuals(years)
    weekly_dd = build_weekly_dd(hdd, cdd, week_endings)

    print("[3/3] merging + engineering features...")
    df = storage.merge(weekly_dd, on="report_period_ending", how="left")
    df["week_hdd"] = df["week_hdd"].fillna(0)
    df["week_cdd"] = df["week_cdd"].fillna(0)

    df = _add_features(df)

    # drop rows where lags aren't available yet
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    return df
