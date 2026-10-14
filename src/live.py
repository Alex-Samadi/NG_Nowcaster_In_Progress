"""
live.py - produce a nowcast for an upcoming/recent storage week

this is the "trade-ahead" piece: run it Mon-Wed before Thursday's EIA release
and get a predicted storage change + confidence band.
"""
from __future__ import annotations
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SERIES_WORKING_GAS
from .eia import get_storage
from .weather import (
    fetch_hdd_actuals,
    fetch_cdd_actuals,
    fetch_dd_forecast,
    build_weekly_dd,
)
from .dataset import FEATURE_COLS, _add_features
from .model import load_model, TrainedModel


def divergence_signal(gap_bcf: float) -> tuple[str, str]:
    """(strength, direction) for a model-minus-consensus gap"""
    a = abs(gap_bcf)
    strength = "neutral" if a < 5 else "moderate" if a < 15 else "high" if a < 25 else "extreme"
    if strength == "neutral":
        return strength, "no edge: model agrees with consensus"
    # works in both seasons: below consensus = less gas in storage than expected
    direction = ("bullish: model sees less gas in storage than consensus" if gap_bcf < 0
                 else "bearish: model sees more gas in storage than consensus")
    return strength, direction


def _most_recent_friday(d: date) -> date:
    """return the most recent Friday on or before d"""
    wd = d.weekday()  # Mon=0 ... Sun=6
    delta = (wd - 4) % 7
    return d - timedelta(days=delta)


def nowcast(
    week_end: date | None = None,
    model_path: Path | None = None,
    consensus_bcf: float | None = None,
) -> dict:
    """
    produce a nowcast dict with:
        week_ending, pred_bcf, confidence_lo, confidence_hi,
        week_hdd, week_cdd, consensus_bcf, gap_bcf, regime_hint
    """
    if week_end is None:
        week_end = _most_recent_friday(date.today())

    tm: TrainedModel = load_model(model_path)

    # pull storage history (need lags)
    storage = get_storage(start="2020-01-01")

    # pull degree-days
    hdd_daily = fetch_hdd_actuals(years=list(range(2020, date.today().year + 1)))
    cdd_daily = fetch_cdd_actuals(years=list(range(2020, date.today().year + 1)))

    # include target week in the endings list
    all_endings = sorted(set(storage["report_period_ending"].tolist() + [week_end]))
    weekly_dd = build_weekly_dd(hdd_daily, cdd_daily, all_endings)

    # merge
    df = storage.merge(weekly_dd, on="report_period_ending", how="outer")
    df["week_hdd"] = df["week_hdd"].fillna(0)
    df["week_cdd"] = df["week_cdd"].fillna(0)
    df = df.sort_values("report_period_ending").reset_index(drop=True)

    # forward-fill storage for the target week if it hasn't been reported yet
    df["storage_change_bcf"] = df["storage_change_bcf"].ffill()
    df["working_gas_bcf"]    = df["working_gas_bcf"].ffill()

    df = _add_features(df)

    # grab the row for our target week
    target_row = df[df["report_period_ending"] == week_end]
    if target_row.empty:
        raise ValueError(f"no data row for week ending {week_end}")

    row = target_row.iloc[-1]

    # check we have all features
    missing = [c for c in tm.feature_names if c not in row.index or pd.isna(row[c])]
    if missing:
        print(f"  warning: missing features {missing}, filling with 0")

    X = np.array([[row.get(c, 0.0) for c in tm.feature_names]])
    pred = float(tm.model.predict(X)[0])

    # confidence band using test RMSE
    lo = pred - tm.test_rmse
    hi = pred + tm.test_rmse

    # consensus comparison
    gap = None
    if consensus_bcf is not None:
        gap = pred - consensus_bcf

    # regime hint: based on HDD deviation
    hdd_dev = float(row.get("hdd_deviation", 0))
    if abs(hdd_dev) > 30:
        regime_hint = "weather_dominant (large HDD deviation)"
    else:
        regime_hint = "mixed/fundamental (HDD near seasonal)"

    result = {
        "week_ending": week_end,
        "pred_bcf": round(pred, 1),
        "confidence_lo": round(lo, 1),
        "confidence_hi": round(hi, 1),
        "week_hdd": round(float(row.get("week_hdd", 0)), 1),
        "week_cdd": round(float(row.get("week_cdd", 0)), 1),
        "hdd_deviation": round(hdd_dev, 1),
        "regime_hint": regime_hint,
        "consensus_bcf": consensus_bcf,
        "gap_bcf": round(gap, 1) if gap is not None else None,
        "model_test_mae": round(tm.test_mae, 1),
        "model_test_rmse": round(tm.test_rmse, 1),
    }

    return result


def log_revision(result: dict, path: Path | None = None):
    """append nowcast result to revision CSV"""
    if path is None:
        path = Path("outputs/revisions.csv")
    path.parent.mkdir(exist_ok=True)

    row = {
        "run_date": date.today().isoformat(),
        "week_ending": result["week_ending"],
        "pred_bcf": result["pred_bcf"],
        "week_hdd": result["week_hdd"],
        "consensus_bcf": result.get("consensus_bcf"),
        "gap_bcf": result.get("gap_bcf"),
    }
    df_row = pd.DataFrame([row])

    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, df_row], ignore_index=True)
    else:
        combined = df_row

    combined.to_csv(path, index=False)
