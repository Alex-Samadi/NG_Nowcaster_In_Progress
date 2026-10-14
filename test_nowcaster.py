"""quick checks: python test_nowcaster.py  (or pytest)"""
from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.dataset import _add_features, FEATURE_COLS
from src.live import divergence_signal
from src.weather import build_weekly_dd


def _toy(n=12):
    fridays = [date(2024, 1, 5) + timedelta(weeks=i) for i in range(n)]
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "report_period_ending": fridays,
        "storage_change_bcf": rng.normal(-100, 50, n),
        "week_hdd": rng.uniform(100, 250, n),
        "week_cdd": rng.uniform(0, 5, n),
    })


def test_no_target_leakage():
    """changing this week's storage change must not change this week's features"""
    a = _toy()
    b = a.copy()
    b.loc[len(b) - 1, "storage_change_bcf"] += 999
    fa, fb = _add_features(a).iloc[-1], _add_features(b).iloc[-1]
    pd.testing.assert_series_equal(fa[FEATURE_COLS], fb[FEATURE_COLS])


def test_weekly_dd_sums_sat_to_fri():
    days = pd.date_range("2024-01-05", "2024-01-13")  # Fri .. next Sat
    hdd = pd.DataFrame({"date": days, "hdd": range(len(days))})
    cdd = pd.DataFrame({"date": days, "cdd": 0.0})
    wk = build_weekly_dd(hdd, cdd, [date(2024, 1, 12)])
    assert wk["week_hdd"].iloc[0] == sum(range(1, 8))  # Sat 6th .. Fri 12th only


def test_divergence_signal():
    assert divergence_signal(3)[0] == "neutral"
    assert divergence_signal(-10)[0] == "moderate"
    assert divergence_signal(20)[0] == "high"
    assert divergence_signal(-30)[0] == "extreme"
    assert divergence_signal(-30)[1].startswith("bullish")
    assert divergence_signal(30)[1].startswith("bearish")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
