"""
weather.py - pull NOAA CPC degree-day data (actuals + forecast)
"""
from __future__ import annotations
import io, os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# CPC URLs
CPC_BASE = "https://ftp.cpc.ncep.noaa.gov/htdocs/degree_days/weighted/daily_data"


def _download_text(url, cache_name=None, refresh=False):
    """past years never change, so read them from cache. refresh=True (current
    year) re-downloads, falling back to the cache when offline."""
    path = CACHE_DIR / cache_name if cache_name else None
    if path and path.exists() and not refresh:
        return path.read_text(encoding="utf-8")

    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
    except requests.RequestException:
        if path and path.exists():
            print(f"  warning: offline, using cached {cache_name}")
            return path.read_text(encoding="utf-8")
        raise
    text = r.text

    if path:
        path.write_text(text, encoding="utf-8")
    return text


def _parse_cpc_txt(text):
    """
    parse CPC .txt file format:
        line 0: Product: Daily Heating Degree Days
        line 1: Regions: ...
        line 2: Weights: ...
        line 3: Region|20260101|20260102|...  (header)
        line 4+: AL|21|20|13|...              (state data)

    returns DataFrame with columns [date, dd] where dd is the
    average across all states (already population-weighted by CPC).
    """
    lines = text.strip().split("\n")

    # find the header line (starts with "Region|")
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Region|") or line.strip().startswith("Region |"):
            header_idx = i
            break

    if header_idx is None:
        return pd.DataFrame(columns=["date", "dd"])

    # parse header to get date columns
    header_parts = [p.strip() for p in lines[header_idx].split("|")]
    date_cols = []
    for p in header_parts[1:]:  # skip "Region"
        try:
            dt = pd.to_datetime(p, format="%Y%m%d")
            date_cols.append(dt.date())
        except Exception:
            continue

    if not date_cols:
        return pd.DataFrame(columns=["date", "dd"])

    # parse state rows and compute average across states
    all_state_values = []
    for line in lines[header_idx + 1:]:
        parts = line.strip().split("|")
        if len(parts) < 2:
            continue
        # first part is state abbreviation, rest are values
        values = []
        for v in parts[1:len(date_cols) + 1]:
            try:
                values.append(float(v.strip()))
            except (ValueError, TypeError):
                values.append(np.nan)
        if len(values) == len(date_cols):
            all_state_values.append(values)

    if not all_state_values:
        return pd.DataFrame(columns=["date", "dd"])

    # average across all states for each day
    arr = np.array(all_state_values)
    daily_avg = np.nanmean(arr, axis=0)

    records = []
    for dt, val in zip(date_cols, daily_avg):
        if not np.isnan(val):
            records.append({"date": dt, "dd": round(float(val), 2)})

    return pd.DataFrame(records)


def fetch_hdd_actuals(years=None):
    if years is None:
        cur = date.today().year
        years = list(range(cur - 12, cur + 1))

    frames = []
    for yr in years:
        url = f"{CPC_BASE}/{yr}/StatesCONUS.Heating.txt"
        cache = f"hdd_actual_{yr}.txt"
        try:
            text = _download_text(url, cache_name=cache, refresh=yr == date.today().year)
            df = _parse_cpc_txt(text)
            if not df.empty:
                frames.append(df.rename(columns={"dd": "hdd"}))
        except Exception as e:
            print(f"  warning: could not fetch HDD for {yr}: {e}")
            continue

    if not frames:
        return pd.DataFrame(columns=["date", "hdd"])
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)


def fetch_cdd_actuals(years=None):
    if years is None:
        cur = date.today().year
        years = list(range(cur - 12, cur + 1))

    frames = []
    for yr in years:
        url = f"{CPC_BASE}/{yr}/StatesCONUS.Cooling.txt"
        cache = f"cdd_actual_{yr}.txt"
        try:
            text = _download_text(url, cache_name=cache, refresh=yr == date.today().year)
            df = _parse_cpc_txt(text)
            if not df.empty:
                frames.append(df.rename(columns={"dd": "cdd"}))
        except Exception as e:
            print(f"  warning: could not fetch CDD for {yr}: {e}")
            continue

    if not frames:
        return pd.DataFrame(columns=["date", "cdd"])
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)


def fetch_dd_forecast():
    """CPC 7-day daily forecast"""
    try:
        url = f"{CPC_BASE}/forecast_daily.csv"
        text = _download_text(url)
        df = _parse_cpc_txt(text)
        return df
    except Exception:
        return pd.DataFrame(columns=["date", "dd"])


def build_weekly_dd(hdd_daily, cdd_daily, week_endings):
    """
    for each storage week (Sat -> Fri), sum HDD and CDD.
    returns DataFrame[report_period_ending, week_hdd, week_cdd]
    """
    hdd_daily = hdd_daily.copy()
    cdd_daily = cdd_daily.copy()
    hdd_daily["date"] = pd.to_datetime(hdd_daily["date"]).dt.date
    cdd_daily["date"] = pd.to_datetime(cdd_daily["date"]).dt.date

    hdd_lookup = dict(zip(hdd_daily["date"], hdd_daily["hdd"]))
    cdd_lookup = dict(zip(cdd_daily["date"], cdd_daily["cdd"]))

    rows = []
    for fri in week_endings:
        sat = fri - timedelta(days=6)
        days = [sat + timedelta(days=i) for i in range(7)]
        wk_hdd = sum(hdd_lookup.get(d, 0.0) for d in days)
        wk_cdd = sum(cdd_lookup.get(d, 0.0) for d in days)
        rows.append({
            "report_period_ending": fri,
            "week_hdd": wk_hdd,
            "week_cdd": wk_cdd,
        })

    return pd.DataFrame(rows)