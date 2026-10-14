"""
eia.py - pull weekly storage from EIA v2 API
"""
from __future__ import annotations
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from pathlib import Path

from .config import EIA_API_KEY, EIA_V2_BASE, SERIES_WORKING_GAS

CACHE_FILE = Path("cache/eia_storage.csv")


class EIAKeyMissing(RuntimeError):
    pass


def _make_session() -> requests.Session:
    """session w/ automatic retries on transient failures"""
    s = requests.Session()
    retries = Retry(
        total=2,  # fail fast; get_storage falls back to the cache
        backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def fetch_series(series_id: str, start: str = "2014-01-01") -> pd.DataFrame:
    """pull a single EIA v2 series -> DataFrame[date, value]"""
    if not EIA_API_KEY:
        raise EIAKeyMissing(
            "Set EIA_API_KEY env var before running.\n"
            "  export EIA_API_KEY='your_key_here'"
        )

    url = EIA_V2_BASE.format(series_id=series_id)
    params = {"api_key": EIA_API_KEY, "start": start}

    sess = _make_session()
    r = sess.get(url, params=params, timeout=(5, 30))
    r.raise_for_status()

    data = r.json().get("response", {}).get("data", [])
    if not data:
        return pd.DataFrame(columns=["date", "value"])

    df = pd.DataFrame(data)
    if "period" in df.columns:
        df = df.rename(columns={"period": "date"})

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "value"])
    return df[["date", "value"]].sort_values("date").reset_index(drop=True)


def get_storage(start: str = "2014-01-01") -> pd.DataFrame:
    """
    returns DataFrame with columns:
        report_period_ending  (date)  - the Friday that ends the storage week
        working_gas_bcf       (float)
        storage_change_bcf    (float) - week-over-week diff
    """
    # cache the raw series so the app still runs offline / without a key
    try:
        raw = fetch_series(SERIES_WORKING_GAS, start="2010-01-01")
        raw.to_csv(CACHE_FILE, index=False)
    except (EIAKeyMissing, requests.RequestException) as e:
        if not CACHE_FILE.exists():
            raise
        print(f"  warning: EIA unavailable ({type(e).__name__}), using {CACHE_FILE}")
        raw = pd.read_csv(CACHE_FILE, parse_dates=["date"])
    raw = raw[raw["date"] >= pd.Timestamp(start)]
    raw = raw.rename(columns={"value": "working_gas_bcf"})
    raw["report_period_ending"] = raw["date"].dt.date
    raw = raw.sort_values("report_period_ending").reset_index(drop=True)
    raw["storage_change_bcf"] = raw["working_gas_bcf"].diff()
    return raw.dropna(subset=["storage_change_bcf"]).reset_index(drop=True)
