from __future__ import annotations
import os

# ── EIA ──────────────────────────────────────────────────────────────
EIA_API_KEY = os.getenv("EIA_API_KEY", "")

# v2 series-id endpoint (Lower-48 weekly working gas in storage)
EIA_V2_BASE = "https://api.eia.gov/v2/seriesid/{series_id}"
SERIES_WORKING_GAS = "NG.NW2_EPG0_SWO_R48_BCF.W"

# ── NOAA CPC degree-day products ────────────────────────────────────
# daily actuals  (UtilityGas-weighted HDD, Population-weighted CDD)
CPC_DD_ACTUAL_URL = (
    "https://ftp.cpc.ncep.noaa.gov/htdocs/degree_days"
    "/weighted/daily_data/{year}/{product}.txt"
)
# 7-day daily forecast
CPC_DD_FCST_URL = (
    "https://ftp.cpc.ncep.noaa.gov/htdocs/degree_days"
    "/weighted/daily_data/forecast_daily.csv"
)

CPC_HDD_PRODUCT = "StatesCONUS.Heating"       # gas-weighted HDD
CPC_CDD_PRODUCT = "StatesCONUS.Cooling"       # pop-weighted CDD

# ── storage week convention ──────────────────────────────────────────
# EIA storage week runs Saturday -> Friday
# report releases following Thursday (with holiday exceptions)
STORAGE_WEEK_END_DOW = 4   # Friday = 4 in weekday()
