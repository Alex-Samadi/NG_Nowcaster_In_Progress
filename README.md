# NG Storage Nowcaster

weather → storage prediction → consensus divergence → edge

## what this does

predicts the weekly EIA Natural Gas Storage change (Bcf) using:
- NOAA CPC gas-weighted HDD / pop-weighted CDD
- lagged storage changes + rolling stats
- seasonal encoding (sin/cos week-of-year)
- HDD deviation from seasonal norm

then compares your prediction to the street consensus (Celsius, etc.)
to find weeks where the market may be leaning the wrong way on weather.

## the actual edge

this model won't beat Celsius on the point estimate. the edge is:

1. **regime detection** – the model tags weeks as "weather_dominant" or
   "fundamental_driven" based on how well weather alone explains the move.
   when you're in a weather-dominant week AND your model disagrees with
   consensus, that's your highest-conviction signal.

2. **revision tracking** – run the nowcast Mon/Tue/Wed before Thursday's
   report. watch how the prediction revises as weather updates. the
   *direction* of revision is often more tradable than the absolute number.

3. **consensus gap** – enter the Celsius/NGI estimate. the model computes
   the gap and flags divergence levels (neutral / moderate / high / extreme).

## setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # Git Bash on Windows
pip install -r requirements.txt
export EIA_API_KEY="your_key_here"
```

get your free EIA API key: https://www.eia.gov/opendata/register.php

## usage

### train the model
```bash
python run.py backtest --start 2014-01-01 --test-years 2
```

### run a live nowcast
```bash
# most recent completed storage week
python run.py live

# specific week + consensus
python run.py live --week 2026-02-06 --consensus -271
```

### launch the dashboard
```bash
streamlit run app.py
```

## project structure

```
├── run.py              # CLI entry point
├── app.py              # Streamlit dashboard
├── requirements.txt
├── src/
│   ├── config.py       # API keys, URLs, constants
│   ├── eia.py          # EIA storage data fetcher
│   ├── weather.py      # NOAA CPC degree-day fetcher
│   ├── dataset.py      # merge + feature engineering
│   ├── model.py        # GBRT training + regime detection
│   └── live.py         # live nowcast logic
├── models/             # saved model files
├── outputs/            # dataset, metrics, revisions
└── cache/              # downloaded weather files
```

## what the model uses

| feature | description |
|---------|-------------|
| week_hdd | gas-weighted HDD for the storage week (Sat→Fri) |
| week_cdd | pop-weighted CDD for the storage week |
| woy_sin/cos | continuous week-of-year encoding |
| storage_chg_lag1-3 | prior 3 weeks' storage changes |
| hdd_lag1-3 | prior 3 weeks' HDD |
| cdd_lag1 | prior week CDD |
| hdd_roll4_mean/std | 4-week rolling HDD stats |
| chg_roll4_mean | 4-week rolling storage change |
| hdd_deviation | HDD vs expanding seasonal mean |
| hdd_x_sin/cos | HDD × season interaction |

## future improvements

- [ ] add LNG feedgas / dry gas production (daily EIA or genscape)
- [ ] regional HDD weighting (midwest/east heavier)
- [ ] freeze-off flags for extreme cold events
- [ ] ARIMA residual correction layer
- [ ] probability distribution output (not just point estimate)
- [ ] automated consensus scraping (if API available)
