# NG Storage Nowcaster

Predicts the weekly EIA natural gas storage change (Bcf) from weather data, days before
Thursday's report, and flags weeks where the model disagrees with market consensus.

## results

Out-of-sample test, Oct 2024 – Oct 2026 (the last 2 years, never seen in training):

| model | MAE (Bcf) |
|-------|-----------|
| **gradient boosting (this model)** | **13.9** |
| 5-year same-week average (EIA's own benchmark) | 31.1 |
| "same as last week" | 39.4 |

Walk-forward check (train on all prior years, predict the next year, for every year 2015–2026):
the model beats the 5-year average in **12 / 12 years**, typically by about half.

What I learned along the way:
- **A plain ridge regression matches the gradient-boosted trees** (13.5 vs 14.6 Bcf average walk-forward MAE; ridge wins 8 of 12 years).
  Train MAE 5.1 vs test 13.9 shows the trees overfit; most of the signal is close to linear in degree days.
- **COVID wasn't an outlier for this model.** Mar–Jun 2020 error was 15.0 Bcf vs 14.4 for the rest of 2020
  (≈ +7 Bcf bias: lockdown demand loss meant bigger injections). Excluding it from training changed MAE by < 0.5 Bcf.
- **Found and fixed a target leak**: a rolling-mean feature included the current week's storage change.
  `test_nowcaster.py` guards it.

## how it works

```
NOAA CPC daily HDD/CDD ──► sum over Sat→Fri storage week ──┐
EIA weekly working gas  ──► week-over-week change, lags ────┼─► features ─► model ─► Δ storage (Bcf)
                                                            │                           │
                                         consensus (manual) ┴──────────────► gap + regime signal
```

Features: week HDD/CDD, HDD deviation from seasonal norm, 1–3 week lags of storage change and HDD,
4-week rolling stats, sin/cos week-of-year, HDD × season interactions. All backward-looking.

## the edge isn't the point estimate

The model won't beat professional estimates (e.g. Celsius) on raw accuracy. The use is:

1. **regime**: when temperatures are far from normal, weather explains most of the move, so a model-vs-consensus
   gap is meaningful. When weather is near normal, LNG exports / production / power burn dominate. Be skeptical.
2. **revisions**: run Mon→Wed as weather data updates; the *direction* of revision is often more informative than the level.
3. **consensus gap**: neutral (< 5 Bcf) / moderate / high / extreme (> 25 Bcf).

## run it

```bash
pip install -r requirements.txt
python test_nowcaster.py                               # sanity checks
streamlit run app.py                                   # dashboard (works offline from cache/)
python run.py backtest --start 2010-01-01              # retrain + walk-forward
python run.py live --consensus 80    # CLI nowcast
```

Live data needs a free EIA key (https://www.eia.gov/opendata/register.php): `export EIA_API_KEY=...`.
Without one, the app falls back to the cached series in `cache/`.

## structure

```
run.py              CLI: backtest / live
app.py              Streamlit dashboard
test_nowcaster.py   leakage, week-alignment and signal checks
src/
  eia.py            EIA v2 storage fetch (+ offline cache)
  weather.py        NOAA CPC degree days (+ cache)
  dataset.py        merge + feature engineering
  model.py          training, benchmarks, walk-forward, regime tagging
  live.py           single-week nowcast + revision log
```

## known limitations / next steps

- HDD is a simple average across 48 state series; true **gas-consumption weighting** (heavier on Midwest/Northeast) should help.
- Supply side is missing: LNG feedgas, dry gas production, freeze-offs.
- The confidence band is ±1 test RMSE; quantile regression would give a real distribution.
- Regime labels on history use the realized residual (hindsight); the live hint uses HDD deviation instead.
- Consensus is entered by hand.
