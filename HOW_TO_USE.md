# How to Use the NG Storage Nowcaster (and Actually Get Value From It)

## What This Tool Does in Plain English

Every Thursday at 10:30am ET, the EIA releases the weekly natural gas storage report. Markets react. Before that report drops, this tool gives you a predicted storage change (in Bcf) based on weather data, and tells you how confident to be in that prediction.

The tool does NOT try to beat Celsius Energy or Wall Street consensus on the exact number. Instead, it tells you:

- What weather alone says storage should do this week
- Whether that's meaningfully different from what the street expects
- Whether this is a week where weather is the dominant driver (so your lean matters) or a week where fundamentals are driving (so weather-based signals are noise)

That's the edge. Not a better number — a better understanding of when weather matters.

---

## Setup (One Time)

### 1. Get Your EIA API Key
Go to https://www.eia.gov/opendata/register.php — it's free. They email you the key.

### 2. Install
Open Git Bash, navigate to the project folder, and run:

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

### 3. Set Your API Key
Every time you open a new Git Bash window:

```bash
export EIA_API_KEY="your_key_here"
```

To make it permanent so you don't have to do this every time:

```bash
echo 'export EIA_API_KEY="your_key_here"' >> ~/.bashrc
source ~/.bashrc
```

### 4. Train the Model (Run Once, Then Periodically)
```bash
python run.py backtest --start 2014-01-01 --test-years 2
```

This pulls all historical data, trains the model, and saves everything. Takes 1-2 minutes depending on your connection. Re-run this every month or so to incorporate new data.

---

## Weekly Workflow (Where the Value Is)

This is the routine you follow each week to get actionable information.

### Monday: First Read

Run your first nowcast for the upcoming report:

```bash
python run.py live
```

This predicts the storage change for the most recently completed storage week (the one ending last Friday). Write down the number.

### Tuesday-Wednesday: Track Revisions

Run the same command again each day:

```bash
python run.py live
```

Each run gets logged to `outputs/revisions.csv`. As CPC weather data updates, your prediction may shift. The direction of that shift is often more informative than the absolute number.

Example revision pattern:
```
Monday:    -210 Bcf
Tuesday:   -225 Bcf
Wednesday: -236 Bcf
```

This tells you: weather data is revising toward a LARGER withdrawal. If consensus hasn't moved, the market might be underpricing the cold.

### Wednesday Evening: Compare to Consensus

Get the consensus estimate (Celsius Energy, NGI, Bloomberg, wherever you look). Then run:

```bash
python run.py live --consensus -271
```

The output will show:
- Your model's prediction
- The gap between your model and consensus
- A signal strength (neutral / moderate / high / extreme)
- Whether your model is bullish or bearish relative to consensus
- The regime hint (weather-dominant or fundamental-driven)

### Thursday Morning: Pre-Report Assessment

Before the 10:30am ET release, you now have:
- Your weather-based estimate
- How it revised throughout the week
- How it compares to consensus
- Whether this is a weather-driven week or not

This informs your positioning decision.

---

## How to Read the Output

### The Prediction
```
predicted Δ storage:  -236.7 Bcf
```
Negative = withdrawal (bullish gas). Positive = injection (bearish gas).

### The Confidence Band
```
confidence band:      [-256.7, -216.7]
```
Based on the model's test RMSE (~20 Bcf). The actual EIA number should land within this range most of the time. If consensus is OUTSIDE this band, that's a strong divergence signal.

### HDD vs Seasonal
```
HDD vs seasonal:      +45.2
```
How much colder (or warmer) this week was compared to the historical average for this time of year. Large positive = unusually cold. Large negative = unusually warm.

### Regime Hint
```
regime hint:          weather_dominant (large HDD deviation)
```
Two modes:
- **weather_dominant**: HDD deviation is large, meaning weather is the primary driver this week. Your weather-based model is most useful here.
- **mixed/fundamental**: HDD is near seasonal norms, meaning LNG exports, production changes, or power burn are likely driving the storage number. Your model is less reliable here — be cautious about divergence signals.

### Consensus Gap
```
gap (model-consensus): +34.3 Bcf
signal:                HIGH divergence → BEARISH (smaller draw)
```
Interpretation:
- Gap > 0: your model expects a SMALLER withdrawal than consensus → bearish signal
- Gap < 0: your model expects a LARGER withdrawal than consensus → bullish signal

Signal strength:
- |gap| < 5 Bcf: neutral, no meaningful divergence
- 5-15 Bcf: moderate, worth noting but not high conviction
- 15-25 Bcf: high, meaningful disagreement
- 25+ Bcf: extreme, very high conviction if regime is weather-dominant

---

## Using the Dashboard

Launch it:
```bash
streamlit run app.py
```

### Live Nowcast Page
- Pick the EIA report date (Thursday)
- It auto-calculates the covered storage week
- Enter consensus, click "run nowcast"
- See prediction + gap + signal all in one view

### Backtest History Page
- Model vs actual over the full history
- Residual distribution (are errors random or biased?)
- Regime breakdown: see the MAE for weather-dominant vs fundamental-driven weeks separately

### Model vs Consensus Page
- Upload a CSV of historical consensus estimates
- See side-by-side: EIA actual vs your model vs consensus
- Track how often your model is closer to actual than consensus

### Revision Tracker Page
- Pick a storage week
- See how your prediction changed day-to-day
- Visualize revision momentum

---

## When This Tool Is Most Valuable

### High-Value Weeks (Lean Into Your Signal)
- HDD deviation is large (very cold or very warm vs seasonal)
- Your model disagrees with consensus by 15+ Bcf
- Regime hint says "weather_dominant"
- Your revisions are trending in one direction consistently

### Low-Value Weeks (Be Cautious)
- HDD is near seasonal norms
- Regime hint says "mixed/fundamental"
- Your model and consensus are within 5-10 Bcf
- This is likely a week where LNG/production/power burn matters more than weather

### The Key Insight
You're not trying to be right every week. You're trying to identify the 10-15 weeks per year where weather is the dominant driver AND the market is mispricing it. Those are your highest-conviction setups.

---

## File Structure

```
outputs/
  weekly_dataset.csv      # full historical dataset with predictions + regimes
  metrics.json            # model accuracy stats
  revisions.csv           # daily prediction log (builds over time)
  consensus_history.csv   # your entered consensus values

models/
  model.joblib            # trained model file

cache/
  hdd_actual_YYYY.csv     # cached weather data (speeds up re-runs)
  cdd_actual_YYYY.csv
```

---

## Troubleshooting

### "EIA_API_KEY not set"
Run `export EIA_API_KEY="your_key"` in the same terminal window. Or add it to ~/.bashrc.

### Timeout errors from EIA
The code has automatic retries built in. If it still fails, just run the command again — EIA's API can be slow sometimes.

### "No model found"
Run `python run.py backtest` first to train the model before running `python run.py live`.

### Numbers look way off
Check that your week alignment is correct. The EIA report released on Thursday covers the storage week ending the PRIOR Friday. Make sure you're comparing apples to apples.
