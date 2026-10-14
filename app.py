"""
app.py - Streamlit dashboard for NG Storage Nowcaster

run:  streamlit run app.py
"""
from __future__ import annotations
import json
from datetime import timedelta
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="NG Storage Nowcaster", page_icon="🔥", layout="wide")

OUTPUTS   = Path("outputs")
DATASET   = OUTPUTS / "weekly_dataset.csv"
METRICS   = OUTPUTS / "metrics.json"
WALK_FWD  = OUTPUTS / "walk_forward.csv"
REVISIONS = OUTPUTS / "revisions.csv"
MODEL_F   = Path("models/model.joblib")

ACTUAL, MODEL, BENCH = "#1f2937", "#2563eb", "#d97706"
SERIES_COLORS = alt.Scale(domain=["EIA actual", "model", "5-yr average"], range=[ACTUAL, MODEL, BENCH])
FEATURE_LABELS = {
    "week_hdd": "heating degree days (this week)",
    "week_cdd": "cooling degree days (this week)",
    "woy_sin": "time of year (sin)",
    "woy_cos": "time of year (cos)",
    "storage_chg_lag1": "storage change 1 week ago",
    "storage_chg_lag2": "storage change 2 weeks ago",
    "storage_chg_lag3": "storage change 3 weeks ago",
    "hdd_lag1": "heating degree days 1 week ago",
    "hdd_lag2": "heating degree days 2 weeks ago",
    "hdd_lag3": "heating degree days 3 weeks ago",
    "cdd_lag1": "cooling degree days 1 week ago",
    "hdd_roll4_mean": "heating degree days, 4-week avg",
    "hdd_roll4_std": "heating degree days, 4-week variability",
    "chg_roll4_mean": "storage change, prior 4-week avg",
    "hdd_deviation": "heating degree days vs normal",
    "hdd_x_sin": "heating × time of year (sin)",
    "hdd_x_cos": "heating × time of year (cos)",
}
WF_LABELS = {"ensemble": "ensemble (this model)", "gbrt": "gradient boosting", "ridge": "ridge (linear)",
             "five_year_avg": "5-yr average"}
WF_COLORS = alt.Scale(domain=list(WF_LABELS.values()), range=[MODEL, "#93c5fd", "#7c3aed", BENCH])


# ── data ─────────────────────────────────────────────────────────────

if not (DATASET.exists() and METRICS.exists() and MODEL_F.exists()):
    st.error("No trained model yet. Run `python run.py backtest --start 2010-01-01` first.")
    st.stop()

df = pd.read_csv(DATASET, parse_dates=["report_period_ending"])
metrics = json.loads(METRICS.read_text())
test_start = pd.Timestamp(metrics["test_start"])
base = metrics["baseline_test_mae"]
wf = pd.read_csv(WALK_FWD) if WALK_FWD.exists() else None


def long_form(frame: pd.DataFrame, cols: dict) -> pd.DataFrame:
    out = frame[["report_period_ending", *cols]].rename(columns=cols)
    return out.melt("report_period_ending", var_name="series", value_name="Bcf")


def test_band(frame: pd.DataFrame) -> alt.Chart:
    """shaded rectangle over the out-of-sample period"""
    band = pd.DataFrame({"start": [test_start], "end": [frame["report_period_ending"].max()]})
    return alt.Chart(band).mark_rect(opacity=0.07, color=MODEL).encode(x="start:T", x2="end:T")


def series_chart(frame: pd.DataFrame, cols: dict, height=320) -> alt.Chart:
    lines = alt.Chart(long_form(frame, cols)).mark_line(strokeWidth=1.6).encode(
        x=alt.X("report_period_ending:T", title=None),
        y=alt.Y("Bcf:Q", title="weekly storage change (Bcf)"),
        color=alt.Color("series:N", scale=SERIES_COLORS, legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("report_period_ending:T", title="week ending"), "series:N",
                 alt.Tooltip("Bcf:Q", format="+.1f")],
    )
    return (test_band(frame) + lines).properties(height=height)


# ── sidebar ──────────────────────────────────────────────────────────

st.sidebar.title("🔥 NG Storage Nowcaster")
st.sidebar.caption("Predicts the weekly EIA natural gas storage change from weather, before the report drops.")
page = st.sidebar.radio("Page", ["Overview", "Nowcast", "Backtest", "Revisions"], label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption(
    f"Trained on {df['report_period_ending'].min():%b %Y} – {df['report_period_ending'].max():%b %Y}. "
    f"Out-of-sample test from {test_start:%b %Y} (shaded on charts)."
)


# ═══════════════════════════════════════════════════════════════════
#  OVERVIEW
# ═══════════════════════════════════════════════════════════════════

if page == "Overview":
    st.title("NG Storage Nowcaster")
    st.markdown(
        "Every Thursday the EIA reports how much natural gas went into or out of US storage, and prices move on it. "
        "Most of that number is **weather**: cold weeks burn gas for heat, hot weeks burn it for power. "
        "This model turns NOAA degree-day data into a storage estimate **days before the report**, "
        "then flags weeks where it disagrees with the market consensus."
    )

    improvement = 1 - metrics["test_mae"] / base["five_year_avg"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model error (test MAE)", f"{metrics['test_mae']:.1f} Bcf")
    c2.metric("5-yr average error", f"{base['five_year_avg']:.1f} Bcf",
              help="Same week averaged over the prior 5 years — the benchmark EIA itself reports against.")
    c3.metric("Error reduction", f"{improvement:.0%}", help="vs the 5-year-average benchmark")
    if wf is not None:
        c4.metric("Years beating benchmark", f"{(wf['ensemble'] < wf['five_year_avg']).sum()} / {len(wf)}",
                  help="Walk-forward: each year predicted by a model trained only on prior years.")

    st.subheader("Out-of-sample: model vs EIA actual")
    recent = df[df["report_period_ending"] >= test_start - pd.DateOffset(years=1)]
    cols = {"storage_change_bcf": "EIA actual", "pred_bcf": "model", "five_yr_avg": "5-yr average"}
    st.altair_chart(series_chart(recent, cols), width="stretch")
    st.caption("Shaded region = data the model never saw during training. Negative = withdrawal, positive = injection.")

    if wf is not None:
        st.subheader("Does it hold up every year?")
        st.caption("Walk-forward test: for each year, train on all prior years only, then predict that year.")
        wf_long = wf.rename(columns=WF_LABELS).melt(["year", "weeks"], list(WF_LABELS.values()),
                                                     var_name="series", value_name="MAE")
        st.altair_chart(
            alt.Chart(wf_long).mark_line(point=True).encode(
                x=alt.X("year:O", title=None),
                y=alt.Y("MAE:Q", title="mean absolute error (Bcf)", scale=alt.Scale(zero=True)),
                color=alt.Color("series:N", scale=WF_COLORS, legend=alt.Legend(orient="top", title=None)),
                tooltip=["year", "series", "MAE", "weeks"],
            ).properties(height=260),
            width="stretch",
        )

    st.subheader("How it works")
    a, b, c, d = st.columns(4)
    a.markdown("**1 · Weather**  \nNOAA CPC daily heating & cooling degree days, summed over each Sat→Fri storage week.")
    b.markdown("**2 · Features**  \nDegree days, their deviation from seasonal norm, lagged storage changes, seasonality.")
    c.markdown("**3 · Model**  \nAverage of gradient-boosted trees and a linear (ridge) model, trained only on past data.")
    d.markdown("**4 · Signal**  \nCompare to consensus; flag the gap only in weeks where weather is the main driver.")


# ═══════════════════════════════════════════════════════════════════
#  NOWCAST
# ═══════════════════════════════════════════════════════════════════

elif page == "Nowcast":
    from src.live import nowcast, log_revision, divergence_signal

    @st.cache_data(ttl=3600, show_spinner=False)  # weather/storage data don't change within an hour
    def cached_nowcast(week, consensus):
        return nowcast(week_end=week, consensus_bcf=consensus)

    st.title("Nowcast a storage week")

    last = df["report_period_ending"].max().date()
    nxt = last + timedelta(days=7)
    weeks = [nxt] + sorted(df["report_period_ending"].dt.date.unique(), reverse=True)[:52]
    actuals = df.set_index(df["report_period_ending"].dt.date)["storage_change_bcf"]

    c1, c2 = st.columns([2, 1])
    week = c1.selectbox(
        "Storage week ending (Friday)", weeks,
        format_func=lambda w: f"{w:%a %b %d, %Y}" + ("  —  not yet reported" if w == nxt else ""),
        help="EIA storage weeks run Saturday→Friday; the report comes out the following Thursday.",
    )
    consensus = c2.number_input("Consensus (Bcf, optional)", value=None, step=1.0,
                                placeholder="e.g. -85", help="From Reuters/Bloomberg/NGI. Negative = withdrawal.")

    if st.button("Run nowcast", type="primary"):
        with st.spinner("Pulling weather + storage data…"):
            try:
                st.session_state.result = cached_nowcast(week, consensus)
                log_revision(st.session_state.result)
            except Exception as e:
                st.error(f"Nowcast failed: {e}")

    r = st.session_state.get("result")
    if r and r["week_ending"] == week:
        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted change", f"{r['pred_bcf']:+.1f} Bcf")
        m2.metric("±1 RMSE range", f"{r['confidence_lo']:+.0f} to {r['confidence_hi']:+.0f}")
        m3.metric("Heating degree days", f"{r['week_hdd']:.0f}", f"{r['hdd_deviation']:+.0f} vs normal",
                  delta_color="off")
        if week in actuals.index:
            err = r["pred_bcf"] - actuals[week]
            m4.metric("EIA actual", f"{actuals[week]:+.0f} Bcf", f"model off by {err:+.1f}", delta_color="off")
        else:
            m4.metric("EIA actual", "pending")

        if "weather_dominant" in r["regime_hint"]:
            st.info("**Weather-driven week.** Temperatures were far from normal, so weather explains most of "
                    "this week's move — a gap vs consensus is most meaningful here.")
        else:
            st.info("**Near-normal weather.** LNG exports, production and power burn matter more this week; "
                    "treat any gap vs consensus with caution.")

        if r["gap_bcf"] is not None:
            strength, direction = divergence_signal(r["gap_bcf"])
            g1, g2, g3 = st.columns(3)
            g1.metric("Consensus", f"{r['consensus_bcf']:+.1f} Bcf")
            g2.metric("Model − consensus", f"{r['gap_bcf']:+.1f} Bcf")
            g3.metric("Divergence", strength.title())
            st.markdown(f"**Read:** {direction}.")


# ═══════════════════════════════════════════════════════════════════
#  BACKTEST
# ═══════════════════════════════════════════════════════════════════

elif page == "Backtest":
    st.title("Backtest")

    c1, c2, c3 = st.columns(3)
    c1.metric("Train MAE", f"{metrics['train_mae']:.1f} Bcf")
    c2.metric("Test MAE", f"{metrics['test_mae']:.1f} Bcf")
    c3.metric("Test RMSE", f"{metrics['test_rmse']:.1f} Bcf")

    st.subheader("Full history")
    st.altair_chart(series_chart(df, {"storage_change_bcf": "EIA actual", "pred_bcf": "model", "five_yr_avg": "5-yr average"}, height=280),
                    width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("What drives the prediction")
        import joblib
        tm = joblib.load(MODEL_F)
        trees = getattr(tm.model, "named_estimators_", {}).get("gbrt", tm.model)
        if hasattr(trees, "feature_importances_"):
            imp = pd.DataFrame({"feature": [FEATURE_LABELS.get(f, f) for f in tm.feature_names],
                                "importance": trees.feature_importances_})
            st.altair_chart(
                alt.Chart(imp).mark_bar(color=MODEL).encode(
                    x=alt.X("importance:Q", title="importance in the tree half of the ensemble",
                            axis=alt.Axis(format="%")),
                    y=alt.Y("feature:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False, labelLimit=260)),
                    tooltip=["feature", alt.Tooltip("importance:Q", format=".1%")],
                ).properties(height=440),
                width="stretch",
            )
        else:
            st.caption("Feature importance is shown for the GBRT model only.")

    with right:
        st.subheader("Test-period errors")
        test = df[df["report_period_ending"] >= test_start]
        st.altair_chart(
            alt.Chart(test).mark_bar(color=MODEL, opacity=0.8).encode(
                x=alt.X("residual:Q", bin=alt.Bin(maxbins=30), title="actual − predicted (Bcf)"),
                y=alt.Y("count()", title="weeks"),
            ).properties(height=200),
            width="stretch",
        )
        bias = test["residual"].mean()
        st.caption(f"Mean error {bias:+.1f} Bcf — {'roughly unbiased' if abs(bias) < 3 else 'some systematic bias'}.")

        st.subheader("Error by regime")
        reg = test.groupby("regime")["residual"].agg(
            weeks="count", MAE=lambda x: x.abs().mean()
        ).round(1)
        st.dataframe(reg, width="stretch")
        st.caption("Regime is tagged after the fact: *weather_dominant* weeks are ones the weather model explained well; "
                   "*fundamental_driven* ones had something else (LNG, production, power burn) move storage.")

    if wf is not None:
        st.subheader("Walk-forward by year (MAE, Bcf)")
        st.dataframe(wf.rename(columns=WF_LABELS).set_index("year"), width="stretch")

    with st.expander("Last 20 weeks of data"):
        cols = ["report_period_ending", "storage_change_bcf", "pred_bcf", "residual",
                "week_hdd", "week_cdd", "hdd_deviation", "regime"]
        st.dataframe(df[cols].tail(20).iloc[::-1].reset_index(drop=True), width="stretch")


# ═══════════════════════════════════════════════════════════════════
#  REVISIONS
# ═══════════════════════════════════════════════════════════════════

elif page == "Revisions":
    st.title("Revision tracker")
    st.markdown("Each nowcast run is logged. Running Mon→Wed shows how the estimate moves as weather data updates — "
                "the *direction* of revision is often more informative than the level.")

    if not REVISIONS.exists():
        st.info("No runs logged yet. Run a nowcast first.")
        st.stop()
    rev = pd.read_csv(REVISIONS, parse_dates=["run_date", "week_ending"])
    week = st.selectbox("Storage week", sorted(rev["week_ending"].unique(), reverse=True),
                        format_func=lambda w: f"{pd.Timestamp(w):%b %d, %Y}")
    wk = rev[rev["week_ending"] == week].sort_values("run_date")
    wk["revision"] = wk["pred_bcf"].diff()

    if len(wk) > 1:
        st.altair_chart(
            alt.Chart(wk).mark_line(point=True, color=MODEL).encode(
                x=alt.X("run_date:T", title="run date"),
                y=alt.Y("pred_bcf:Q", title="predicted change (Bcf)", scale=alt.Scale(zero=False)),
            ).properties(height=260),
            width="stretch",
        )
    else:
        st.caption("Only one run for this week so far — run it again tomorrow to see a revision.")
    st.dataframe(wk[["run_date", "pred_bcf", "revision", "week_hdd", "consensus_bcf", "gap_bcf"]],
                 width="stretch", hide_index=True)
