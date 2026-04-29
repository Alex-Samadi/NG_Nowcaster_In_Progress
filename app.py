"""
app.py - Streamlit dashboard for NG Storage Nowcaster

run:  streamlit run app.py
"""
from __future__ import annotations
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="NG Storage Nowcaster",
    page_icon="🔥",
    layout="wide",
)

# ── paths ────────────────────────────────────────────────────────────
OUTPUTS   = Path("outputs")
MODELS    = Path("models")
DATASET   = OUTPUTS / "weekly_dataset.csv"
METRICS   = OUTPUTS / "metrics.json"
REVISIONS = OUTPUTS / "revisions.csv"
CONSENSUS = OUTPUTS / "consensus_history.csv"
MODEL_F   = MODELS / "model.joblib"


# ── helpers ──────────────────────────────────────────────────────────

def _most_recent_friday(d: date) -> date:
    wd = d.weekday()
    delta = (wd - 4) % 7
    return d - timedelta(days=delta)


def _load_dataset() -> pd.DataFrame | None:
    if not DATASET.exists():
        return None
    df = pd.read_csv(DATASET, parse_dates=["report_period_ending"])
    df["report_period_ending"] = df["report_period_ending"].dt.date
    return df


def _load_metrics() -> dict | None:
    if not METRICS.exists():
        return None
    return json.loads(METRICS.read_text())


def _load_consensus() -> pd.DataFrame:
    if not CONSENSUS.exists():
        return pd.DataFrame(columns=["report_period_ending", "consensus_bcf", "source"])
    df = pd.read_csv(CONSENSUS, parse_dates=["report_period_ending"])
    df["report_period_ending"] = df["report_period_ending"].dt.date
    return df


def _save_consensus(df: pd.DataFrame):
    OUTPUTS.mkdir(exist_ok=True)
    df.to_csv(CONSENSUS, index=False)


def _load_revisions() -> pd.DataFrame | None:
    if not REVISIONS.exists():
        return None
    return pd.read_csv(REVISIONS, parse_dates=["run_date", "week_ending"])


# ── sidebar ──────────────────────────────────────────────────────────

st.sidebar.title("NG Storage Nowcaster")
st.sidebar.caption("weather -> storage -> edge")

page = st.sidebar.radio(
    "navigate",
    ["live nowcast", "backtest history", "model vs consensus", "revision tracker"],
)

metrics = _load_metrics()
if metrics:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**model performance (test)**")
    st.sidebar.metric("MAE", f"{metrics['test_mae']:.1f} Bcf")
    st.sidebar.metric("RMSE", f"{metrics['test_rmse']:.1f} Bcf")
    mape = metrics.get("test_mape_big_weeks")
    if mape is not None:
        st.sidebar.metric("MAPE (big weeks)", f"{mape:.1f}%")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**how to use**\n\n"
    "1. run `python run.py backtest` to train\n"
    "2. run `python run.py live` or use this UI\n"
    "3. enter consensus to see divergence signal"
)


# ═══════════════════════════════════════════════════════════════════
#  PAGE: LIVE NOWCAST
# ═══════════════════════════════════════════════════════════════════

if page == "live nowcast":
    st.title("live nowcast")
    st.markdown(
        "run a prediction for the current/upcoming EIA storage report. "
        "the model uses HDD/CDD + lags + seasonality to estimate the weekly "
        "storage change (Bcf). compare against consensus to find divergence."
    )

    if not MODEL_F.exists():
        st.error(
            "no trained model found. run this first:\n\n"
            "```\npython run.py backtest --start 2014-01-01 --test-years 2\n```"
        )
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        report_date = st.date_input(
            "EIA report date (Thursday)",
            value=date.today(),
            help="the Thursday the report drops. covered week = prior Friday.",
        )
        # covered week = most recent Friday before report date
        covered_week = _most_recent_friday(report_date - timedelta(days=1))
        st.info(f"covered storage week ending: **{covered_week}**")

    with col2:
        consensus_input = st.number_input(
            "consensus estimate (Bcf)",
            value=0.0,
            step=1.0,
            help="from Celsius, NGI, or wherever you get it. negative = withdrawal.",
        )
        use_consensus = st.checkbox("compare to consensus", value=consensus_input != 0.0)

    if st.button("run nowcast", type="primary"):
        with st.spinner("pulling data + running model ..."):
            from src.live import nowcast, log_revision

            consensus_val = consensus_input if use_consensus else None
            try:
                result = nowcast(week_end=covered_week, consensus_bcf=consensus_val)
            except Exception as e:
                st.error(f"nowcast failed: {e}")
                st.stop()

            log_revision(result)

        # display results
        st.markdown("---")
        st.subheader("prediction")

        c1, c2, c3 = st.columns(3)
        c1.metric("predicted Δ storage", f"{result['pred_bcf']:+.1f} Bcf")
        c2.metric("confidence band", f"[{result['confidence_lo']:+.1f}, {result['confidence_hi']:+.1f}]")
        c3.metric("week HDD", f"{result['week_hdd']:.0f}")

        st.markdown(f"**HDD vs seasonal:** {result['hdd_deviation']:+.1f}  |  **regime hint:** {result['regime_hint']}")

        if use_consensus and result["gap_bcf"] is not None:
            st.markdown("---")
            st.subheader("consensus comparison")

            gc1, gc2, gc3 = st.columns(3)
            gc1.metric("consensus", f"{consensus_input:+.1f} Bcf")
            gc2.metric("gap (model − consensus)", f"{result['gap_bcf']:+.1f} Bcf")

            abs_gap = abs(result["gap_bcf"])
            if abs_gap < 5:
                signal_text = "NEUTRAL"
                signal_color = "gray"
            elif abs_gap < 15:
                signal_text = "MODERATE divergence"
                signal_color = "orange"
            elif abs_gap < 25:
                signal_text = "HIGH divergence"
                signal_color = "red"
            else:
                signal_text = "EXTREME divergence"
                signal_color = "red"

            direction = "BULLISH (model expects larger draw)" if result["gap_bcf"] < 0 else "BEARISH (model expects smaller draw)"
            gc3.metric("signal", f"{signal_text}")
            st.markdown(f"**direction:** {direction}")

            # save consensus to history
            cons_df = _load_consensus()
            new_row = pd.DataFrame([{
                "report_period_ending": covered_week,
                "consensus_bcf": consensus_input,
                "source": "manual",
            }])
            cons_df = pd.concat([cons_df, new_row], ignore_index=True)
            cons_df = cons_df.drop_duplicates(subset=["report_period_ending"], keep="last")
            _save_consensus(cons_df)


# ═══════════════════════════════════════════════════════════════════
#  PAGE: BACKTEST HISTORY
# ═══════════════════════════════════════════════════════════════════

elif page == "backtest history":
    st.title("backtest history")

    df = _load_dataset()
    if df is None:
        st.warning("no dataset found. run `python run.py backtest` first.")
        st.stop()

    # ── accuracy over time ────────────────────────────────────────
    st.subheader("model vs actual (storage change)")

    if "pred_bcf" in df.columns:
        fig, ax = plt.subplots(figsize=(14, 5))
        dates = pd.to_datetime(df["report_period_ending"])
        ax.plot(dates, df["storage_change_bcf"], label="actual", color="#2563eb", linewidth=1.2)
        ax.plot(dates, df["pred_bcf"], label="model", color="#dc2626", linewidth=1.0, alpha=0.8)
        ax.fill_between(
            dates,
            df["pred_bcf"] - 20,
            df["pred_bcf"] + 20,
            alpha=0.1, color="#dc2626", label="±RMSE band"
        )
        ax.set_ylabel("Bcf")
        ax.legend(loc="lower left")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.tight_layout()
        st.pyplot(fig)

    # ── residual distribution ─────────────────────────────────────
    if "residual" in df.columns:
        st.subheader("residual distribution")
        col1, col2 = st.columns(2)

        with col1:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.hist(df["residual"].dropna(), bins=40, color="#2563eb", alpha=0.7, edgecolor="white")
            ax2.axvline(0, color="#dc2626", linestyle="--", linewidth=1)
            ax2.set_xlabel("residual (Bcf)")
            ax2.set_ylabel("count")
            plt.tight_layout()
            st.pyplot(fig2)

        with col2:
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            dates = pd.to_datetime(df["report_period_ending"])
            ax3.scatter(dates, df["residual"], s=8, alpha=0.5, color="#2563eb")
            ax3.axhline(0, color="#dc2626", linestyle="--", linewidth=1)
            ax3.set_ylabel("residual (Bcf)")
            ax3.xaxis.set_major_locator(mdates.YearLocator())
            ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            plt.tight_layout()
            st.pyplot(fig3)

    # ── regime breakdown ──────────────────────────────────────────
    if "regime" in df.columns:
        st.subheader("regime breakdown")
        regime_stats = df.groupby("regime").agg(
            count=("residual", "count"),
            mae=("residual", lambda x: x.abs().mean()),
            rmse=("residual", lambda x: (x**2).mean()**0.5),
        ).round(1)
        st.dataframe(regime_stats)

        st.markdown(
            "**weather_dominant** = model residual is small -> weather explained the move.  \n"
            "**fundamental_driven** = model missed -> something else (LNG, production, etc.) drove it.  \n"
            "**your edge:** when you're in a weather_dominant regime AND your model diverges from consensus, lean into it."
        )

    # ── raw data ──────────────────────────────────────────────────
    with st.expander("raw dataset (last 20 weeks)"):
        show_cols = [
            "report_period_ending", "storage_change_bcf", "pred_bcf",
            "residual", "week_hdd", "week_cdd", "hdd_deviation", "regime",
        ]
        show_cols = [c for c in show_cols if c in df.columns]
        st.dataframe(df[show_cols].tail(20).reset_index(drop=True))


# ═══════════════════════════════════════════════════════════════════
#  PAGE: MODEL VS CONSENSUS
# ═══════════════════════════════════════════════════════════════════

elif page == "model vs consensus":
    st.title("model vs EIA actual vs consensus")
    st.markdown(
        "compare your weather-model predictions against the actual EIA number "
        "and (when available) the Celsius/street consensus. "
        "this is where you see if your model adds value."
    )

    df = _load_dataset()
    cons_df = _load_consensus()

    if df is None:
        st.warning("no dataset found. run backtest first.")
        st.stop()

    if "pred_bcf" not in df.columns:
        st.warning("no predictions in dataset. run backtest first.")
        st.stop()

    # merge consensus
    if not cons_df.empty:
        df = df.merge(cons_df[["report_period_ending", "consensus_bcf"]], on="report_period_ending", how="left")
    else:
        df["consensus_bcf"] = np.nan

    # add errors
    df["model_error"] = (df["pred_bcf"] - df["storage_change_bcf"]).abs()
    df["consensus_error"] = (df["consensus_bcf"] - df["storage_change_bcf"]).abs()

    # ── upload consensus CSV ──────────────────────────────────────
    st.subheader("add consensus data")
    st.markdown("upload a CSV with columns: `report_period_ending` (YYYY-MM-DD), `consensus_bcf`")

    uploaded = st.file_uploader("consensus CSV", type=["csv"])
    if uploaded:
        try:
            new_cons = pd.read_csv(uploaded, parse_dates=["report_period_ending"])
            new_cons["report_period_ending"] = new_cons["report_period_ending"].dt.date
            new_cons["source"] = "upload"
            existing = _load_consensus()
            combined = pd.concat([existing, new_cons], ignore_index=True)
            combined = combined.drop_duplicates(subset=["report_period_ending"], keep="last")
            _save_consensus(combined)
            st.success(f"saved {len(new_cons)} consensus entries")
            st.rerun()
        except Exception as e:
            st.error(f"failed to parse CSV: {e}")

    # ── manual entry ──────────────────────────────────────────────
    with st.expander("or enter manually"):
        mc1, mc2 = st.columns(2)
        with mc1:
            manual_week = st.date_input("week ending (Friday)", value=_most_recent_friday(date.today()))
        with mc2:
            manual_bcf = st.number_input("consensus (Bcf)", value=0.0, step=1.0)
        if st.button("save consensus entry"):
            existing = _load_consensus()
            new_row = pd.DataFrame([{
                "report_period_ending": manual_week,
                "consensus_bcf": manual_bcf,
                "source": "manual",
            }])
            combined = pd.concat([existing, new_row], ignore_index=True)
            combined = combined.drop_duplicates(subset=["report_period_ending"], keep="last")
            _save_consensus(combined)
            st.success("saved")

    # ── comparison table ──────────────────────────────────────────
    st.markdown("---")
    st.subheader("comparison (recent weeks)")

    has_consensus = df["consensus_bcf"].notna()
    if has_consensus.any():
        comp = df[has_consensus].copy()
        comp = comp.sort_values("report_period_ending", ascending=False).head(20)

        display_cols = [
            "report_period_ending", "storage_change_bcf", "pred_bcf",
            "model_error", "consensus_bcf", "consensus_error", "regime",
        ]
        display_cols = [c for c in display_cols if c in comp.columns]
        st.dataframe(comp[display_cols].reset_index(drop=True))

        # who wins?
        model_wins = (comp["model_error"] < comp["consensus_error"]).sum()
        total = len(comp)
        st.metric(
            "model closer to actual than consensus",
            f"{model_wins}/{total} weeks ({model_wins/total*100:.0f}%)"
        )
    else:
        st.info("no consensus data entered yet. add some above to see comparisons.")

    # ── chart: actual vs model vs consensus ───────────────────────
    if has_consensus.any():
        comp = df[has_consensus].sort_values("report_period_ending")
        fig, ax = plt.subplots(figsize=(12, 5))
        dates = pd.to_datetime(comp["report_period_ending"])
        ax.plot(dates, comp["storage_change_bcf"], "o-", label="EIA actual", color="#2563eb", markersize=5)
        ax.plot(dates, comp["pred_bcf"], "s--", label="model", color="#dc2626", markersize=4)
        ax.plot(dates, comp["consensus_bcf"], "^--", label="consensus", color="#16a34a", markersize=4)
        ax.set_ylabel("Bcf")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)


# ═══════════════════════════════════════════════════════════════════
#  PAGE: REVISION TRACKER
# ═══════════════════════════════════════════════════════════════════

elif page == "revision tracker":
    st.title("revision tracker")
    st.markdown(
        "each time you run `python run.py live` or use the live nowcast page, "
        "the prediction gets logged here. track how the forecast revises day-to-day "
        "as weather updates."
    )

    revisions = _load_revisions()
    if revisions is None or revisions.empty:
        st.info("no revisions logged yet. run a live nowcast first.")
        st.stop()

    # pick a week to view
    weeks = sorted(revisions["week_ending"].dropna().unique(), reverse=True)
    selected_week = st.selectbox("select storage week", weeks)

    wk_data = revisions[revisions["week_ending"] == selected_week].sort_values("run_date")

    if len(wk_data) > 1:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(wk_data["run_date"], wk_data["pred_bcf"], "o-", color="#2563eb", markersize=6)
        if wk_data["consensus_bcf"].notna().any():
            cons_val = wk_data["consensus_bcf"].dropna().iloc[-1]
            ax.axhline(cons_val, color="#16a34a", linestyle="--", label=f"consensus ({cons_val:.0f})")
        ax.set_ylabel("predicted Δ storage (Bcf)")
        ax.set_xlabel("run date")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

        # revision deltas
        wk_data = wk_data.copy()
        wk_data["revision"] = wk_data["pred_bcf"].diff()
        st.dataframe(wk_data[["run_date", "pred_bcf", "revision", "week_hdd", "consensus_bcf"]].reset_index(drop=True))
    else:
        st.dataframe(wk_data.reset_index(drop=True))
        st.info("only 1 run for this week. run the live nowcast again tomorrow to see revisions.")
