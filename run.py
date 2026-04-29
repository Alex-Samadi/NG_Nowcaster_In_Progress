#!/usr/bin/env python3
"""
run.py - CLI entry point

usage:
    python run.py backtest --start 2014-01-01 --test-years 2
    python run.py live --week 2026-02-06 --consensus -271
    python run.py live
"""
from __future__ import annotations
import argparse, json, sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd


def cmd_backtest(args):
    from src.dataset import build_dataset, FEATURE_COLS, TARGET_COL
    from src.model import train, classify_regime, save_model

    df = build_dataset(start=args.start)

    print(f"\ndataset: {len(df)} rows, {len(FEATURE_COLS)} features")
    print(f"date range: {df['report_period_ending'].min()} -> {df['report_period_ending'].max()}\n")

    tm, df = train(df, test_years=args.test_years, model_type=args.model)

    # regime tagging
    df = classify_regime(df)

    # save
    save_model(tm)
    Path("outputs").mkdir(exist_ok=True)
    df.to_csv("outputs/weekly_dataset.csv", index=False)

    metrics = {
        "train_mae": round(tm.train_mae, 2),
        "train_rmse": round(tm.train_rmse, 2),
        "test_mae": round(tm.test_mae, 2),
        "test_rmse": round(tm.test_rmse, 2),
        "test_mape_big_weeks": round(tm.test_mape_big, 2) if pd.notna(tm.test_mape_big) else None,
    }
    Path("outputs/metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\n-- backtest results -------------------------")
    print(f"  train MAE:  {tm.train_mae:.2f} Bcf")
    print(f"  test  MAE:  {tm.test_mae:.2f} Bcf")
    print(f"  test  RMSE: {tm.test_rmse:.2f} Bcf")
    if pd.notna(tm.test_mape_big):
        print(f"  test  MAPE (|actual|>=50): {tm.test_mape_big:.1f}%")
    print("---------------------------------------------")
    print(f"\nsaved: outputs/weekly_dataset.csv, outputs/metrics.json, models/model.joblib")

    # regime summary
    regime_counts = df["regime"].value_counts()
    print(f"\nregime breakdown (full dataset):")
    for r, c in regime_counts.items():
        print(f"  {r}: {c} weeks ({c/len(df)*100:.0f}%)")


def cmd_live(args):
    from src.live import nowcast, log_revision

    week_end = None
    if args.week and args.week != "recent":
        week_end = date.fromisoformat(args.week)

    consensus = args.consensus

    print("\n-- live nowcast -----------------------------")
    result = nowcast(week_end=week_end, consensus_bcf=consensus)

    print(f"\n  storage week ending:  {result['week_ending']}")
    print(f"  predicted storage:    {result['pred_bcf']:+.1f} Bcf")
    print(f"  confidence band:      [{result['confidence_lo']:+.1f}, {result['confidence_hi']:+.1f}]")
    print(f"  week HDD:             {result['week_hdd']}")
    print(f"  week CDD:             {result['week_cdd']}")
    print(f"  HDD vs seasonal:      {result['hdd_deviation']:+.1f}")
    print(f"  regime hint:          {result['regime_hint']}")

    if consensus is not None:
        print(f"\n  consensus:            {consensus:+.1f} Bcf")
        print(f"  gap (model-consensus):{result['gap_bcf']:+.1f} Bcf")

        abs_gap = abs(result['gap_bcf'])
        if abs_gap < 5:
            signal = "NEUTRAL - model ~ consensus"
        elif abs_gap < 15:
            signal = "MODERATE divergence"
        elif abs_gap < 25:
            signal = "HIGH divergence"
        else:
            signal = "EXTREME divergence"

        direction = "BULLISH (larger draw)" if result['gap_bcf'] < 0 else "BEARISH (smaller draw)"
        print(f"  signal:               {signal} -> {direction}")

    print("---------------------------------------------\n")

    log_revision(result)
    print("  revision logged -> outputs/revisions.csv")


def main():
    parser = argparse.ArgumentParser(description="NG Storage Nowcaster")
    subs = parser.add_subparsers(dest="command")

    # backtest
    bt = subs.add_parser("backtest", help="build dataset + train + evaluate")
    bt.add_argument("--start", default="2014-01-01", help="earliest EIA data")
    bt.add_argument("--test-years", type=int, default=2, help="hold-out test years")
    bt.add_argument("--model", default="gbrt", choices=["gbrt", "ridge"])
    bt.set_defaults(func=cmd_backtest)

    # live
    lv = subs.add_parser("live", help="nowcast for upcoming/recent storage week")
    lv.add_argument("--week", default=None, help="week-ending Friday (YYYY-MM-DD) or 'recent'")
    lv.add_argument("--consensus", type=float, default=None, help="consensus estimate (Bcf)")
    lv.set_defaults(func=cmd_live)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()