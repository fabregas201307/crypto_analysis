#!/usr/bin/env python3
"""
Daily Alpha Signals Generator (Production)

Purpose:
- Use a fixed set of 20 alphas (determined in research) to compute a per-asset combined_alpha signal
- Rank cross-sectionally per day and produce normalized portfolio weights for daily trading
- Designed to run on latest daily CSV data (same schema as research)

Fairness defaults:
- Uses long-only mode with weights summing to 1.0 each day
- Applies a 1-day signal lag by default (use t-1 signals to trade on day t)

Outputs:
- data/live_signals_YYYY-MM-DD.csv (per-asset weights and diagnostics for the signal date)
- data/live_exposure_YYYY-MM-DD.csv (daily exposure summary)

Usage examples:
  python daily_alpha_signals.py --input data/Binance_AllCrypto_d.csv
  python daily_alpha_signals.py --input data/Binance_AllCrypto_d.csv --date 2025-09-05 --lag-days 1
  python daily_alpha_signals.py --mode long_short_gross1 --long-frac 0.3 --short-frac 0.3

Notes:
- By default this script computes ONLY the fixed top-20 alphas for speed.
- You can opt into computing the full alpha set via --full-factors if desired.
- If any required alphas are missing on a given day, it will drop missing ones and continue.
"""

from __future__ import annotations
import argparse
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Ensure local imports resolve
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(THIS_DIR)

# Import factor generation from research module (optional, only if --full-factors is used)
try:
    from crypto_strategy import calculate_extended_alpha_factors  # type: ignore
except Exception:
    calculate_extended_alpha_factors = None  # Full path optional; fast path is default

# Fixed top-20 alphas (by |IC|) discovered in research run
FIXED_TOP_ALPHAS = [
    'alpha_97', 'alpha_100', 'alpha_156', 'alpha_110', 'alpha_129',
    'alpha_147', 'alpha_148', 'alpha_120', 'alpha_159', 'alpha_112',
    'alpha_116', 'alpha_158', 'alpha_155', 'alpha_160', 'alpha_15',
    'alpha_42', 'alpha_31', 'alpha_149', 'alpha_146', 'alpha_121'
]


def _pick_signal_date(dates: pd.Series, user_date: str | None, lag_days: int) -> pd.Timestamp:
    # unique() may return a DatetimeArray without .sort(); use numpy sort for robustness
    dates_sorted = np.sort(pd.to_datetime(dates.unique()))
    if len(dates_sorted) == 0:
        raise ValueError("No dates found in input data")

    if user_date:
        d = pd.to_datetime(user_date)
        if d not in dates_sorted:
            # choose the last date before or equal to requested date
            d = dates_sorted[dates_sorted <= d].max()
        signal_date = d
    else:
        signal_date = dates_sorted.max()

    # apply lag: use t-lag_days as signal date
    idx = np.where(dates_sorted == signal_date)[0]
    if len(idx) == 0:
        raise ValueError("Signal date not found after normalization")
    i = idx[0] - lag_days
    if i < 0:
        raise ValueError("Not enough history to apply lag; reduce --lag-days or provide later --date")
    return pd.Timestamp(dates_sorted[i])


def compute_fixed_top_alphas(df: pd.DataFrame, top_alphas: list[str]) -> pd.DataFrame:
    """Compute only the fixed top-N alphas needed for production.

    This uses prepare_df and imports only the exact alpha functions needed across
    original 1-101, additional (102-140, 121-140), and extended (141-160) sets.
    """
    # Ensure deterministic column order and types
    df = df.copy()

    # Prepare common columns
    try:
        from crypto_data_analysis_new import prepare_df, alpha15, alpha31, alpha42, alpha97, alpha100
    except Exception as e:
        raise ImportError(f"Failed to import base alphas from crypto_data_analysis_new.py: {e}")

    try:
        # Additional alphas 102-140 and enhanced 121-140
        from additional_alpha_factors import (
            alpha_110, alpha_112, alpha_116, alpha_120, alpha_121, alpha_129
        )
    except Exception as e:
        raise ImportError(f"Failed to import additional alphas from additional_alpha_factors.py: {e}")

    try:
        # Extended alphas 141-160
        from extended_alpha_factors import (
            alpha_146, alpha_147, alpha_148, alpha_149, alpha_155,
            alpha_156, alpha_158, alpha_159, alpha_160
        )
    except Exception as e:
        raise ImportError(f"Failed to import extended alphas from extended_alpha_factors.py: {e}")

    # Build function map
    func_map: dict[str, callable] = {
        'alpha_15': alpha15,
        'alpha_31': alpha31,
        'alpha_42': alpha42,
        'alpha_97': alpha97,
        'alpha_100': alpha100,
        'alpha_110': alpha_110,
        'alpha_112': alpha_112,
        'alpha_116': alpha_116,
        'alpha_120': alpha_120,
        'alpha_121': alpha_121,
        'alpha_129': alpha_129,
        'alpha_146': alpha_146,
        'alpha_147': alpha_147,
        'alpha_148': alpha_148,
        'alpha_149': alpha_149,
        'alpha_155': alpha_155,
        'alpha_156': alpha_156,
        'alpha_158': alpha_158,
        'alpha_159': alpha_159,
        'alpha_160': alpha_160,
    }

    # Prepare base features required by many alphas
    df = prepare_df(df)

    # Compute only requested alphas
    for a in top_alphas:
        if a in df.columns:
            continue
        func = func_map.get(a)
        if func is None:
            # Skip unknown alpha names gracefully
            print(f"[WARN] No function mapped for {a}; filling zeros")
            df[a] = 0.0
            continue
        try:
            print(f"  Computing {a}...")
            series = func(df)
            # Ensure alignment and handle NaNs/infs
            df[a] = pd.Series(series, index=df.index)
            df[a] = df[a].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        except Exception as e:
            print(f"  ✗ Error computing {a}: {e}; filling zeros")
            df[a] = 0.0

    return df


def build_weights_for_date(df: pd.DataFrame, date: pd.Timestamp,
                           top_alphas: list[str],
                           mode: str = 'long_only',
                           long_frac: float = 0.3,
                           short_frac: float = 0.3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute combined_alpha, alpha_rank, and normalized weights for a single date.

    Returns (weights_table, exposure_summary).
    """
    day_df = df[df['Date'] == date].copy()
    if day_df.empty:
        raise ValueError(f"No rows for signal date {date.date()}")

    # Use only alphas that exist for this dataset
    available = [c for c in top_alphas if c in day_df.columns]
    missing = [c for c in top_alphas if c not in day_df.columns]
    if len(available) == 0:
        raise ValueError("None of the fixed top alphas are available in the input dataframe")
    if missing:
        print(f"[WARN] Missing alphas on {date.date()}: {missing} (using {len(available)} available)")

    # Combined signal
    day_df['combined_alpha'] = day_df[available].mean(axis=1)

    # Cross-sectional percentile rank
    day_df['alpha_rank'] = day_df['combined_alpha'].rank(pct=True, method='first')

    # Selection masks
    long_mask = day_df['alpha_rank'] >= (1 - long_frac)
    short_mask = day_df['alpha_rank'] <= short_frac

    # Initialize weights
    day_df['weight'] = 0.0

    if mode == 'long_only':
        n_long = int(long_mask.sum())
        if n_long > 0:
            day_df.loc[long_mask, 'weight'] = 1.0 / n_long
    elif mode == 'long_short_gross1':
        n_long = int(long_mask.sum())
        n_short = int(short_mask.sum())
        if n_long > 0:
            day_df.loc[long_mask, 'weight'] = 0.5 / n_long
        if n_short > 0:
            day_df.loc[short_mask, 'weight'] = -(0.5 / n_short)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Exposure summary
    exposure = day_df.groupby('Date').agg(
        net_exposure=('weight', 'sum'),
        gross_exposure=('weight', lambda x: x.abs().sum()),
        n_longs=('weight', lambda x: (x > 0).sum()),
        n_shorts=('weight', lambda x: (x < 0).sum())
    ).reset_index()

    # Final table
    out_cols = ['Date', 'Ticker', 'combined_alpha', 'alpha_rank', 'weight']
    weights = day_df.sort_values(['Date', 'Ticker'])[out_cols]
    return weights, exposure


def main():
    parser = argparse.ArgumentParser(description='Generate daily alpha weights from fixed top 20 alphas')
    parser.add_argument('--input', required=True, help='Path to daily CSV (e.g., data/Binance_AllCrypto_d.csv)')
    parser.add_argument('--date', default=None, help='Target trade date (YYYY-MM-DD). If omitted, uses latest date minus lag-days')
    parser.add_argument('--lag-days', type=int, default=1, help='Signal lag in days (default: 1)')
    parser.add_argument('--mode', choices=['long_only', 'long_short_gross1'], default='long_only', help='Weighting mode')
    parser.add_argument('--long-frac', type=float, default=0.3, help='Top fraction for longs (default: 0.3)')
    parser.add_argument('--short-frac', type=float, default=0.3, help='Bottom fraction for shorts (default: 0.3)')
    parser.add_argument('--output-dir', default='data', help='Directory to write outputs')
    parser.add_argument('--universe', nargs='*', help='Optional list of tickers to include (e.g., BTCUSDT ETHUSDT)')
    parser.add_argument('--full-factors', action='store_true', help='Compute full factor set (slow). Default uses fast top-20 only')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(args.input)
    if 'Date' not in df.columns or 'Ticker' not in df.columns:
        raise ValueError("Input CSV must contain at least 'Date' and 'Ticker' columns")

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Date', 'Ticker']).copy()

    if args.universe:
        df = df[df['Ticker'].isin(set(args.universe))].copy()
        if df.empty:
            raise ValueError("No rows left after universe filter")

    # Compute factors
    if args.full_factors:
        if calculate_extended_alpha_factors is None:
            raise RuntimeError("Full factor computation unavailable (import failed). Use fast path or fix imports.")
        print("Computing FULL alpha set (slow)...")
        df, _ = calculate_extended_alpha_factors(df)
    else:
        print("Computing FAST top-20 alphas only...")
        df = compute_fixed_top_alphas(df, FIXED_TOP_ALPHAS)

    # Determine signal date with lag
    signal_date = _pick_signal_date(df['Date'], args.date, args.lag_days)
    print(f"Signal date (after lag={args.lag_days}): {signal_date.date()}")

    # Build weights
    weights, exposure = build_weights_for_date(
        df, signal_date, FIXED_TOP_ALPHAS, mode=args.mode,
        long_frac=args.long_frac, short_frac=args.short_frac
    )

    date_str = signal_date.strftime('%Y-%m-%d')
    weights_file = os.path.join(args.output_dir, f'live_signals_{date_str}.csv')
    exposure_file = os.path.join(args.output_dir, f'live_exposure_{date_str}.csv')

    weights.to_csv(weights_file, index=False)
    exposure.to_csv(exposure_file, index=False)

    print(f"\nSignals written to: {weights_file}")
    print(f"Exposure written to: {exposure_file}")

    # Show a quick summary of the top positions
    if not weights.empty:
        w = weights.sort_values('weight', ascending=False).head(10)
        print("\nTop positions:")
        for r in w.itertuples(index=False):
            print(f"  {r.Ticker:>10}  weight={r.weight: .4f}  rank={r.alpha_rank: .3f}  ca={r.combined_alpha: .5f}")

    # Helpful reminder for live trading
    print("\nReminder:")
    print("- This uses fixed (in-sample) top-20 alphas and 1-day lag to mitigate look-ahead.")
    print("- Weights are normalized per the chosen mode; long_only sums to 1.0 daily.")
    print("- Consider adding basic costs and universe/liq filters in production.")


if __name__ == '__main__':
    main()


# . /Users/kaiwen/miniconda3/etc/profile.d/conda.sh && conda activate jpm && python /Users/kaiwen/Desktop/projects/crypto_analysis/daily_alpha_signals.py --input /Users/kaiwen/Desktop/projects/crypto_analysis/data/Binance_AllCrypto_d.csv --lag-days 0 --mode long_only

