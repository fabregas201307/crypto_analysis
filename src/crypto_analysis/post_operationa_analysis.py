"""Post Operational Analysis

Purpose:
  Take the generated live signal files (data/live_signals_YYYY-MM-DD.csv)
  and build a lightweight attribution & performance summary:
    * Consolidated signals table
    * Daily exposure diagnostics (net, gross, concentration, top weight, n assets)
    * Realized strategy returns using next-day close-to-close returns
    * Cumulative performance curve & optional plots

Assumptions:
  - Each live_signals_*.csv contains columns: Date, Ticker, combined_alpha, alpha_rank, weight
  - We treat weights on Date D as the weights that will earn returns from D->D+1 (consistent with 1-day lag signaling process)
  - Price file (e.g. data/Binance_AllCrypto_d.csv) has at least: Date, Ticker, Close (case-insensitive accepted)

Usage Example:
  python -m crypto_analysis.post_operationa_analysis \
      --signals-dir data \
      --prices data/Binance_AllCrypto_d.csv \
      --output-dir data/post_analysis \
      --plot

Outputs written to output-dir:
  combined_signals.csv
  exposure_summary.csv
  strategy_daily_returns.csv
  strategy_equity_curve.png (if --plot)
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import List

try:
    import matplotlib.pyplot as plt  # optional
    _HAVE_MPL = True
except Exception:  # pragma: no cover
    _HAVE_MPL = False


def _discover_signal_files(signals_dir: str) -> List[str]:
    pattern = os.path.join(signals_dir, "live_signals_*.csv")
    files = sorted(glob.glob(pattern))
    return files


def load_and_concat_signals(signals_dir: str) -> pd.DataFrame:
    files = _discover_signal_files(signals_dir)
    if not files:
        raise FileNotFoundError(f"No live_signals_*.csv files found in {signals_dir}")
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if 'Date' not in df.columns or 'Ticker' not in df.columns or 'weight' not in df.columns:
                print(f"[WARN] Skipping {f} (missing required columns)")
                continue
            df['Date'] = pd.to_datetime(df['Date'])
            dfs.append(df)
        except Exception as e:  # pragma: no cover
            print(f"[WARN] Failed to read {f}: {e}")
    if not dfs:
        raise ValueError("No valid signal files parsed.")
    all_signals = pd.concat(dfs, ignore_index=True)
    all_signals = all_signals.sort_values(['Date', 'Ticker']).reset_index(drop=True)
    return all_signals


def compute_exposure_stats(signals: pd.DataFrame) -> pd.DataFrame:
    g = signals.groupby('Date')
    exposure = g.agg(
        net_exposure=('weight', 'sum'),
        gross_exposure=('weight', lambda x: x.abs().sum()),
        n_positions=('weight', lambda x: (x != 0).sum()),
        n_longs=('weight', lambda x: (x > 0).sum()),
        n_shorts=('weight', lambda x: (x < 0).sum()),
        herfindahl=('weight', lambda x: (x[x>0]**2).sum()),
        top_weight=('weight', lambda x: x[x>0].max() if (x>0).any() else 0.0),
    ).reset_index()
    return exposure


def _standardize_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_lower = {c.lower(): c for c in df.columns}
    # Accept 'close' or 'Close'
    close_col = cols_lower.get('close')
    if close_col is None:
        raise ValueError("Price file must contain 'Close' column")
    # Ensure schema uniformity
    needed = ['Date', 'Ticker', close_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Price file missing columns: {missing}")
    out = df[['Date', 'Ticker', close_col]].copy()
    out.rename(columns={close_col: 'Close'}, inplace=True)
    out['Date'] = pd.to_datetime(out['Date'])
    return out


def compute_portfolio_returns(signals: pd.DataFrame, prices_csv: str) -> pd.DataFrame:
    prices = pd.read_csv(prices_csv)
    prices = _standardize_price_columns(prices)
    prices = prices.sort_values(['Ticker', 'Date']).reset_index(drop=True)

    # Compute forward daily returns per asset (Close_t+1 / Close_t - 1)
    prices['fwd_close'] = prices.groupby('Ticker')['Close'].shift(-1)
    prices['asset_ret'] = prices['fwd_close'] / prices['Close'] - 1.0

    # Align weights: weights on Date D apply to return from D -> D+1 (thus multiply by asset_ret at D)
    merged = signals.merge(prices[['Date', 'Ticker', 'asset_ret']], on=['Date', 'Ticker'], how='left')
    # Drop last day where no forward return yet
    merged = merged[~merged['asset_ret'].isna()].copy()

    daily = merged.groupby('Date').apply(lambda d: (d['weight'] * d['asset_ret']).sum()).to_frame('strategy_ret')
    daily.reset_index(inplace=True)
    daily['cum_return'] = (1 + daily['strategy_ret']).cumprod() - 1
    return daily


def maybe_plot(daily: pd.DataFrame, exposure: pd.DataFrame, output_dir: str):  # pragma: no cover
    if not _HAVE_MPL:
        print("[INFO] matplotlib not available; skipping plots")
        return
    import matplotlib.dates as mdates
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={'height_ratios': [2,1]})
    ax[0].plot(daily['Date'], daily['cum_return'], label='Cumulative Return')
    ax[0].axhline(0, color='gray', lw=0.8)
    ax[0].set_ylabel('Cum Return')
    ax[0].legend()

    ax[1].plot(exposure['Date'], exposure['gross_exposure'], label='Gross')
    ax[1].plot(exposure['Date'], exposure['net_exposure'], label='Net')
    ax[1].set_ylabel('Exposure')
    ax[1].legend()
    ax[1].xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    out_path = os.path.join(output_dir, 'strategy_equity_curve.png')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Plot saved: {out_path}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Aggregate live signal CSVs and compute realized performance.")
    p.add_argument('--signals-dir', default='data', help='Directory holding live_signals_*.csv files')
    p.add_argument('--prices', default='data/Binance_AllCrypto_d.csv', help='Daily price CSV used to compute returns')
    p.add_argument('--output-dir', default='data/post_analysis', help='Directory for output summary files')
    p.add_argument('--plot', action='store_true', help='Generate PNG plot if matplotlib available')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading signals ...")
    signals = load_and_concat_signals(args.signals_dir)
    print(f"Loaded {len(signals):,} signal rows across {signals['Date'].nunique()} dates")

    print("Computing exposure stats ...")
    exposure = compute_exposure_stats(signals)

    print("Computing realized returns ...")
    daily = compute_portfolio_returns(signals, args.prices)

    # Persist
    signals.to_csv(os.path.join(args.output_dir, 'combined_signals.csv'), index=False)
    exposure.to_csv(os.path.join(args.output_dir, 'exposure_summary.csv'), index=False)
    daily.to_csv(os.path.join(args.output_dir, 'strategy_daily_returns.csv'), index=False)

    # Weight matrix (rows=Date ascending, columns=Tickers, entries=weights)
    weight_matrix = (signals
                      .pivot_table(index='Date', columns='Ticker', values='weight', aggfunc='last', fill_value=0.0)
                      .sort_index())
    weight_matrix.to_csv(os.path.join(args.output_dir, 'weight_matrix.csv'))
    print(f"Wrote outputs to {args.output_dir} (including weight_matrix.csv)")

    if args.plot:
        maybe_plot(daily, exposure, args.output_dir)

    # Brief console summary
    print("\nSummary:")
    print("  Dates           :", exposure['Date'].min().date(), '->', exposure['Date'].max().date())
    print("  # Signal days   :", exposure.shape[0])
    print("  Final Cum Return:", f"{daily['cum_return'].iloc[-1]:.2%}")
    print("  Avg Gross Exp   :", f"{exposure['gross_exposure'].mean():.2f}")
    print("  Avg # Positions :", f"{exposure['n_positions'].mean():.1f}")
    print("  Avg Herfindahl  :", f"{exposure['herfindahl'].mean():.3f}")
    print("Done.")


if __name__ == '__main__':  # pragma: no cover
    main()
    