"""
Production Pipeline Orchestrator

This script manages the daily data-to-signal pipeline:
1. It checks the latest date in the existing crypto data file.
2. It runs the download script to get fresh data.
3. It compares the latest date in the new file against the old one.
4. If a new date is found, it triggers the daily alpha signal generation script.

This allows you to run this script frequently (e.g., via cron) without
re-calculating signals unless new daily data is actually available.
"""
import os
import pandas as pd
import subprocess
import sys
from datetime import datetime

DATA_FILE = "data/Binance_AllCrypto_d.csv"
DOWNLOAD_SCRIPT = "crypto_data_download.py"
SIGNAL_SCRIPT = "daily_alpha_signals.py"

def get_latest_date(file_path: str) -> pd.Timestamp | None:
    """Reads a CSV file and returns the latest date found in the 'Date' column."""
    if not os.path.exists(file_path):
        print(f"[INFO] Data file not found at {file_path}. Cannot determine latest date.")
        return None
    try:
        df = pd.read_csv(file_path, usecols=['Date'])
        if df.empty:
            return None
        return pd.to_datetime(df['Date']).max()
    except Exception as e:
        print(f"[ERROR] Failed to read latest date from {file_path}: {e}")
        return None

def run_script(script_name: str, *args):
    """Executes a Python script in the same environment and handles errors."""
    print(f"\n{'='*20} RUNNING: {script_name} {'='*20}")
    try:
        # Use sys.executable to ensure the same Python interpreter is used
        process = subprocess.run(
            [sys.executable, script_name, *args],
            check=True,
            capture_output=True,
            text=True
        )
        print(process.stdout)
        if process.stderr:
            print("[STDERR]:")
            print(process.stderr)
        print(f"[SUCCESS] Finished running {script_name}.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to run {script_name}.")
        print(e.stdout)
        print(e.stderr)
        return False
    except FileNotFoundError:
        print(f"[ERROR] Script not found: {script_name}. Make sure it's in the correct path.")
        return False

def main():
    """Main pipeline logic."""
    print(f"Pipeline started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Get the latest date before downloading new data
    last_known_date = get_latest_date(DATA_FILE)
    if last_known_date:
        print(f"Last known data date is: {last_known_date.strftime('%Y-%m-%d')}")
    else:
        print("No previous data found. Will run signal generation after download.")

    # 2. Run the download script
    if not run_script(DOWNLOAD_SCRIPT):
        print("Halting pipeline due to download failure.")
        return

    # 3. Get the latest date after downloading
    new_latest_date = get_latest_date(DATA_FILE)
    if not new_latest_date:
        print("Could not determine date from new data. Halting.")
        return
    print(f"Latest data date is now: {new_latest_date.strftime('%Y-%m-%d')}")

    # 4. Compare dates and trigger signal generation if new data is found
    if last_known_date is None or new_latest_date > last_known_date:
        print("\nNew daily data detected! Triggering signal generation...")
        # Run daily_alpha_signals.py
        # You can customize the arguments here if needed
        if not run_script(SIGNAL_SCRIPT, "--input", DATA_FILE):
            print("Signal generation script failed.")
        else:
            print("Signal generation complete.")
    else:
        print("\nNo new daily data found. No action taken.")

    print(f"\nPipeline finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
