import requests
import pandas as pd
import time

def get_binance_klines(symbol, interval, start_str, end_str=None, limit=1000):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": int(pd.Timestamp(start_str).timestamp() * 1000),
        "limit": limit
    }
    if end_str:
        params["endTime"] = int(pd.Timestamp(end_str).timestamp() * 1000)
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data, columns=[
        "Open time", "Open", "High", "Low", "Close", "Volume",
        "Close time", "Quote asset volume", "Number of trades",
        "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"
    ])
    df["Open time"] = pd.to_datetime(df["Open time"], unit="ms")
    df["Close time"] = pd.to_datetime(df["Close time"], unit="ms")
    return df[["Open time", "Open", "High", "Low", "Close", "Volume"]]

if __name__ == "__main__":
    # Example usage:
    btc_df = get_binance_klines("BTCUSDT", "1d", "2024-01-01", "2024-12-01")
    eth_df = get_binance_klines("ETHUSDT", "1d", "2024-01-01", "2024-12-01")

    btc_df.to_csv(f"data/BTCUSDT_1d.csv", index=False)
    eth_df.to_csv(f"data/ETHUSDT_1d.csv", index=False)