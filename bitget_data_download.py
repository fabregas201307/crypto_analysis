import requests
import pandas as pd

def get_bitget_klines(symbol, interval, start_time, end_time=None, limit=1000):
    url = "https://api.bitget.com/api/spot/v1/market/history-candles"
    params = {
        "symbol": symbol,
        "granularity": interval,  # Use "1DAY", "1HOUR", "1MIN"
        "limit": limit,
        "startTime": int(pd.Timestamp(start_time).timestamp() * 1000)
    }
    if end_time:
        params["endTime"] = int(pd.Timestamp(end_time).timestamp() * 1000)
    response = requests.get(url, params=params)
    print("Request URL:", response.url)
    print("Status Code:", response.status_code)
    print("Response:", response.text)
    print("Params:", params)
    data = response.json()
    if "data" not in data or not data["data"]:
        print("No data returned:", data)
        return pd.DataFrame()
    df = pd.DataFrame(data["data"], columns=[
        "Open time", "Open", "High", "Low", "Close", "Volume"
    ])
    df["Open time"] = pd.to_datetime(df["Open time"], unit="ms")
    return df

if __name__ == "__main__":
    # Bitget symbols: "BTCUSDT", "ETHUSDT"
    btc_df = get_bitget_klines("BTCUSDT", "1DAY", "2024-01-01", "2024-12-01")
    eth_df = get_bitget_klines("ETHUSDT", "1DAY", "2024-01-01", "2024-12-01")

    btc_df.to_csv("data/BTCUSDT_bitget_1d.csv", index=False)
    eth_df.to_csv("data/ETHUSDT_bitget_1d.csv", index=False)