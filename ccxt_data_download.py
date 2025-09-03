import ccxt
import pandas as pd

def fetch_crypto_data(exchange, symbol, timeframe='1d', since_str='2025-01-01T00:00:00Z', limit=1000):
    since = exchange.parse8601(since_str)
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        df = pd.DataFrame(ohlcv, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    tickers = ["BTC/USDT", "ETH/USDT"]  # Add more tickers as needed
    exchange = ccxt.binance()
    dfs = []
    for ticker in tickers:
        df = fetch_crypto_data(exchange, ticker, timeframe='1d', since_str='2025-01-01T00:00:00Z', limit=1000)
        if not df.empty:
            df["Ticker"] = ticker.replace("/", "")
            dfs.append(df)
            df.to_csv(f"data/ccxt_Binance_{ticker.replace('/', '')}_ccxt_1d.csv", index=False)
            print(f"Saved: data/ccxt_Binance_{ticker.replace('/', '')}_ccxt_1d.csv")
    if dfs:
        all_df = pd.concat(dfs, ignore_index=True)
        all_df.to_csv("data/ccxt_Binance_AllCrypto_ccxt_1d.csv", index=False)
        print("Saved: data/ccxt_Binance_AllCrypto_ccxt_1d.csv")