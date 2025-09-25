import pandas as pd

def download_cryptodatadownload_csv(url, output_path):
    df = pd.read_csv(url, skiprows=1)  # First row is header info, skip it
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    # List of tickers to download (major and low-fee cryptos)
    tickers = [
        "BTCUSDT",  # Bitcoin
        "ETHUSDT",  # Ethereum
        "SOLUSDT",  # Solana
        "BNBUSDT",  # Binance Coin
        "ADAUSDT",  # Cardano
        "XRPUSDT",  # Ripple (low fee)
        "DOGEUSDT", # Dogecoin (low fee)
        "MATICUSDT",# Polygon
        "DOTUSDT",  # Polkadot
        "AVAXUSDT", # Avalanche
        "TRXUSDT",  # TRON (very low fee)
        "LTCUSDT",  # Litecoin (low fee)
        "LINKUSDT", # Chainlink
        "XLMUSDT"   # Stellar (low fee)
    ]
    base_url = "https://www.cryptodatadownload.com/cdd/Binance_{}_d.csv"
    dfs = []
    for ticker in tickers:
        url = base_url.format(ticker)
        df = pd.read_csv(url, skiprows=1)
        df["Ticker"] = ticker
        df.rename(columns={f"Volume {ticker.replace('USDT', '')}": "Volume"}, inplace=True)
        dfs.append(df)
    # Concatenate all dataframes
    all_df = pd.concat(dfs, ignore_index=True)
    # Save to a single CSV file
    all_df.to_csv("data/Binance_AllCrypto_d.csv", index=False)
    print("Saved: data/Binance_AllCrypto_d.csv")