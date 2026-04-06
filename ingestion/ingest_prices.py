# ingestion/ingest_prices.py

import yfinance as yf
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def fetch_prices(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df.columns = [c.lower() for c in df.columns]
    df.index.name = "date"
    df["ticker"] = ticker
    return df

def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=True)
    print(f"saved → {path} ({len(df)} rows)")

def run():
    config = load_config()

    tickers = config["assets"]["equities"] + config["assets"]["benchmarks"]
    start = config["data"]["start_date"]
    end = config["data"]["end_date"]
    interval = config["data"]["interval"]
    raw_path = Path(config["data"]["raw_path"])

    print(f"\nIngesting {len(tickers)} tickers | {start} → {end}\n")

    for ticker in tickers:
        try:
            df = fetch_prices(ticker, start, end, interval)
            slug = ticker.replace(".", "_").replace("^", "").replace("=", "")
            filename = f"{slug}_{start[:4]}_{end[:4]}.parquet"
            save_parquet(df, raw_path / filename)
        except Exception as e:
            print(f"  ERROR {ticker}: {e}")

    print("\nDone.")

if __name__ == "__main__":
    run()