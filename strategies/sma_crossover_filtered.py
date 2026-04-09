import pandas as pd
import numpy as np
from pathlib import Path
import yaml

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def generate_signals(df: pd.DataFrame, fast: int = 20, slow: int = 50,
                     persistence: int = 2) -> pd.DataFrame:
    df = df.copy()

    df["volume_sma20"] = df["volume"].rolling(20).mean()

    trend = df[f"sma_{fast}"] > df[f"sma_{slow}"]
    not_overbought = df["rsi_14"] < 70
    vol_confirm = df["volume"] > df["volume_sma20"]

    raw_signal = (trend & not_overbought & vol_confirm).astype(int)

    df["signal"] = raw_signal.rolling(persistence).min().fillna(0).astype(int)

    df["crossover"] = df["signal"].diff()

    return df

def run():
    config = load_config()
    proc_path = Path(config["data"]["processed_path"])
    equities = [t.replace(".", "_") for t in config["assets"]["equities"]]

    print(f"\n{'Ticker':<12} {'Buys':>6} {'Sells':>6} {'Position':>10}")
    print("-" * 38)

    for slug in equities:
        file = proc_path / f"{slug}_2020_2024.parquet"
        if not file.exists():
            continue

        df = pd.read_parquet(file).sort_index()
        df = generate_signals(df)

        buys = (df["crossover"] == 1).sum()
        sells = (df["crossover"] == -1).sum()
        pos = "LONG" if df["signal"].iloc[-1] == 1 else "FLAT"

        print(f"{slug:<12} {buys:>6} {sells:>6} {pos:>10}")

    print("\nDone.")

if __name__ == "__main__":
    run()