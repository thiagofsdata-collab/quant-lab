import pandas as pd
import numpy as np
from pathlib import Path
import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def generate_signals(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    df = df.copy()

    df["bb_mid"] = df["close"].rolling(window).mean()
    df["bb_std"] = df["close"].rolling(window).std()
    df["bb_upper"] = df["bb_mid"] + (num_std * df["bb_std"])
    df["bb_lower"] = df["bb_mid"] - (num_std * df["bb_std"])

    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    buy_signal = df["close"] < df["bb_lower"]
    sell_signal = df["close"] > df["bb_upper"]

    df["signal"] = np.nan
    df.loc[buy_signal,"signal"] = 1
    df.loc[sell_signal,"signal"] = 0
    df["signal"] = df["signal"].ffill().fillna(0).astype(int)

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

        buys = (df["crossover"] ==  1).sum()
        sells = (df["crossover"] == -1).sum()
        pos = "LONG" if df["signal"].iloc[-1] == 1 else "FLAT"

        print(f"{slug:<12} {buys:>6} {sells:>6} {pos:>10}")

    print("\nDone.")


if __name__ == "__main__":
    run()

