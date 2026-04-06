import pandas as pd
import numpy as np
from pathlib import Path
import yaml

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def generate_signals(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.DataFrame:
    df = df.copy()

    df["signal"] = np.where(df[f"sma_{fast}"] > df[f"sma_{slow}"], 1, 0)

    df["crossover"] = df["signal"].diff()
    # crossover =  1 → entrada (compra)
    # crossover = -1 → saída  (venda)

    return df

def run():
    config = load_config()
    proc_path = Path(config["data"]["processed_path"])

    equities = [
        t.replace(".", "_") for t in config["assets"]["equities"]
    ]

    for slug in equities:
        file = proc_path / f"{slug}_2020_2024.parquet"
        if not file.exists():
            print(f"  SKIP {slug} — file not found")
            continue

        df = pd.read_parquet(file).sort_index()
        df = generate_signals(df)

        buys = (df["crossover"] ==  1).sum()
        sells = (df["crossover"] == -1).sum()

        print(f"{slug:12} | buys: {buys:3}  sells: {sells:3}  "
              f"final position: {'LONG' if df['signal'].iloc[-1] == 1 else 'FLAT'}")

    print("\nDone.")

if __name__ == "__main__":
    run()