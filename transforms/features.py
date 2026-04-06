import duckdb
import pandas as pd
import yaml
from pathlib import Path

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def compute_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    t = config["transforms"]

    df["return_1d"] = df["close"].pct_change(t["returns_window"])

    df["volatility_21d"] = df["return_1d"].rolling(t["volatility_window"]).std()

    for w in t["sma_windows"]:
        df[f"sma_{w}"] = df["close"].rolling(w).mean()

    df["rsi_14"] = compute_rsi(df["close"], t["rsi_window"])

    return df

def compute_rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def run():
    config = load_config()
    raw_path = Path(config["data"]["raw_path"])
    out_path = Path(config["data"]["processed_path"])

    parquet_files = list(raw_path.glob("*.parquet"))
    print(f"\nTransforming {len(parquet_files)} files...\n")

    for file in parquet_files:
        df = pd.read_parquet(file)
        df = df.sort_index()

        df = compute_features(df, config)
        df = df.dropna(subset=["return_1d"])

        out_file = out_path / file.name
        df.to_parquet(out_file, index=True)
        print(f"saved → {out_file} ({len(df)} rows, {len(df.columns)} cols)")

    print("\nDone.")

if __name__ == "__main__":
    run()