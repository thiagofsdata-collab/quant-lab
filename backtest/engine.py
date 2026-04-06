import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from strategies.sma_crossover import generate_signals

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_backtest(df: pd.DataFrame, config: dict) -> dict:
    capital = config["backtest"]["initial_capital"]
    commission = config["backtest"]["commission"]
    slippage = config["backtest"]["slippage"]

    df = df.copy()
    df = generate_signals(df)

    df["market_return"] = df["close"].pct_change()
    df["strategy_return"] = df["market_return"] * df["signal"].shift(1)

    cost = commission + slippage
    df.loc[df["crossover"] != 0, "strategy_return"] -= cost

    df["equity"] = capital * (1 + df["strategy_return"]).cumprod()
    df["equity"] = df["equity"].fillna(capital)

    df["bh_equity"] = capital * (1 + df["market_return"]).cumprod()
    df["bh_equity"] = df["bh_equity"].fillna(capital)

    total_return  = (df["equity"].iloc[-1] / capital) - 1
    bh_return = (df["bh_equity"].iloc[-1] / capital) - 1

    annual_return = (1 + total_return) ** (252 / len(df)) - 1
    volatility = df["strategy_return"].std() * np.sqrt(252)
    sharpe = annual_return / volatility if volatility != 0 else 0

    roll_max = df["equity"].cummax()
    drawdown = (df["equity"] - roll_max) / roll_max
    max_drawdown = drawdown.min()

    trades = int((df["crossover"] == 1).sum())

    return {
        "equity": df[["equity", "bh_equity"]],
        "drawdown": drawdown,
        "metrics": {
            "total_return": round(total_return * 100, 2),
            "bh_return": round(bh_return * 100, 2),
            "annual_return": round(annual_return * 100, 2),
            "volatility": round(volatility * 100, 2),
            "sharpe": round(sharpe, 3),
            "max_drawdown": round(max_drawdown * 100, 2),
            "trades": trades,
        }
    }

def run():
    config = load_config()
    proc_path = Path(config["data"]["processed_path"])
    equities = [t.replace(".", "_") for t in config["assets"]["equities"]]

    print(f"\n{'Ticker':<12} {'Return':>8} {'B&H':>8} {'Annual':>8} "
          f"{'Sharpe':>8} {'MaxDD':>8} {'Trades':>7}")
    print("-" * 65)

    for slug in equities:
        file = proc_path / f"{slug}_2020_2024.parquet"
        if not file.exists():
            continue

        df = pd.read_parquet(file).sort_index()
        result = run_backtest(df, config)
        m = result["metrics"]

        print(f"{slug:<12} {m['total_return']:>7}%  {m['bh_return']:>7}%  "
              f"{m['annual_return']:>7}%  {m['sharpe']:>7}  "
              f"{m['max_drawdown']:>7}%  {m['trades']:>6}")

    print("\nDone.")

if __name__ == "__main__":
    run()