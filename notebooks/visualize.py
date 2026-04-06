import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import yaml
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategies.sma_crossover import generate_signals
from backtest.engine import run_backtest

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def plot_ticker(slug: str, df: pd.DataFrame, result: dict, config: dict):
    equity = result["equity"]
    drawdown = result["drawdown"]
    m = result["metrics"]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"{slug} — SMA Crossover Backtest (2020–2024)",
                 fontsize=14, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1.2, 1], hspace=0.4)

    # --- painel 1: preço + médias móveis ---
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df.index, df["close"],  color="#aaaaaa", linewidth=1,   label="Close",  zorder=1)
    ax1.plot(df.index, df["sma_20"], color="#4C9BE8", linewidth=1.2, label="SMA 20", zorder=2)
    ax1.plot(df.index, df["sma_50"], color="#E88B4C", linewidth=1.2, label="SMA 50", zorder=2)

    # sinais de compra e venda
    df_sig = generate_signals(df)
    buys = df_sig[df_sig["crossover"] ==  1]
    sells = df_sig[df_sig["crossover"] == -1]
    ax1.scatter(buys.index, buys["close"],  marker="^", color="#2ecc71", s=80, zorder=5, label="Buy")
    ax1.scatter(sells.index, sells["close"], marker="v", color="#e74c3c", s=80, zorder=5, label="Sell")

    ax1.set_ylabel("Preço (R$)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)

    # --- painel 2: equity curve ---
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(equity.index, equity["equity"],    color="#4C9BE8", linewidth=1.5, label="Estratégia")
    ax2.plot(equity.index, equity["bh_equity"], color="#aaaaaa", linewidth=1,   label="Buy & Hold", linestyle="--")
    ax2.set_ylabel("Capital (R$)")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(alpha=0.3)

    # --- painel 3: drawdown ---
    ax3 = fig.add_subplot(gs[2])
    ax3.fill_between(drawdown.index, drawdown * 100, 0, color="#e74c3c", alpha=0.4)
    ax3.plot(drawdown.index, drawdown * 100, color="#e74c3c", linewidth=0.8)
    ax3.set_ylabel("Drawdown (%)")
    ax3.set_xlabel("Data")
    ax3.grid(alpha=0.3)

    # --- caixa de métricas ---
    metrics_text = (
        f"Return: {m['total_return']}%   |   "
        f"B&H: {m['bh_return']}%   |   "
        f"Annual: {m['annual_return']}%   |   "
        f"Sharpe: {m['sharpe']}   |   "
        f"MaxDD: {m['max_drawdown']}%   |   "
        f"Trades: {m['trades']}"
    )
    fig.text(0.5, 0.01, metrics_text, ha="center", fontsize=8,
             bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))

    out_dir = Path("notebooks")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"{slug}_backtest.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out_file}")

def run():
    config = load_config()
    proc_path = Path(config["data"]["processed_path"])
    equities = [t.replace(".", "_") for t in config["assets"]["equities"]]

    print(f"\nGenerating charts for {len(equities)} tickers...\n")

    for slug in equities:
        file = proc_path / f"{slug}_2020_2024.parquet"
        if not file.exists():
            print(f"SKIP {slug}")
            continue

        df = pd.read_parquet(file).sort_index()
        result = run_backtest(df, config)
        plot_ticker(slug, df, result, config)

    print("\nDone. Charts saved in notebooks/")

if __name__ == "__main__":
    run()