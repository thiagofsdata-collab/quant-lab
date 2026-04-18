import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.engine import run_backtest, load_config
from strategies.sma_crossover import generate_signals as sma
from strategies.bollinger_bands import generate_signals as bb


def run_simulation(returns: pd.Series, capital: float, n_simulations: int = 1000, n_days: int = 252) -> np.ndarray:
    mu = returns.mean()
    sigma = returns.std()

    simulations = np.zeros((n_days, n_simulations))
    for i in range(n_simulations):
        daily_returns = np.random.normal(mu, sigma, n_days)
        simulations[:, i] = capital * np.cumprod(1 + daily_returns)

    return simulations


def plot_simulation(simulations: np.ndarray, capital: float, slug: str, strategy_name: str, out_dir: Path):
    final_values = simulations[-1, :]
    n_sims = simulations.shape[1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{slug} — Monte Carlo | {strategy_name} ({n_sims} simulações, 1 ano)", fontweight="bold")

    sample_idx = np.random.choice(n_sims, size=200, replace=False)
    for i in sample_idx:
        ax1.plot(simulations[:, i], alpha=0.15, color="#4C9BE8", linewidth=0.6)

    ax1.plot(simulations.mean(axis=1), color="#e74c3c", linewidth=2, label="Média", zorder=5)
    ax1.plot(np.percentile(simulations, 5,  axis=1), color="#e74c3c", linewidth=1,
             linestyle="--", label="VaR 5%", zorder=5)
    ax1.plot(np.percentile(simulations, 95, axis=1), color="#2ecc71", linewidth=1,
             linestyle="--", label="P95", zorder=5)
    ax1.axhline(capital, color="#aaaaaa", linestyle=":", linewidth=1, label="Capital inicial")
    ax1.set_ylabel("Capital (R$)")
    ax1.set_xlabel("Dias")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.hist(final_values, bins=60, color="#4C9BE8", alpha=0.8, edgecolor="white")
    ax2.axvline(capital, color="#aaaaaa", linestyle=":", linewidth=1.5, label="Capital inicial")
    ax2.axvline(np.percentile(final_values, 5),  color="#e74c3c", linewidth=2, label=f"VaR 5%: R${np.percentile(final_values, 5):,.0f}")
    ax2.axvline(np.percentile(final_values, 95), color="#2ecc71", linewidth=2, label=f"P95: R${np.percentile(final_values, 95):,.0f}")
    ax2.axvline(np.mean(final_values), color="#e67e22", linewidth=2, label=f"Média: R${np.mean(final_values):,.0f}")
    ax2.set_xlabel("Capital final (R$)")
    ax2.set_ylabel("Frequência")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    out_file = out_dir / f"{slug}_{strategy_name}_montecarlo.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out_file}")


def run():
    config = load_config()
    proc_path = Path(config["data"]["processed_path"])
    out_dir = Path("notebooks")
    equities = [t.replace(".", "_") for t in config["assets"]["equities"]]
    capital = config["backtest"]["initial_capital"]

    strategies = {
        "SMA": sma,
        "Bollinger": bb,
    }

    print(f"\n{'Ticker':<12} {'Strategy':<12} {'Média':>12} {'VaR 5%':>12} {'P95':>12} {'Prob Lucro':>12}")
    print("-" * 68)

    for slug in equities:
        file = proc_path / f"{slug}_2020_2024.parquet"
        if not file.exists():
            continue

        df = pd.read_parquet(file).sort_index()

        for strategy_name, signal_fn in strategies.items():
            result = run_backtest(df, config, signal_fn=signal_fn)
            strategy_returns = result["equity"]["equity"].pct_change().dropna()

            simulations = run_simulation(strategy_returns, capital)
            final_values = simulations[-1, :]

            mean_final = np.mean(final_values)
            var_5 = np.percentile(final_values, 5)
            p95 = np.percentile(final_values, 95)
            prob_profit = (final_values > capital).mean() * 100

            print(f"{slug:<12} {strategy_name:<12} {mean_final:>11.0f}  "
                  f"{var_5:>11.0f}  {p95:>11.0f}  {prob_profit:>10.1f}%")

            plot_simulation(simulations, capital, slug, strategy_name, out_dir)

    print("\nDone. Charts saved in notebooks/")


if __name__ == "__main__":
    run()