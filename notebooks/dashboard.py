import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
from pathlib import Path
import yaml
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.engine import run_backtest, load_config
from strategies.sma_crossover import generate_signals as sma
from strategies.sma_crossover_filtered import generate_signals as sma_filtered
from strategies.bollinger_bands import generate_signals as bb


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_dashboard():
    config = load_config()
    proc_path = Path(config["data"]["processed_path"])
    equities = [t.replace(".", "_") for t in config["assets"]["equities"]]
    capital = config["backtest"]["initial_capital"]

    strategies = {
        "SMA": sma,
        "SMA Filtered": sma_filtered,
        "Bollinger": bb,
    }

    # coleta métricas e equity curves
    all_metrics = []
    equity_curves = {slug: {} for slug in equities}

    for slug in equities:
        file = proc_path / f"{slug}_2020_2024.parquet"
        if not file.exists():
            continue

        df = pd.read_parquet(file).sort_index()

        for strategy_name, signal_fn in strategies.items():
            result = run_backtest(df, config, signal_fn=signal_fn)
            m = result["metrics"]
            m["ticker"] = slug
            m["strategy"] = strategy_name
            all_metrics.append(m)
            equity_curves[slug][strategy_name] = result["equity"]

    metrics_df = pd.DataFrame(all_metrics)

    # --- figura principal ---
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Retorno Total (%)",
            "Sharpe Ratio",
            "Max Drawdown (%)",
            "Probabilidade de superar B&H",
            "Equity Curves — VALE3",
            "Equity Curves — PETR4",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    colors = {"SMA": "#4C9BE8", "SMA Filtered": "#E88B4C", "Bollinger": "#2ecc71"}

    # painel 1: retorno total
    for strategy_name in strategies:
        subset = metrics_df[metrics_df["strategy"] == strategy_name]
        fig.add_trace(go.Bar(
            name=strategy_name,
            x=subset["ticker"],
            y=subset["total_return"],
            marker_color=colors[strategy_name],
            showlegend=True,
        ), row=1, col=1)

    # painel 2: sharpe
    for strategy_name in strategies:
        subset = metrics_df[metrics_df["strategy"] == strategy_name]
        fig.add_trace(go.Bar(
            name=strategy_name,
            x=subset["ticker"],
            y=subset["sharpe"],
            marker_color=colors[strategy_name],
            showlegend=False,
        ), row=1, col=2)

    # painel 3: max drawdown
    for strategy_name in strategies:
        subset = metrics_df[metrics_df["strategy"] == strategy_name]
        fig.add_trace(go.Bar(
            name=strategy_name,
            x=subset["ticker"],
            y=subset["max_drawdown"],
            marker_color=colors[strategy_name],
            showlegend=False,
        ), row=2, col=1)

    # painel 4: retorno estratégia vs B&H
    bh_df = metrics_df[metrics_df["strategy"] == "SMA"][["ticker", "bh_return"]].drop_duplicates()
    for strategy_name in strategies:
        subset = metrics_df[metrics_df["strategy"] == strategy_name]
        beat_bh = (subset["total_return"].values > bh_df["bh_return"].values).astype(int) * 100
        fig.add_trace(go.Bar(
            name=strategy_name,
            x=subset["ticker"].values,
            y=beat_bh,
            marker_color=colors[strategy_name],
            showlegend=False,
        ), row=2, col=2)

    # painel 5: equity curves VALE3
    for strategy_name, curves in equity_curves["VALE3_SA"].items():
        fig.add_trace(go.Scatter(
            name=strategy_name,
            x=curves.index,
            y=curves["equity"],
            line=dict(color=colors[strategy_name], width=1.5),
            showlegend=False,
        ), row=3, col=1)
    fig.add_trace(go.Scatter(
        name="B&H",
        x=equity_curves["VALE3_SA"]["SMA"].index,
        y=equity_curves["VALE3_SA"]["SMA"]["bh_equity"],
        line=dict(color="#aaaaaa", width=1, dash="dash"),
        showlegend=False,
    ), row=3, col=1)

    # painel 6: equity curves PETR4
    for strategy_name, curves in equity_curves["PETR4_SA"].items():
        fig.add_trace(go.Scatter(
            name=strategy_name,
            x=curves.index,
            y=curves["equity"],
            line=dict(color=colors[strategy_name], width=1.5),
            showlegend=False,
        ), row=3, col=2)
    fig.add_trace(go.Scatter(
        name="B&H",
        x=equity_curves["PETR4_SA"]["SMA"].index,
        y=equity_curves["PETR4_SA"]["SMA"]["bh_equity"],
        line=dict(color="#aaaaaa", width=1, dash="dash"),
        showlegend=False,
    ), row=3, col=2)

    fig.update_layout(
        title="Quant Lab — Strategy Dashboard (2020–2024)",
        height=1000,
        barmode="group",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    out_file = Path("notebooks/dashboard.html")
    fig.write_html(str(out_file))
    print(f"\nDashboard saved → {out_file}")


if __name__ == "__main__":
    build_dashboard()