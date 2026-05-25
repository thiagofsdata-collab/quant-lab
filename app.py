import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest.engine import run_backtest, load_config
from strategies.sma_crossover import generate_signals as sma
from strategies.sma_crossover_filtered import generate_signals as sma_filtered
from strategies.bollinger_bands import generate_signals as bb
from strategies.zscore_mean_reversion import generate_signals as zscore


# --- config ---
st.set_page_config(
    page_title="Quant Lab",
    page_icon="📊",
    layout="wide"
)

STRATEGIES = {
    "SMA Crossover": sma,
    "SMA Filtered": sma_filtered,
    "Bollinger Bands": bb,
    "Z-Score Mean Reversion": zscore,
}

TICKERS = {
    "PETR4": "PETR4_SA",
    "VALE3": "VALE3_SA",
    "ITUB4": "ITUB4_SA",
    "BBDC4": "BBDC4_SA",
    "WEGE3": "WEGE3_SA",
}


def load_data(slug: str, start: str, end: str) -> pd.DataFrame:
    config = load_config()
    proc_path = Path(config["data"]["processed_path"])
    file = proc_path / f"{slug}_2020_2024.parquet"
    df = pd.read_parquet(file).sort_index()
    df.index = pd.to_datetime(df.index)
    return df.loc[start:end]


def plot_equity(result: dict, ticker: str, strategy_name: str) -> go.Figure:
    equity = result["equity"]
    drawdown = result["drawdown"]

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("Equity Curve", "Drawdown")
    )

    fig.add_trace(go.Scatter(
        x=equity.index, y=equity["equity"],
        name=strategy_name,
        line=dict(color="#4C9BE8", width=2)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=equity.index, y=equity["bh_equity"],
        name="Buy & Hold",
        line=dict(color="#aaaaaa", width=1.5, dash="dash")
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown * 100,
        name="Drawdown",
        fill="tozeroy",
        line=dict(color="#e74c3c", width=1),
        fillcolor="rgba(231,76,60,0.2)"
    ), row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=500,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    fig.update_yaxes(title_text="Capital (R$)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

    return fig


def plot_monte_carlo(returns: pd.Series, capital: float, n_sims: int = 500) -> go.Figure:
    mu = returns.mean()
    sigma = returns.std()
    n_days = 252

    simulations = np.zeros((n_days, n_sims))
    for i in range(n_sims):
        daily = np.random.normal(mu, sigma, n_days)
        simulations[:, i] = capital * np.cumprod(1 + daily)

    final = simulations[-1, :]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Trajetórias", "Distribuição Final"))

    sample = np.random.choice(n_sims, size=150, replace=False)
    for i in sample:
        fig.add_trace(go.Scatter(
            y=simulations[:, i], mode="lines",
            line=dict(color="#4C9BE8", width=0.5),
            opacity=0.15, showlegend=False
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        y=simulations.mean(axis=1), mode="lines",
        name="Média", line=dict(color="#e74c3c", width=2)
    ), row=1, col=1)

    fig.add_trace(go.Histogram(
        x=final, nbinsx=50,
        marker_color="#4C9BE8", opacity=0.8,
        name="Distribuição", showlegend=False
    ), row=1, col=2)

    for pct, color, label in [
        (5,  "#e74c3c", f"VaR 5%: R${np.percentile(final,5):,.0f}"),
        (50, "#f39c12", f"Mediana: R${np.percentile(final,50):,.0f}"),
        (95, "#2ecc71", f"P95: R${np.percentile(final,95):,.0f}"),
    ]:
        fig.add_vline(
            x=np.percentile(final, pct), row=1, col=2,
            line_color=color, line_width=2,
            annotation_text=label, annotation_position="top"
        )

    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig, final


# --- sidebar ---
st.sidebar.image("https://img.icons8.com/fluency/48/combo-chart.png", width=48)
st.sidebar.title("Quant Lab")
st.sidebar.markdown("---")

ticker = st.sidebar.selectbox("Ativo", list(TICKERS.keys()))
strategy_name = st.sidebar.selectbox("Estratégia", list(STRATEGIES.keys()))
start_date = st.sidebar.date_input("Data inicial", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("Data final", value=pd.to_datetime("2024-12-31"))
run_mc = st.sidebar.checkbox("Rodar Monte Carlo", value=False)
run_button = st.sidebar.button("Rodar Backtest", use_container_width=True)

# --- main ---
st.title("📊 Quant Lab — Strategy Dashboard")

if run_button:
    config = load_config()
    slug = TICKERS[ticker]
    signal_fn = STRATEGIES[strategy_name]

    with st.spinner("Rodando backtest..."):
        df = load_data(slug, str(start_date), str(end_date))
        result = run_backtest(df, config, signal_fn=signal_fn)
        m = result["metrics"]

    # métricas
    st.subheader(f"{ticker} — {strategy_name}")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Retorno Total", f"{m['total_return']}%", f"{m['total_return'] - m['bh_return']:.1f}% vs B&H")
    col2.metric("Retorno Anual", f"{m['annual_return']}%")
    col3.metric("Sharpe Ratio", m['sharpe'])
    col4.metric("Max Drawdown", f"{m['max_drawdown']}%")
    col5.metric("Trades", m['trades'])

    st.markdown("---")

    # equity curve
    st.plotly_chart(plot_equity(result, ticker, strategy_name), use_container_width=True)

    # monte carlo
    if run_mc:
        st.subheader("Monte Carlo — 500 simulações (1 ano)")
        strategy_returns = result["equity"]["equity"].pct_change().dropna()
        fig_mc, final_values = plot_monte_carlo(strategy_returns, config["backtest"]["initial_capital"])
        st.plotly_chart(fig_mc, use_container_width=True)

        prob = (final_values > config["backtest"]["initial_capital"]).mean() * 100
        st.info(f"Probabilidade de lucro em 1 ano: **{prob:.1f}%**")

else:
    st.info("Configure os parâmetros na sidebar e clique em **Rodar Backtest**.")