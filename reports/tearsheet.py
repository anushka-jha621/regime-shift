"""
REGIME-SHIFT | Phase 5: Final Tearsheet
=========================================
This pulls everything together into one clean report.
Nothing new computationally — just making it presentable.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import warnings
import os
warnings.filterwarnings("ignore")


def load_all_data():
    """Load everything we generated in phases 1-4."""
    df         = pd.read_csv("data/master_data.csv",      index_col=0, parse_dates=True)
    prices     = pd.read_csv("data/prices.csv",           index_col=0, parse_dates=True)
    regimes    = pd.read_csv("data/regimes.csv",          index_col=0, parse_dates=True).squeeze()
    weights    = pd.read_csv("data/weights.csv",          index_col=0, parse_dates=True)
    backtest   = pd.read_csv("data/backtest_results.csv", index_col=0, parse_dates=True)
    return df, prices, regimes, weights, backtest


def compute_metrics(returns_series, rf=0.04, label="Strategy"):
    """Standard performance metrics — same as Phase 4."""
    r = returns_series.dropna()
    n_years = len(r) / 252

    cumulative   = (1 + r).cumprod()
    total_ret    = cumulative.iloc[-1] - 1
    annual_ret   = (1 + total_ret) ** (1 / n_years) - 1
    annual_vol   = r.std() * np.sqrt(252)
    sharpe       = (annual_ret - rf) / annual_vol if annual_vol > 0 else 0
    down_vol     = r[r < 0].std() * np.sqrt(252)
    sortino      = (annual_ret - rf) / down_vol if down_vol > 0 else 0
    rolling_max  = cumulative.cummax()
    max_dd       = ((cumulative - rolling_max) / rolling_max).min()
    calmar       = annual_ret / abs(max_dd) if max_dd != 0 else 0
    win_rate     = (r > 0).mean()

    return dict(label=label, total_ret=total_ret, annual_ret=annual_ret,
                annual_vol=annual_vol, sharpe=sharpe, sortino=sortino,
                max_dd=max_dd, calmar=calmar, win_rate=win_rate,
                cumulative=cumulative, returns=r, n_years=n_years)


def build_tearsheet(df, prices, regimes, weights, backtest,
                    save_path="reports/05_final_tearsheet.png"):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # ── Compute returns for all three strategies ───────────────────────────
    strat_rets = backtest["strategy_ret"]
    start      = strat_rets.index[0]
    df_bt      = df.loc[start:]

    ret_6040 = (0.60 * df_bt["SPY_ret"] + 0.40 * df_bt["TLT_ret"]).rename("60/40")
    ret_ew   = ((df_bt["SPY_ret"] + df_bt["TLT_ret"] + df_bt["GLD_ret"]) / 3).rename("EW")

    m_s    = compute_metrics(strat_rets,                                   label="Regime-Shift")
    m_6040 = compute_metrics(ret_6040.reindex(strat_rets.index).fillna(0), label="60/40")
    m_ew   = compute_metrics(ret_ew.reindex(strat_rets.index).fillna(0),   label="Equal Weight")

    C_STRAT  = "#2563eb"
    C_6040   = "#6b7280"
    C_EW     = "#d97706"
    C_BULL   = "#16a34a"
    C_BEAR   = "#d97706"
    C_CRISIS = "#dc2626"

    reg_colors = {0: C_BULL, 1: C_BEAR, 2: C_CRISIS}
    reg_alpha  = {0: 0.08, 1: 0.12, 2: 0.20}

    def shade(ax, reg_series):
        curr    = reg_series.iloc[0]
        start_d = reg_series.index[0]
        for date, r in reg_series.items():
            if r != curr:
                ax.axvspan(start_d, date, color=reg_colors[curr],
                           alpha=reg_alpha[curr], zorder=0)
                curr, start_d = r, date
        ax.axvspan(start_d, reg_series.index[-1], color=reg_colors[curr],
                   alpha=reg_alpha[curr], zorder=0)

    regime_bt = backtest["regime"].reindex(strat_rets.index)

    fig = plt.figure(figsize=(20, 22))
    fig.patch.set_facecolor("#0f172a")

    gs = gridspec.GridSpec(
        4, 3, figure=fig,
        hspace=0.45, wspace=0.35,
        top=0.91, bottom=0.05, left=0.07, right=0.97
    )

    fig.text(0.5, 0.955, "REGIME-SHIFT", fontsize=28, fontweight="bold",
             ha="center", color="white", fontfamily="monospace")
    fig.text(0.5, 0.938,
             "Macro-Aware Tactical Asset Allocation Engine  |  Walk-Forward Backtest",
             fontsize=11, ha="center", color="#94a3b8")

    legend_patches = [
        mpatches.Patch(color=C_BULL,   alpha=0.7, label="Bull regime"),
        mpatches.Patch(color=C_BEAR,   alpha=0.7, label="Bear regime"),
        mpatches.Patch(color=C_CRISIS, alpha=0.7, label="Crisis regime"),
    ]
    fig.legend(handles=legend_patches, loc="upper right",
               bbox_to_anchor=(0.97, 0.955), fontsize=9,
               framealpha=0.2, labelcolor="white",
               facecolor="#1e293b", edgecolor="#334155")

    def style_ax(ax, title="", ylabel="", xlabel="Date"):
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8", labelsize=8)
        ax.xaxis.label.set_color("#94a3b8")
        ax.yaxis.label.set_color("#94a3b8")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")
        ax.grid(True, alpha=0.15, color="#475569")
        if title:  ax.set_title(title, color="#e2e8f0", fontsize=10, pad=6)
        if ylabel: ax.set_ylabel(ylabel, color="#94a3b8", fontsize=8)
        if xlabel: ax.set_xlabel(xlabel, color="#94a3b8", fontsize=8)

    # ── Panel 1: Equity curves ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(m_s["cumulative"],    color=C_STRAT, lw=2.2, label="Regime-Shift", zorder=4)
    ax1.plot(m_6040["cumulative"], color=C_6040,  lw=1.5, linestyle="--", label="60/40", zorder=3)
    ax1.plot(m_ew["cumulative"],   color=C_EW,    lw=1.5, linestyle=":",  label="Equal Weight", zorder=3)
    shade(ax1, regime_bt)
    ax1.set_yscale("log")
    style_ax(ax1, title="Equity Curves — Growth of $1  (log scale)", ylabel="Portfolio Value ($)")
    ax1.legend(fontsize=9, framealpha=0.3, facecolor="#1e293b",
               labelcolor="white", edgecolor="#334155")
    for m, color in [(m_s, C_STRAT), (m_6040, C_6040), (m_ew, C_EW)]:
        fv = m["cumulative"].iloc[-1]
        ax1.annotate(f"${fv:.2f}",
                     xy=(m["cumulative"].index[-1], fv),
                     xytext=(8, 0), textcoords="offset points",
                     color=color, fontsize=8, va="center", fontweight="bold")

    # ── Panel 2: Drawdown ──────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    def drawdown(cum): return (cum - cum.cummax()) / cum.cummax()
    ax2.fill_between(m_s["cumulative"].index,
                     drawdown(m_s["cumulative"]).values, 0,
                     color=C_STRAT, alpha=0.35, label="Strategy")
    ax2.plot(drawdown(m_6040["cumulative"]), color=C_6040,
             lw=1.2, linestyle="--", label="60/40", alpha=0.85)
    ax2.plot(drawdown(m_ew["cumulative"]), color=C_EW,
             lw=1.2, linestyle=":", label="Equal Weight", alpha=0.85)
    shade(ax2, regime_bt)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    style_ax(ax2, title="Drawdown — how far below peak at each point in time", ylabel="Drawdown")
    ax2.legend(fontsize=9, framealpha=0.3, facecolor="#1e293b",
               labelcolor="white", edgecolor="#334155", loc="lower left")

    # ── Panel 3: Rolling Sharpe ────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, :2])
    def roll_sharpe(r):
        return r.rolling(252).mean() / r.rolling(252).std() * np.sqrt(252)
    ax3.plot(roll_sharpe(strat_rets), color=C_STRAT, lw=1.5, label="Strategy")
    ax3.plot(roll_sharpe(ret_6040.reindex(strat_rets.index).fillna(0)),
             color=C_6040, lw=1.2, linestyle="--", label="60/40", alpha=0.8)
    ax3.axhline(0, color="#475569", lw=0.8)
    ax3.axhline(1, color=C_BULL, lw=0.8, linestyle=":", alpha=0.6, label="Sharpe = 1")
    shade(ax3, regime_bt)
    style_ax(ax3, title="Rolling 1-Year Sharpe Ratio", ylabel="Sharpe")
    ax3.legend(fontsize=9, framealpha=0.3, facecolor="#1e293b",
               labelcolor="white", edgecolor="#334155")

    # ── Panel 4: Stats table ───────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 2])
    ax4.axis("off")
    ax4.set_facecolor("#1e293b")
    rows = [
        ("Total Return",  f"{m_s['total_ret']:.1%}",  f"{m_6040['total_ret']:.1%}",  f"{m_ew['total_ret']:.1%}"),
        ("Annual Return", f"{m_s['annual_ret']:.1%}",  f"{m_6040['annual_ret']:.1%}", f"{m_ew['annual_ret']:.1%}"),
        ("Volatility",    f"{m_s['annual_vol']:.1%}",  f"{m_6040['annual_vol']:.1%}", f"{m_ew['annual_vol']:.1%}"),
        ("Sharpe",        f"{m_s['sharpe']:.2f}",      f"{m_6040['sharpe']:.2f}",     f"{m_ew['sharpe']:.2f}"),
        ("Sortino",       f"{m_s['sortino']:.2f}",     f"{m_6040['sortino']:.2f}",    f"{m_ew['sortino']:.2f}"),
        ("Max Drawdown",  f"{m_s['max_dd']:.1%}",      f"{m_6040['max_dd']:.1%}",     f"{m_ew['max_dd']:.1%}"),
        ("Calmar",        f"{m_s['calmar']:.2f}",      f"{m_6040['calmar']:.2f}",     f"{m_ew['calmar']:.2f}"),
        ("Win Rate",      f"{m_s['win_rate']:.1%}",    f"{m_6040['win_rate']:.1%}",   f"{m_ew['win_rate']:.1%}"),
    ]
    table = ax4.table(
        cellText=rows,
        colLabels=["Metric", "Regime\nShift", "60/40", "Eq Wt"],
        loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.55)
    for (row, col), cell in table.get_celld().items():
        cell.set_facecolor("#0f172a" if row == 0 else ("#1e293b" if row % 2 == 0 else "#162032"))
        cell.set_text_props(color="white" if row == 0 else "#e2e8f0")
        cell.set_edgecolor("#334155")
        if col == 1 and row > 0:
            cell.set_facecolor("#1e3a5f")
    ax4.set_title("Performance Summary", color="#e2e8f0", fontsize=10, pad=10)

    # ── Panel 5: Weights stacked area ─────────────────────────────────────
    ax5 = fig.add_subplot(gs[3, :2])
    wt_bt = weights.reindex(strat_rets.index).ffill()
    ax5.stackplot(
        wt_bt.index,
        wt_bt["SPY_w"].fillna(1/3),
        wt_bt["TLT_w"].fillna(1/3),
        wt_bt["GLD_w"].fillna(1/3),
        labels=["SPY (stocks)", "TLT (bonds)", "GLD (gold)"],
        colors=["#3b82f6", "#10b981", "#f59e0b"],
        alpha=0.80
    )
    ax5.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax5.set_ylim(0, 1)
    style_ax(ax5, title="Portfolio Weights Over Time", ylabel="Weight")
    ax5.legend(fontsize=9, loc="upper left", framealpha=0.3,
               facecolor="#1e293b", labelcolor="white", edgecolor="#334155")

    # ── Panel 6: Regime pie ────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[3, 2])
    ax6.set_facecolor("#1e293b")
    regime_counts   = regime_bt.value_counts().sort_index()
    labels_pie      = [{0:"Bull",1:"Bear",2:"Crisis"}.get(i,str(i)) for i in regime_counts.index]
    colors_pie      = [reg_colors.get(i,"#888") for i in regime_counts.index]
    pcts            = regime_counts.values / regime_counts.values.sum()
    wedges, texts, _ = ax6.pie(
        regime_counts.values,
        labels=[f"{l}\n{p:.0%}" for l, p in zip(labels_pie, pcts)],
        colors=colors_pie, autopct="", startangle=90,
        wedgeprops=dict(edgecolor="#0f172a", linewidth=2),
    )
    for t in texts:
        t.set_color("#e2e8f0")
        t.set_fontsize(9)
    ax6.set_title("Time in Each Regime", color="#e2e8f0", fontsize=10, pad=10)

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"   ✓ Tearsheet saved → {save_path}")


if __name__ == "__main__":
    print("\n" + "="*55)
    print("REGIME-SHIFT | Phase 5: Final Tearsheet")
    print("="*55 + "\n")

    print("📂  Loading all data...")
    df, prices, regimes, weights, backtest = load_all_data()

    print("\n📊  Building tearsheet...")
    build_tearsheet(df, prices, regimes, weights, backtest)

    print("\n✅  Done! Your final tearsheet is at reports/05_final_tearsheet.png")
    print("    This is your CV-ready output. Put it in your README on GitHub.")