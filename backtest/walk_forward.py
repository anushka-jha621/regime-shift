"""
Walk-Forward Backtester
========================
Simulates running this strategy in real life with no look-ahead bias.

The core loop:
  1. Train HMM on all data before month t
  2. Detect today's regime
  3. Optimize weights for that regime
  4. Record the actual return
  5. Roll forward one month and repeat
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
import os
warnings.filterwarnings("ignore")

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import cvxpy as cp


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def compute_metrics(returns_series, rf_rate=0.04, label="Strategy"):
    """
    Standard performance metrics.

    Sharpe  = (annual_return - risk_free) / annual_vol
              how much return per unit of risk. > 1 is good.

    Sortino = same but only penalizes downside vol, not upside swings.
              better reflects what investors actually care about.

    Max Drawdown = worst peak-to-trough loss.
                   a -50% drawdown needs +100% to recover.

    Calmar  = annual_return / |max_drawdown|
              preferred by risk managers over Sharpe.
    """
    trading_days  = 252
    r             = returns_series.dropna()
    n_years       = len(r) / trading_days
    cumulative    = (1 + r).cumprod()
    total_return  = cumulative.iloc[-1] - 1
    annual_return = (1 + total_return) ** (1 / n_years) - 1
    annual_vol    = r.std() * np.sqrt(trading_days)
    sharpe        = (annual_return - rf_rate) / annual_vol if annual_vol > 0 else 0
    downside_vol  = r[r < 0].std() * np.sqrt(trading_days)
    sortino       = (annual_return - rf_rate) / downside_vol if downside_vol > 0 else 0
    rolling_max   = cumulative.cummax()
    max_drawdown  = ((cumulative - rolling_max) / rolling_max).min()
    calmar        = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    win_rate      = (r > 0).mean()

    return {
        "label":        label,
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_vol":   annual_vol,
        "sharpe":       sharpe,
        "sortino":      sortino,
        "max_drawdown": max_drawdown,
        "calmar":       calmar,
        "win_rate":     win_rate,
        "n_years":      n_years
    }


def print_metrics_table(metrics_list):
    """Side-by-side performance comparison."""
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)
    labels = [m["label"] for m in metrics_list]
    print(f"{'Metric':<25}" + "".join(f"{l:>18}" for l in labels))
    print("-"*70)
    rows = [
        ("Total Return",      "total_return",  "{:>17.1%}"),
        ("Annual Return",     "annual_return", "{:>17.1%}"),
        ("Annual Volatility", "annual_vol",    "{:>17.1%}"),
        ("Sharpe Ratio",      "sharpe",        "{:>17.2f}"),
        ("Sortino Ratio",     "sortino",       "{:>17.2f}"),
        ("Max Drawdown",      "max_drawdown",  "{:>17.1%}"),
        ("Calmar Ratio",      "calmar",        "{:>17.2f}"),
        ("Win Rate",          "win_rate",      "{:>17.1%}"),
    ]
    for row_label, key, fmt in rows:
        line = f"{row_label:<25}"
        for m in metrics_list:
            line += fmt.format(m[key])
        print(line)
    print("="*70)


# ============================================================
# BENCHMARKS
# ============================================================

def compute_benchmark_returns(returns):
    """
    Two passive benchmarks to beat.

    60/40 — the industry standard. Works great in normal times,
    falls apart when stocks and bonds crash together (e.g. 2022).

    Equal weight — surprisingly hard to beat. DeMiguel (2009) showed
    1/N often beats complex optimizers out-of-sample because
    estimation errors in mu and Sigma compound over time.
    """
    ret_6040 = 0.60 * returns["SPY_ret"] + 0.40 * returns["TLT_ret"]
    ret_ew   = (returns["SPY_ret"] + returns["TLT_ret"] + returns["GLD_ret"]) / 3
    return ret_6040.rename("60/40"), ret_ew.rename("Equal Weight")


# ============================================================
# WALK-FORWARD ENGINE
# ============================================================

def build_features(df):
    """Same features as the regime detector — has to be identical."""
    features = pd.DataFrame(index=df.index)
    features["spy_ret"]     = df["SPY_ret"]
    features["spy_vol_20d"] = df["SPY_ret"].rolling(20).std() * np.sqrt(252)
    features["vix"]         = df["VIX"]
    features["tlt_ret"]     = df["TLT_ret"]
    features["gld_ret"]     = df["GLD_ret"]
    return features.dropna()


def train_hmm_on_window(features_window, n_states=3):
    """
    Trains a fresh HMM on everything up to today, returns
    the regime label for the last day in the window.

    States come out as arbitrary numbers so we sort by avg SPY
    return to get consistent labels: 0=Bull, 1=Bear, 2=Crisis.
    """
    scaler     = StandardScaler()
    scaled     = scaler.fit_transform(features_window.values)
    model      = GaussianHMM(n_components=n_states, covariance_type="full",
                              n_iter=200, random_state=42)
    model.fit(scaled)
    raw_states = model.predict(scaled)

    spy_ret       = features_window["spy_ret"].values
    state_returns = {s: spy_ret[raw_states == s].mean() for s in range(n_states)}
    sorted_states = sorted(state_returns, key=state_returns.get, reverse=True)
    label_map     = {sorted_states[0]: 0, sorted_states[1]: 1, sorted_states[2]: 2}

    return label_map[raw_states[-1]]


def optimize_for_regime(regime, returns_window, prev_weights):
    """
    Bull   → max Sharpe  (go offensive, capture upside)
    Bear   → min variance with light turnover penalty
    Crisis → min variance with heavy turnover penalty
             last thing you want in a crash is to be constantly trading
    """
    assets = ["SPY_ret", "TLT_ret", "GLD_ret"]
    r      = returns_window[assets]
    mu     = r.mean().values * 252
    Sigma  = r.cov().values * 252 + 1e-6 * np.eye(3)
    n      = 3

    def min_variance(penalty=0.001):
        w    = cp.Variable(n)
        cost = cp.quad_form(w, Sigma)
        if prev_weights is not None:
            cost += penalty * cp.norm1(w - prev_weights)
        prob = cp.Problem(cp.Minimize(cost),
                          [cp.sum(w) == 1, w >= 0, w <= 0.75])
        prob.solve(solver=cp.CLARABEL)
        if w.value is None:
            return np.ones(n) / n
        wts = np.maximum(w.value, 0)
        return wts / wts.sum()

    def max_sharpe():
        excess = mu - 0.04
        if np.all(excess <= 0):
            return min_variance()
        w    = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)),
                          [excess @ w == 1, w >= 0, w <= 0.90])
        prob.solve(solver=cp.CLARABEL)
        if w.value is None:
            return np.ones(n) / n
        wts = np.maximum(w.value, 0)
        return wts / wts.sum()

    if regime == 0:   return max_sharpe()
    elif regime == 1: return min_variance(penalty=0.001)
    else:             return min_variance(penalty=0.005)


def run_walk_forward(df, train_years=2, retrain_months=3):
    """
    Main loop. Retrains HMM every quarter on all past data,
    holds those weights until the next retraining date.

    Quarterly retraining is the sweet spot — frequent enough to
    catch real regime changes, slow enough to avoid overfitting noise.
    """
    features     = build_features(df)
    common       = df.index.intersection(features.index)
    df           = df.loc[common]
    features     = features.loc[common]
    train_days   = train_years * 252
    retrain_days = retrain_months * 21

    results         = []
    prev_weights    = np.array([1/3, 1/3, 1/3])
    current_regime  = 1  # start conservative
    current_weights = prev_weights.copy()

    for i in range(train_days, len(df)):
        date = df.index[i]

        if (i - train_days) % retrain_days == 0:
            try:
                proposed = train_hmm_on_window(features.iloc[:i])

                # Only switch regime if proposed twice in a row —
                # prevents thrashing on noisy regime boundaries
                confirmed_count = getattr(run_walk_forward, '_confirm', 0) + 1
                if proposed != current_regime and confirmed_count < 2:
                    run_walk_forward._confirm = confirmed_count
                    proposed = current_regime
                else:
                    run_walk_forward._confirm = 0

                current_regime  = proposed
                current_weights = optimize_for_regime(current_regime, df.iloc[:i], prev_weights)
                prev_weights    = current_weights.copy()

            except Exception:
                pass

        today_returns    = df.iloc[i][["SPY_ret", "TLT_ret", "GLD_ret"]].values
        portfolio_return = np.dot(current_weights, today_returns)

        results.append({
            "date":         date,
            "strategy_ret": portfolio_return,
            "regime":       current_regime,
            "SPY_w":        current_weights[0],
            "TLT_w":        current_weights[1],
            "GLD_w":        current_weights[2],
        })

    return pd.DataFrame(results).set_index("date")


# ============================================================
# CHARTS
# ============================================================

def plot_equity_curves(strategy_rets, ret_6040, ret_ew, results_df,
                       save_path="reports/04_backtest.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    common      = strategy_rets.index
    r6040       = ret_6040.reindex(common).fillna(0)
    rew         = ret_ew.reindex(common).fillna(0)
    eq_strategy = (1 + strategy_rets).cumprod()
    eq_6040     = (1 + r6040).cumprod()
    eq_ew       = (1 + rew).cumprod()
    drawdown    = (eq_strategy - eq_strategy.cummax()) / eq_strategy.cummax()

    regime_colors = {0: "#16a34a", 1: "#d97706", 2: "#dc2626"}
    regime_alpha  = {0: 0.08, 1: 0.12, 2: 0.18}
    regimes       = results_df["regime"].reindex(common)

    def shade(ax):
        curr  = regimes.iloc[0]
        start = regimes.index[0]
        for date, reg in regimes.items():
            if reg != curr:
                ax.axvspan(start, date, color=regime_colors[curr],
                           alpha=regime_alpha[curr], zorder=0)
                curr, start = reg, date
        ax.axvspan(start, regimes.index[-1], color=regime_colors[curr],
                   alpha=regime_alpha[curr], zorder=0)

    fig, axes = plt.subplots(3, 1, figsize=(16, 14))
    fig.suptitle("Macro-Aware Tactical Asset Allocation — Backtest Results",
                 fontsize=15, y=0.98)

    ax = axes[0]
    ax.plot(eq_strategy, color="#2563eb", lw=2.0, label="Regime-Shift", zorder=3)
    ax.plot(eq_6040,     color="#6b7280", lw=1.5, linestyle="--", label="60/40", zorder=2)
    ax.plot(eq_ew,       color="#d97706", lw=1.5, linestyle=":",  label="Equal Weight", zorder=2)
    shade(ax)
    ax.set_yscale("log")
    ax.set_title("Equity Curves — Growth of $1 (log scale)", fontsize=11)
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.2)
    for eq, color, name in [(eq_strategy,"#2563eb","Strategy"),
                             (eq_6040,"#6b7280","60/40"),
                             (eq_ew,"#d97706","EqWt")]:
        ax.annotate(f"{name}: ${eq.iloc[-1]:.2f}",
                    xy=(eq.index[-1], eq.iloc[-1]),
                    xytext=(10, 0), textcoords="offset points",
                    fontsize=8, color=color, va="center")

    ax = axes[1]
    dd_6040 = (eq_6040 - eq_6040.cummax()) / eq_6040.cummax()
    ax.fill_between(drawdown.index, drawdown.values, 0,
                    color="#2563eb", alpha=0.4, label="Strategy")
    ax.plot(dd_6040, color="#6b7280", lw=1.2, linestyle="--", label="60/40", alpha=0.8)
    shade(ax)
    ax.set_title("Drawdown", fontsize=11)
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.2)

    ax = axes[2]
    roll_sharpe      = strategy_rets.rolling(252).mean() / strategy_rets.rolling(252).std() * np.sqrt(252)
    roll_sharpe_6040 = r6040.rolling(252).mean() / r6040.rolling(252).std() * np.sqrt(252)
    ax.plot(roll_sharpe,      color="#2563eb", lw=1.5, label="Strategy")
    ax.plot(roll_sharpe_6040, color="#6b7280", lw=1.2, linestyle="--", label="60/40", alpha=0.8)
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.axhline(1, color="#16a34a", lw=0.8, linestyle=":", alpha=0.7, label="Sharpe=1")
    shade(ax)
    ax.set_title("Rolling 1-Year Sharpe Ratio", fontsize=11)
    ax.set_ylabel("Sharpe Ratio")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("Date")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Chart saved → {save_path}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    df = pd.read_csv("data/master_data.csv", index_col=0, parse_dates=True)

    print("Running walk-forward simulation (takes ~5 mins)...")
    results_df = run_walk_forward(df, train_years=2, retrain_months=3)

    backtest_start   = results_df.index[0]
    ret_6040, ret_ew = compute_benchmark_returns(df.loc[backtest_start:])
    strategy_rets    = results_df["strategy_ret"]

    m_strategy = compute_metrics(strategy_rets, label="Regime-Shift")
    m_6040     = compute_metrics(ret_6040.reindex(strategy_rets.index).fillna(0), label="60/40")
    m_ew       = compute_metrics(ret_ew.reindex(strategy_rets.index).fillna(0),   label="Equal Wt")
    print_metrics_table([m_strategy, m_6040, m_ew])

    results_df.to_csv("data/backtest_results.csv")

    print("\nGenerating charts...")
    plot_equity_curves(strategy_rets, ret_6040, ret_ew, results_df)