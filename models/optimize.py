"""
Regime-Aware Portfolio Optimizer
==================================
Given today's regime, finds the mathematically optimal weights
for SPY, TLT, and GLD using convex optimization.

  Bull   → Maximize Sharpe  (go offensive, capture upside)
  Bear   → Minimize Variance (reduce risk, preserve capital)
  Crisis → Minimize Variance with heavier turnover penalty
"""

import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import pickle


def estimate_params(returns_window):
    """
    Estimates expected returns (mu) and covariance matrix (Sigma)
    from a rolling window of historical returns.

    We use 252 days (1 year) — short enough to reflect current conditions,
    long enough to not be dominated by recent noise.

    The +1e-6 diagonal term is Tikhonov regularization — it ensures Sigma
    is positive definite so the optimizer doesn't blow up. Think of it as
    saying each asset has at least a tiny bit of unique variance.
    """
    mu    = returns_window.mean() * 252
    Sigma = returns_window.cov() * 252
    n     = Sigma.shape[0]
    Sigma = Sigma.values + 1e-6 * np.eye(n)
    return mu.values, Sigma


def optimize_max_sharpe(mu, Sigma, risk_free_rate=0.04):
    """
    Finds the portfolio with the highest Sharpe ratio.

    Sharpe = (return - risk_free) / volatility
    Measures excess return per unit of risk. > 1 is good.

    Maximizing a ratio is nonlinear, so we use a standard trick:
    fix excess_return @ w = 1, then minimize variance. The result
    is equivalent to maximizing Sharpe and converts the problem to
    a quadratic program that cvxpy solves exactly.
    """
    n              = len(mu)
    excess_returns = mu - risk_free_rate

    if np.all(excess_returns <= 0):
        return optimize_min_variance(mu, Sigma)

    w    = cp.Variable(n)
    prob = cp.Problem(
        cp.Minimize(cp.quad_form(w, Sigma)),
        [excess_returns @ w == 1, w >= 0]
    )
    prob.solve(solver=cp.CLARABEL)

    if prob.status not in ["optimal", "optimal_inaccurate"] or w.value is None:
        return np.ones(n) / n

    weights = np.maximum(w.value, 0)
    return weights / weights.sum()


def optimize_min_variance(mu, Sigma, prev_weights=None, turnover_penalty=0.001):
    """
    Finds the portfolio with the lowest possible volatility.

    Portfolio variance = w^T @ Sigma @ w — a quadratic form, which is
    exactly what quadratic programming is designed to minimize.

    Turnover penalty: adds cost proportional to how much weights change.
    ||w - w_prev||_1 = total absolute turnover.
    Without this, the optimizer would rebalance aggressively every day,
    destroying returns with transaction costs (portfolio thrashing).
    """
    n = len(mu)
    w = cp.Variable(n)

    cost = cp.quad_form(w, Sigma)
    if prev_weights is not None:
        cost += turnover_penalty * cp.norm1(w - prev_weights)

    prob = cp.Problem(
        cp.Minimize(cost),
        [cp.sum(w) == 1, w >= 0, w <= 0.80]
    )
    prob.solve(solver=cp.CLARABEL)

    if prob.status not in ["optimal", "optimal_inaccurate"] or w.value is None:
        return np.ones(n) / n

    weights = np.maximum(w.value, 0)
    return weights / weights.sum()


def compute_weights_for_regime(regime, mu, Sigma, prev_weights=None):
    """
    Routes to the right optimizer based on the current regime.

    In Bull markets, vol is low and expected returns are high —
    taking risk is rewarded so Max Sharpe makes sense.

    In Crisis, correlations spike toward 1 (everything falls together)
    and vol explodes. The only rational goal is to not get wiped out.
    Min variance with a heavy turnover penalty keeps us from panicking
    and trading constantly during the worst moments.
    """
    if regime == 0:
        return optimize_max_sharpe(mu, Sigma, risk_free_rate=0.04)
    elif regime == 1:
        return optimize_min_variance(mu, Sigma, prev_weights, turnover_penalty=0.001)
    else:
        return optimize_min_variance(mu, Sigma, prev_weights, turnover_penalty=0.005)


def compute_all_weights(returns, regimes, lookback=252, rebalance_freq=5):
    """
    Computes optimal weights at every rebalancing date.

    Rebalances weekly (every 5 days) rather than daily —
    daily rebalancing would be dominated by transaction costs
    and is unrealistic for any real strategy.

    At each date t, only uses data from [t - lookback, t].
    Never touches future data. This is what prevents look-ahead bias
    in the optimization step (full walk-forward validation is in Phase 4).
    """
    assets       = ["SPY_ret", "TLT_ret", "GLD_ret"]
    common_dates = returns.index.intersection(regimes.index)
    returns      = returns.loc[common_dates]
    regimes      = regimes.loc[common_dates]
    dates        = returns.index[lookback:]

    weight_records = []
    prev_weights   = np.array([1/3, 1/3, 1/3])

    print(f"Computing weights for {len(dates)} dates (rebalancing every {rebalance_freq} days)...")

    for i, date in enumerate(dates):
        if i % rebalance_freq != 0:
            weight_records.append({"date": date, "SPY_w": prev_weights[0],
                                   "TLT_w": prev_weights[1], "GLD_w": prev_weights[2],
                                   "regime": regimes.get(date, 1)})
            continue

        window_end   = returns.index.get_loc(date)
        window_start = max(0, window_end - lookback)
        window       = returns.iloc[window_start:window_end][assets]

        if len(window) < 60:
            weight_records.append({"date": date, "SPY_w": prev_weights[0],
                                   "TLT_w": prev_weights[1], "GLD_w": prev_weights[2],
                                   "regime": regimes.get(date, 1)})
            continue

        current_regime = regimes.get(date, 1)
        mu, Sigma      = estimate_params(window)
        weights        = compute_weights_for_regime(current_regime, mu, Sigma, prev_weights)
        prev_weights   = weights

        weight_records.append({"date": date, "SPY_w": weights[0],
                                "TLT_w": weights[1], "GLD_w": weights[2],
                                "regime": current_regime})

        if (i // rebalance_freq) % 100 == 0:
            regime_name = {0: "Bull", 1: "Bear", 2: "Crisis"}[current_regime]
            print(f"  {date.date()} | {regime_name:6s} | "
                  f"SPY={weights[0]:.2f} TLT={weights[1]:.2f} GLD={weights[2]:.2f}")

    weights_df = pd.DataFrame(weight_records).set_index("date")
    print(f"  Done. {len(weights_df)} dates computed.")
    return weights_df


def plot_weights(weights_df, save_path="reports/03_weights.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle("Portfolio Weights Over Time", fontsize=15)

    regime_colors = {0: "#16a34a", 1: "#d97706", 2: "#dc2626"}
    regime_alpha  = {0: 0.10, 1: 0.15, 2: 0.20}
    asset_colors  = {"SPY_w": "#2563eb", "TLT_w": "#059669", "GLD_w": "#d97706"}
    asset_labels  = {"SPY_w": "SPY (stocks)", "TLT_w": "TLT (bonds)", "GLD_w": "GLD (gold)"}

    for ax_idx, ax in enumerate(axes):
        if ax_idx == 0:
            ax.stackplot(weights_df.index,
                         weights_df["SPY_w"], weights_df["TLT_w"], weights_df["GLD_w"],
                         labels=["SPY (stocks)", "TLT (bonds)", "GLD (gold)"],
                         colors=["#2563eb", "#059669", "#d97706"], alpha=0.75)
            ax.set_title("Stacked weights (= 100% at all times)", fontsize=11)
            ax.set_ylim(0, 1)
        else:
            for col, color in asset_colors.items():
                ax.plot(weights_df.index, weights_df[col],
                        label=asset_labels[col], color=color, linewidth=1.2, alpha=0.9)
            ax.set_title("Individual asset weights", fontsize=11)
            ax.set_ylim(-0.05, 1.05)

        ax.set_ylabel("Weight")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.legend(loc="upper left", fontsize=9)

        curr  = weights_df["regime"].iloc[0]
        start = weights_df.index[0]
        for date, row in weights_df.iterrows():
            if row["regime"] != curr:
                ax.axvspan(start, date, color=regime_colors[curr],
                           alpha=regime_alpha[curr], zorder=0)
                curr, start = row["regime"], date
        ax.axvspan(start, weights_df.index[-1], color=regime_colors[curr],
                   alpha=regime_alpha[curr], zorder=0)
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("Date")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved → {save_path}")


def print_weight_summary(weights_df):
    print("\nAverage weights by regime:")
    print(f"  {'Regime':8s} | {'SPY':>8s} | {'TLT':>8s} | {'GLD':>8s} | Days")
    print("  " + "-"*45)
    for regime_id, name in {0: "Bull", 1: "Bear", 2: "Crisis"}.items():
        mask = weights_df["regime"] == regime_id
        if mask.sum() == 0:
            continue
        s = weights_df[mask]
        print(f"  {name:8s} | {s['SPY_w'].mean():>7.1%} | "
              f"{s['TLT_w'].mean():>7.1%} | {s['GLD_w'].mean():>7.1%} | {mask.sum()}")


if __name__ == "__main__":
    df      = pd.read_csv("data/master_data.csv", index_col=0, parse_dates=True)
    regimes = pd.read_csv("data/regimes.csv",     index_col=0, parse_dates=True).squeeze()
    print(f"Loaded {len(df)} days, {len(regimes)} regime labels")

    weights_df = compute_all_weights(returns=df, regimes=regimes,
                                     lookback=252, rebalance_freq=5)
    print_weight_summary(weights_df)

    weights_df.to_csv("data/weights.csv")

    print("\nGenerating charts...")
    plot_weights(weights_df)
    print("Done.")