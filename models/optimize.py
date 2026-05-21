"""
REGIME-SHIFT | Phase 3: Convex Portfolio Optimization
=======================================================
Goal: Given today's regime, find the mathematically optimal
portfolio weights for SPY, TLT, and GLD.

DIFFERENT OBJECTIVE PER REGIME:
  Bull   → Maximize Sharpe Ratio  (go for returns, accept some risk)
  Bear   → Minimize Volatility    (preserve capital, reduce swings)
  Crisis → Minimize Volatility    (maximum defense, heavy bonds/gold)

WHY CONVEX OPTIMIZATION?
Portfolio optimization is a CONVEX problem — meaning there's one
global optimum, no local traps. cvxpy lets you write the math
almost exactly as you'd write it on paper, then solves it instantly.

INTERVIEW EXPLANATION:
"We use cvxpy to solve a quadratic program at each rebalancing date.
In Bull regimes the objective is to maximize the Sharpe ratio —
return per unit of risk. In Bear and Crisis regimes we switch to
minimum variance to protect capital. We add a transaction cost
penalty of 10bps per unit of turnover to prevent excessive trading."
"""

import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import pickle


# ============================================================
# SECTION 1: ESTIMATING EXPECTED RETURNS & COVARIANCE
# ============================================================

def estimate_params(returns_window):
    """
    Estimates the inputs the optimizer needs from a window of historical returns.

    TWO INPUTS NEEDED:
    1. Expected returns (mu) — what return do we expect from each asset?
    2. Covariance matrix (Sigma) — how do assets move relative to each other?

    WHY A ROLLING WINDOW?
    We use the PAST 252 days (1 year) of returns to estimate these.
    This is called a "rolling window" estimator.
    - Too short (e.g. 20 days): noisy estimates, dominated by recent events
    - Too long (e.g. 5 years): stale, doesn't reflect current market conditions
    - 252 days = sweet spot used widely in industry

    INTERVIEW TIP — THE COVARIANCE MATRIX:
    Sigma[i][j] = how much asset i and asset j move together.
    If Sigma[SPY][TLT] is negative → when stocks fall, bonds rise.
    This is the mathematical reason bonds diversify a stock portfolio.
    The optimizer uses this to find combinations that reduce overall
    portfolio variance even when individual assets are volatile.

    INTERVIEW TIP — SHRINKAGE:
    Raw sample covariance is noisy, especially with few observations.
    We add a small regularization term (lambda * I) to the diagonal.
    This "shrinks" extreme correlations toward zero — a technique
    called Ledoit-Wolf shrinkage. It makes the optimizer more stable.
    """
    # Annualized expected returns (mean daily return × 252 trading days)
    mu = returns_window.mean() * 252

    # Annualized covariance matrix
    # Daily cov × 252 = annual cov (variance scales linearly with time)
    Sigma = returns_window.cov() * 252

    # Regularization: add small value to diagonal to ensure positive definite
    # (required for the quadratic optimizer to work)
    # This is like saying "each asset has at least a tiny bit of unique risk"
    n = Sigma.shape[0]
    Sigma = Sigma.values + 1e-6 * np.eye(n)

    return mu.values, Sigma


# ============================================================
# SECTION 2: THE OPTIMIZERS
# ============================================================

def optimize_max_sharpe(mu, Sigma, risk_free_rate=0.04):
    """
    Finds the portfolio with the HIGHEST Sharpe ratio.

    SHARPE RATIO = (Portfolio Return - Risk Free Rate) / Portfolio Volatility
    
    WHY SUBTRACT THE RISK-FREE RATE?
    The risk-free rate (currently ~4-5% for US T-bills) is what you'd earn
    doing NOTHING (just holding cash). Sharpe measures excess return per
    unit of risk — how much are you being compensated for taking risk?
    A Sharpe of 1.0 means you earn 1% extra return for each 1% of volatility.

    THE MATH TRICK (Sharpe maximization):
    Maximizing Sharpe is tricky because it's a ratio (nonlinear).
    We use a standard transformation:
      Let y = w / (w^T * (mu - rf))  →  then maximize over y
      Subject to: Sigma * y ≤ some constraint
    This converts it to a QUADRATIC PROGRAM which cvxpy solves easily.
    
    CONSTRAINTS:
    - weights ≥ 0 (no shorting)
    - weights sum to 1 (fully invested)
    """
    n = len(mu)
    excess_returns = mu - risk_free_rate

    # If expected returns are all below risk-free rate, fall back to min-vol
    if np.all(excess_returns <= 0):
        return optimize_min_variance(mu, Sigma)

    # cvxpy variable: portfolio weights
    w = cp.Variable(n)

    # Portfolio variance: w^T @ Sigma @ w
    portfolio_variance = cp.quad_form(w, Sigma)

    # Portfolio excess return
    portfolio_excess_return = excess_returns @ w

    # Maximize Sharpe = maximize excess_return / sqrt(variance)
    # Equivalent to: minimize variance subject to excess_return = 1
    # (then normalize weights at the end)
    objective = cp.Minimize(portfolio_variance)
    constraints = [
        excess_returns @ w == 1,   # normalization trick
        w >= 0,                     # no short selling
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)

    if prob.status not in ["optimal", "optimal_inaccurate"] or w.value is None:
        # Fallback: equal weights
        return np.ones(n) / n

    # Normalize to get actual weights (they should sum to 1)
    raw_weights = w.value
    raw_weights = np.maximum(raw_weights, 0)  # clip tiny negatives from numerical noise
    weights = raw_weights / raw_weights.sum()

    return weights


def optimize_min_variance(mu, Sigma, prev_weights=None, turnover_penalty=0.001):
    """
    Finds the portfolio with the LOWEST possible volatility.

    Used in Bear and Crisis regimes — we don't care about maximizing
    returns right now, we just want to survive with minimal drawdown.

    PORTFOLIO VARIANCE FORMULA:
    Var(portfolio) = w^T @ Sigma @ w
    where w = weight vector, Sigma = covariance matrix
    This is a quadratic form — exactly what quadratic programming solves.

    TRANSACTION COST PENALTY:
    turnover_penalty × ||w - w_prev||_1
    
    This adds a cost proportional to how much we change our weights.
    ||w - w_prev||_1 = sum of absolute weight changes = total turnover
    10bps (0.001) per unit of turnover means:
    - If you move 10% of your portfolio, you pay 0.1% in costs
    - This discourages the optimizer from making tiny rebalancing trades
    - In real life: bid-ask spread + market impact + commissions

    INTERVIEW TIP:
    Without the transaction cost penalty, the optimizer would rebalance
    every single day, destroying returns with trading costs. This is
    called "portfolio thrashing" and is a classic mistake in naive backtests.
    """
    n = len(mu)

    w = cp.Variable(n)

    # Core objective: minimize portfolio variance
    portfolio_variance = cp.quad_form(w, Sigma)

    # Transaction cost: penalize large changes from previous weights
    if prev_weights is not None:
        turnover = cp.norm1(w - prev_weights)  # L1 norm = total absolute change
        objective = cp.Minimize(portfolio_variance + turnover_penalty * turnover)
    else:
        objective = cp.Minimize(portfolio_variance)

    constraints = [
        cp.sum(w) == 1,   # fully invested
        w >= 0,            # no short selling
        w <= 0.80,         # max 80% in any single asset (diversification floor)
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)

    if prob.status not in ["optimal", "optimal_inaccurate"] or w.value is None:
        return np.ones(n) / n

    weights = np.maximum(w.value, 0)
    weights = weights / weights.sum()

    return weights


# ============================================================
# SECTION 3: REGIME-AWARE WEIGHT COMPUTATION
# ============================================================

def compute_weights_for_regime(regime, mu, Sigma, prev_weights=None):
    """
    Routes to the right optimizer based on the current regime.

    REGIME MAPPING:
    Bull (0)   → Max Sharpe  (offensive: go for returns)
    Bear (1)   → Min Variance (defensive: reduce risk)
    Crisis (2) → Min Variance with extra bond/gold tilt (maximum defense)

    WHY DIFFERENT OBJECTIVES?
    In a Bull market, volatility is low and expected returns are high.
    Taking risk is REWARDED — Max Sharpe makes sense.
    
    In a Crisis, volatility is extreme and correlations spike toward 1
    (everything falls together). Minimizing variance is the only
    rational objective — you're trying to not get wiped out.

    INTERVIEW TIP:
    This is called "regime-conditional optimization." Academic literature
    (e.g. Ang & Bekaert 2002, Guidolin & Timmermann 2007) shows that
    accounting for regime changes in portfolio construction significantly
    improves out-of-sample Sharpe ratios vs static mean-variance.
    """
    if regime == 0:  # Bull
        weights = optimize_max_sharpe(mu, Sigma, risk_free_rate=0.04)
    elif regime == 1:  # Bear
        weights = optimize_min_variance(mu, Sigma, prev_weights, turnover_penalty=0.001)
    else:  # Crisis (regime == 2)
        # Extra penalty in crisis: really don't want to thrash the portfolio
        weights = optimize_min_variance(mu, Sigma, prev_weights, turnover_penalty=0.005)

    return weights


# ============================================================
# SECTION 4: RUN OPTIMIZATION ACROSS ALL DATES
# ============================================================

def compute_all_weights(returns, regimes, lookback=252, rebalance_freq=5):
    """
    Computes optimal portfolio weights for every rebalancing date.

    REBALANCING FREQUENCY:
    We don't optimize every single day — that would be:
    1. Computationally expensive
    2. Unrealistic (too many trades)
    3. Dominated by transaction costs

    Instead, we rebalance every `rebalance_freq` days (default: weekly).
    Between rebalancing dates, we hold the previous weights.

    THE LOOKBACK WINDOW:
    At each date t, we use returns from [t - lookback, t] to estimate
    mu and Sigma. This is the "estimation window."

    IMPORTANT: We only use data BEFORE date t. Never data after t.
    This is what prevents look-ahead bias in this phase.
    (The full walk-forward validation with HMM retraining comes in Phase 4.)

    WHAT THIS FUNCTION RETURNS:
    A DataFrame where each row is a date and columns are weights:
    date       | SPY_w | TLT_w | GLD_w
    2005-12-01 | 0.60  | 0.30  | 0.10
    2005-12-08 | 0.55  | 0.35  | 0.10
    ...
    """
    assets = ["SPY_ret", "TLT_ret", "GLD_ret"]
    asset_names = ["SPY", "TLT", "GLD"]

    # We need at least `lookback` days of history before we can optimize
    start_idx = lookback
    dates = returns.index[start_idx:]

    # Align regimes with returns
    common_dates = returns.index.intersection(regimes.index)
    returns = returns.loc[common_dates]
    regimes = regimes.loc[common_dates]

    weight_records = []
    prev_weights = np.array([1/3, 1/3, 1/3])  # start equal-weighted

    print(f"   Computing weights for {len(dates)} dates (rebalancing every {rebalance_freq} days)...")

    for i, date in enumerate(dates):
        # Only rebalance on scheduled dates (every rebalance_freq days)
        if i % rebalance_freq != 0:
            weight_records.append({
                "date": date,
                "SPY_w": prev_weights[0],
                "TLT_w": prev_weights[1],
                "GLD_w": prev_weights[2],
                "regime": regimes.get(date, 1)
            })
            continue

        # Get the lookback window of returns (past data only)
        window_end_idx = returns.index.get_loc(date)
        window_start_idx = max(0, window_end_idx - lookback)
        returns_window = returns.iloc[window_start_idx:window_end_idx][assets]

        if len(returns_window) < 60:  # need at least 60 days
            weight_records.append({
                "date": date,
                "SPY_w": prev_weights[0],
                "TLT_w": prev_weights[1],
                "GLD_w": prev_weights[2],
                "regime": regimes.get(date, 1)
            })
            continue

        # Get current regime
        current_regime = regimes.get(date, 1)  # default to Bear if missing

        # Estimate parameters from historical window
        mu, Sigma = estimate_params(returns_window)

        # Compute optimal weights for this regime
        weights = compute_weights_for_regime(current_regime, mu, Sigma, prev_weights)

        prev_weights = weights
        weight_records.append({
            "date": date,
            "SPY_w": weights[0],
            "TLT_w": weights[1],
            "GLD_w": weights[2],
            "regime": current_regime
        })

        # Progress update every 100 rebalances
        if (i // rebalance_freq) % 100 == 0:
            regime_name = {0: "Bull", 1: "Bear", 2: "Crisis"}[current_regime]
            print(f"   {date.date()} | Regime: {regime_name:6s} | "
                  f"SPY={weights[0]:.2f} TLT={weights[1]:.2f} GLD={weights[2]:.2f}")

    weights_df = pd.DataFrame(weight_records).set_index("date")
    print(f"\n   ✓ Computed weights for {len(weights_df)} dates")
    return weights_df


# ============================================================
# SECTION 5: VISUALIZATION
# ============================================================

def plot_weights(weights_df, save_path="reports/03_weights.png"):
    """
    Plots how portfolio weights changed over time, colored by regime.

    WHAT YOU SHOULD SEE:
    - During Bull regimes (green): high SPY allocation
    - During Bear regimes (orange): rotating toward TLT (bonds)
    - During Crisis regimes (red): heavy TLT + GLD, minimal SPY
    This is the model doing exactly what a smart investor would do —
    but automatically, systematically, and without emotional bias.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle("Regime-Shift | Phase 3: Portfolio Weights Over Time", fontsize=15)

    regime_colors = {0: "#16a34a", 1: "#d97706", 2: "#dc2626"}
    regime_alpha = {0: 0.10, 1: 0.15, 2: 0.20}
    regime_names = {0: "Bull", 1: "Bear", 2: "Crisis"}

    asset_colors = {"SPY_w": "#2563eb", "TLT_w": "#059669", "GLD_w": "#d97706"}
    asset_labels = {"SPY_w": "SPY (stocks)", "TLT_w": "TLT (bonds)", "GLD_w": "GLD (gold)"}

    for ax_idx, ax in enumerate(axes):
        if ax_idx == 0:
            # Stacked area chart of weights
            ax.stackplot(
                weights_df.index,
                weights_df["SPY_w"],
                weights_df["TLT_w"],
                weights_df["GLD_w"],
                labels=["SPY (stocks)", "TLT (bonds)", "GLD (gold)"],
                colors=["#2563eb", "#059669", "#d97706"],
                alpha=0.75
            )
            ax.set_title("Portfolio weights over time (stacked = 100%)", fontsize=11)
            ax.set_ylabel("Weight")
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
            ax.set_ylim(0, 1)
            ax.legend(loc="upper left", fontsize=9)
        else:
            # Individual weight lines
            for col, color in asset_colors.items():
                ax.plot(weights_df.index, weights_df[col],
                        label=asset_labels[col], color=color,
                        linewidth=1.2, alpha=0.9)
            ax.set_title("Individual asset weights over time", fontsize=11)
            ax.set_ylabel("Weight")
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
            ax.set_ylim(-0.05, 1.05)
            ax.legend(loc="upper left", fontsize=9)

        # Shade by regime
        current_regime = weights_df["regime"].iloc[0]
        start_date = weights_df.index[0]
        for date, row in weights_df.iterrows():
            if row["regime"] != current_regime:
                ax.axvspan(start_date, date,
                           color=regime_colors[current_regime],
                           alpha=regime_alpha[current_regime], zorder=0)
                current_regime = row["regime"]
                start_date = date
        ax.axvspan(start_date, weights_df.index[-1],
                   color=regime_colors[current_regime],
                   alpha=regime_alpha[current_regime], zorder=0)
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("Date")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✓ Chart saved → {save_path}")


def print_weight_summary(weights_df):
    """Prints average weights per regime — the key sanity check."""
    print("\n📊  AVERAGE WEIGHTS BY REGIME:")
    print("    (This is what the optimizer decided on average for each market condition)")
    print(f"\n    {'Regime':8s} | {'SPY':>8s} | {'TLT':>8s} | {'GLD':>8s} | Days")
    print("    " + "-"*45)
    for regime_id, name in {0: "Bull", 1: "Bear", 2: "Crisis"}.items():
        mask = weights_df["regime"] == regime_id
        if mask.sum() == 0:
            continue
        subset = weights_df[mask]
        print(f"    {name:8s} | "
              f"{subset['SPY_w'].mean():>7.1%} | "
              f"{subset['TLT_w'].mean():>7.1%} | "
              f"{subset['GLD_w'].mean():>7.1%} | "
              f"{mask.sum()}")
    print("\n    INTERPRET: Bull should be SPY-heavy. Crisis should be TLT/GLD-heavy.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*55)
    print("REGIME-SHIFT | Phase 3: Portfolio Optimization")
    print("="*55 + "\n")

    # Load Phase 1 + Phase 2 outputs
    print("📂  Loading data...")
    df = pd.read_csv("data/master_data.csv", index_col=0, parse_dates=True)
    regimes = pd.read_csv("data/regimes.csv", index_col=0, parse_dates=True).squeeze()
    print(f"   ✓ Loaded {len(df)} days of returns, {len(regimes)} regime labels")

    # Compute optimal weights across all dates
    print("\n⚙️   Running optimization engine...")
    weights_df = compute_all_weights(
        returns=df,
        regimes=regimes,
        lookback=252,       # 1 year estimation window
        rebalance_freq=5    # rebalance weekly (every 5 trading days)
    )

    # Print summary
    print_weight_summary(weights_df)

    # Save weights (input for Phase 4 backtester)
    weights_df.to_csv("data/weights.csv")
    print(f"\n   ✓ Weights saved → data/weights.csv")

    # Plot
    print("\n📊  Generating weights chart...")
    plot_weights(weights_df)

    print("\n✅  Phase 3 complete!")
    print("    Open reports/03_weights.png to see how weights shifted by regime.")
    print("    Next: python backtest/walk_forward.py")