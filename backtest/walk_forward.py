"""
REGIME-SHIFT | Phase 4: Walk-Forward Backtester
=================================================
Goal: Simulate running this strategy in real life — no cheating,
no future data — and measure whether it beats passive benchmarks.

THE CORE LOOP (explain this in every interview):
  For each month t from start to end:
    1. Train HMM on ALL data before month t
    2. Detect today's regime using the freshly trained model
    3. Optimize portfolio weights for that regime
    4. Apply those weights for the next month
    5. Record the actual return earned
    6. Roll forward: t = t + 1

This means the model at any point in time has NEVER seen future data.
It's exactly what would have happened in live trading.

METRICS WE COMPUTE:
  - Total return: how much did $1 grow to?
  - Annualized return: average yearly growth rate
  - Volatility: how bumpy was the ride?
  - Sharpe ratio: return per unit of risk (higher = better)
  - Sortino ratio: like Sharpe but only penalizes DOWNSIDE volatility
  - Max drawdown: the worst peak-to-trough loss (the gut-punch metric)
  - Calmar ratio: annualized return / max drawdown
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import warnings
import os
warnings.filterwarnings("ignore")

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import cvxpy as cp


# ============================================================
# SECTION 1: PERFORMANCE METRICS
# ============================================================

def compute_metrics(returns_series, rf_rate=0.04, label="Strategy"):
    """
    Computes all the standard performance metrics used in finance.

    EACH METRIC EXPLAINED:

    Total Return: (final_value / initial_value) - 1
      Simple. $1 invested became how much?

    Annualized Return: (1 + total_return)^(1/years) - 1
      Converts total return to a per-year figure for fair comparison.
      A 100% return over 10 years = 7.2% per year (not 10%).

    Volatility (annualized): std(daily returns) × sqrt(252)
      How much does the portfolio value swing day to day?
      Lower is better, all else equal.
      sqrt(252) converts daily std to annual std (variance scales with time).

    Sharpe Ratio: (annual_return - risk_free_rate) / annual_volatility
      The gold standard risk-adjusted return metric.
      Tells you: for each unit of risk taken, how much EXCESS return did you earn?
      > 1.0 = good, > 1.5 = very good, > 2.0 = exceptional
      The S&P500 historically has a Sharpe of ~0.5.

    Sortino Ratio: (annual_return - rf) / downside_volatility
      Like Sharpe but only counts DOWNSIDE volatility in the denominator.
      Rationale: investors don't mind upside volatility (prices going up fast).
      Only downside moves are the "real" risk.
      Sortino > Sharpe means your strategy has more upside than downside moves.

    Maximum Drawdown: max(peak_value - trough_value) / peak_value
      The worst loss from peak to bottom before recovery.
      CRUCIAL for real investors — a -50% drawdown requires a +100% gain to recover.
      Hedge funds often have a "high-water mark" clause: if drawdown > X%, the
      fund closes. This is why max drawdown can be a career-ending metric.

    Calmar Ratio: annualized_return / |max_drawdown|
      How much return per unit of max drawdown risk?
      Preferred by risk managers over Sharpe because max drawdown is more
      intuitive than volatility. > 0.5 is decent, > 1.0 is good.
    """
    trading_days = 252
    r = returns_series.dropna()
    n_years = len(r) / trading_days

    # Cumulative return
    cumulative = (1 + r).cumprod()
    total_return = cumulative.iloc[-1] - 1

    # Annualized return
    annual_return = (1 + total_return) ** (1 / n_years) - 1

    # Volatility
    annual_vol = r.std() * np.sqrt(trading_days)

    # Sharpe ratio
    sharpe = (annual_return - rf_rate) / annual_vol if annual_vol > 0 else 0

    # Sortino ratio (only downside deviation)
    downside_returns = r[r < 0]
    downside_vol = downside_returns.std() * np.sqrt(trading_days)
    sortino = (annual_return - rf_rate) / downside_vol if downside_vol > 0 else 0

    # Maximum drawdown
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Calmar ratio
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    # Win rate (% of days with positive return)
    win_rate = (r > 0).mean()

    return {
        "label": label,
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "win_rate": win_rate,
        "n_years": n_years
    }


def print_metrics_table(metrics_list):
    """Prints a side-by-side comparison table of all strategies."""
    print("\n" + "="*70)
    print("PERFORMANCE TEARSHEET")
    print("="*70)

    labels = [m["label"] for m in metrics_list]
    header = f"{'Metric':<25}" + "".join(f"{l:>18}" for l in labels)
    print(header)
    print("-"*70)

    rows = [
        ("Total Return",     "total_return",  "{:>17.1%}"),
        ("Annual Return",    "annual_return",  "{:>17.1%}"),
        ("Annual Volatility","annual_vol",     "{:>17.1%}"),
        ("Sharpe Ratio",     "sharpe",         "{:>17.2f}"),
        ("Sortino Ratio",    "sortino",        "{:>17.2f}"),
        ("Max Drawdown",     "max_drawdown",   "{:>17.1%}"),
        ("Calmar Ratio",     "calmar",         "{:>17.2f}"),
        ("Win Rate",         "win_rate",       "{:>17.1%}"),
    ]

    for row_label, key, fmt in rows:
        line = f"{row_label:<25}"
        for m in metrics_list:
            line += fmt.format(m[key])
        print(line)

    print("="*70)


# ============================================================
# SECTION 2: BENCHMARK STRATEGIES
# ============================================================

def compute_benchmark_returns(returns):
    """
    Computes daily returns for two passive benchmark portfolios.

    60/40 PORTFOLIO:
    The classic institutional benchmark. 60% stocks (SPY), 40% bonds (TLT).
    Invented in the 1950s by Harry Markowitz. Still used by pension funds,
    endowments, and retail investors worldwide.
    WHY IT WORKS: bonds and stocks are negatively correlated, so when stocks
    crash, bonds cushion the blow.
    WHY IT FAILS: in high-inflation regimes (like 2022), BOTH stocks and bonds
    fall simultaneously — the correlation breaks down.

    EQUAL WEIGHT (1/N):
    Simply put 1/3 in each asset and rebalance monthly.
    Surprisingly hard to beat! Research (DeMiguel 2009) shows that
    naive 1/N often outperforms optimized portfolios out-of-sample
    because optimization errors in mu and Sigma compound.
    This is called the "1/N puzzle" in finance.
    """
    # 60/40: 60% SPY, 40% TLT, held fixed (no rebalancing for simplicity)
    ret_6040 = 0.60 * returns["SPY_ret"] + 0.40 * returns["TLT_ret"]

    # Equal weight: 1/3 each, rebalance monthly
    ret_ew = (returns["SPY_ret"] + returns["TLT_ret"] + returns["GLD_ret"]) / 3

    return ret_6040.rename("60/40"), ret_ew.rename("Equal Weight")


# ============================================================
# SECTION 3: WALK-FORWARD ENGINE
# ============================================================

def build_features(df):
    """Same feature engineering as Phase 2 — must be identical."""
    features = pd.DataFrame(index=df.index)
    features["spy_ret"]     = df["SPY_ret"]
    features["spy_vol_20d"] = df["SPY_ret"].rolling(20).std() * np.sqrt(252)
    features["vix"]         = df["VIX"]
    features["tlt_ret"]     = df["TLT_ret"]
    features["gld_ret"]     = df["GLD_ret"]
    return features.dropna()


def train_hmm_on_window(features_window, n_states=3):
    """Trains HMM on a window of data and returns regime for the last day."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features_window.values)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=200,
        random_state=42
    )
    model.fit(scaled)
    raw_states = model.predict(scaled)

    # Interpret states: highest avg return = Bull(0), lowest = Crisis(2)
    spy_ret = features_window["spy_ret"].values
    state_returns = {s: spy_ret[raw_states == s].mean()
                     for s in range(n_states)}
    sorted_states = sorted(state_returns, key=state_returns.get, reverse=True)
    label_map = {sorted_states[0]: 0, sorted_states[1]: 1, sorted_states[2]: 2}

    # Return the regime for the LAST day in the window (today's regime)
    last_raw_state = raw_states[-1]
    current_regime = label_map[last_raw_state]
    return current_regime


def optimize_for_regime(regime, returns_window, prev_weights):
    """
    Runs the right optimizer for the given regime.
    Simplified version of Phase 3's optimizer for use inside the loop.
    """
    assets = ["SPY_ret", "TLT_ret", "GLD_ret"]
    r = returns_window[assets]
    mu = r.mean().values * 252
    Sigma = r.cov().values * 252 + 1e-6 * np.eye(3)
    n = 3

    def min_variance(penalty=0.001):
        w = cp.Variable(n)
        cost = cp.quad_form(w, Sigma)
        if prev_weights is not None:
            cost += penalty * cp.norm1(w - prev_weights)
        prob = cp.Problem(
            cp.Minimize(cost),
            [cp.sum(w) == 1, w >= 0, w <= 0.75]
        )
        prob.solve(solver=cp.CLARABEL)
        if w.value is None:
            return np.ones(n) / n
        wts = np.maximum(w.value, 0)
        return wts / wts.sum()

    def max_sharpe():
        excess = mu - 0.04
        if np.all(excess <= 0):
            return min_variance()
        w = cp.Variable(n)
        prob = cp.Problem(
            cp.Minimize(cp.quad_form(w, Sigma)),
            [excess @ w == 1, w >= 0, w <= 0.90]
        )
        prob.solve(solver=cp.CLARABEL)
        if w.value is None:
            return np.ones(n) / n
        wts = np.maximum(w.value, 0)
        return wts / wts.sum()

    if regime == 0:    # Bull
        return max_sharpe()
    elif regime == 1:  # Bear
        return min_variance(penalty=0.001)
    else:              # Crisis
        return min_variance(penalty=0.005)


def run_walk_forward(df, train_years=2, retrain_months=3):
    """
    The main walk-forward backtest loop.

    HOW IT WORKS:
    - We start with `train_years` of data (e.g. 2 years) as the initial window
    - Every `retrain_months` months we RETRAIN the HMM from scratch on all
      data up to that point
    - Between retraining dates, we hold the current weights
    - We record the ACTUAL daily return of those weights each day

    WHY RETRAIN EVERY 3 MONTHS?
    Markets evolve. The Bull regime of 2005 looks different from the Bull
    regime of 2020. Periodic retraining lets the model adapt.
    Too frequent (weekly): computationally expensive, overfitting risk
    Too infrequent (yearly): model goes stale, misses regime changes
    Quarterly is the industry standard for systematic strategies.

    WHAT THIS RETURNS:
    A DataFrame with one row per day containing:
    - strategy_ret: the daily return of our regime-shifting portfolio
    - regime: which regime was detected that day
    - SPY_w, TLT_w, GLD_w: what weights we held that day
    """
    features = build_features(df)

    # Align everything to the same dates
    common = df.index.intersection(features.index)
    df = df.loc[common]
    features = features.loc[common]

    trading_days = 252
    train_days = train_years * trading_days
    retrain_days = retrain_months * 21  # ~21 trading days per month

    results = []
    prev_weights = np.array([1/3, 1/3, 1/3])
    current_regime = 1  # start assuming Bear (conservative)
    current_weights = prev_weights.copy()

    print(f"   Walk-forward: {train_years}yr initial window, "
          f"retrain every {retrain_months} months")
    print(f"   Total days to simulate: {len(df) - train_days}")

    for i in range(train_days, len(df)):
        date = df.index[i]

        # ── Retrain HMM periodically ──────────────────────────────────────
        if (i - train_days) % retrain_days == 0:
            window_features = features.iloc[:i]
            window_returns  = df.iloc[:i]

            try:
                proposed_regime = train_hmm_on_window(window_features)
                # Confirmation rule: only switch if new regime differs AND
                # we've seen it proposed at least twice in a row (sticky regimes)
                if proposed_regime == current_regime:
                    confirmed_count = getattr(run_walk_forward, '_confirm', 0) + 1
                else:
                    confirmed_count = getattr(run_walk_forward, '_confirm', 0) + 1
                    if confirmed_count < 2:
                        run_walk_forward._confirm = confirmed_count
                        proposed_regime = current_regime  # don't switch yet
                    else:
                        confirmed_count = 0
                run_walk_forward._confirm = confirmed_count
                current_regime = proposed_regime
                current_weights = optimize_for_regime(
                    current_regime, window_returns, prev_weights
                )
                prev_weights = current_weights.copy()

                regime_name = {0: "Bull", 1: "Bear", 2: "Crisis"}[current_regime]
                if (i - train_days) % (retrain_days * 4) == 0:  # print quarterly
                    print(f"   {date.date()} | Retrained | Regime: {regime_name:6s} | "
                          f"SPY={current_weights[0]:.2f} "
                          f"TLT={current_weights[1]:.2f} "
                          f"GLD={current_weights[2]:.2f}")
            except Exception:
                pass  # keep previous weights if optimization fails

        # ── Record today's return using current weights ────────────────────
        today_returns = df.iloc[i][["SPY_ret", "TLT_ret", "GLD_ret"]].values
        portfolio_return = np.dot(current_weights, today_returns)

        results.append({
            "date": date,
            "strategy_ret": portfolio_return,
            "regime": current_regime,
            "SPY_w": current_weights[0],
            "TLT_w": current_weights[1],
            "GLD_w": current_weights[2],
        })

    results_df = pd.DataFrame(results).set_index("date")
    print(f"\n   ✓ Simulated {len(results_df)} days of live trading")
    return results_df


# ============================================================
# SECTION 4: THE HERO CHART — EQUITY CURVES
# ============================================================

def plot_equity_curves(strategy_rets, ret_6040, ret_ew, results_df,
                       save_path="reports/04_backtest.png"):
    """
    The most important chart in the project.
    Shows how $1 invested in each strategy grew over time.

    EQUITY CURVE: cumulative product of (1 + daily_return)
    If day 1 return = +1%, day 2 = -0.5%:
    Equity = 1 × 1.01 × 0.995 = 1.0050 (0.5% total gain)

    WHAT TO LOOK FOR:
    1. Does your strategy end higher than benchmarks? (Total return)
    2. Is your drawdown shallower in 2008 and 2020? (Risk management)
    3. Does your line recover faster after crashes? (Regime detection working)
    4. Is it smoother overall? (Lower volatility)

    REGIME-SHADED BACKGROUND:
    Green = Bull, Orange = Bear, Red = Crisis
    You should see: line flattens or rises during red zones (crisis defense)
    while benchmarks are crashing.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Align all series to the same dates
    common = strategy_rets.index
    r6040 = ret_6040.reindex(common).fillna(0)
    rew   = ret_ew.reindex(common).fillna(0)

    # Compute equity curves (cumulative product)
    eq_strategy = (1 + strategy_rets).cumprod()
    eq_6040     = (1 + r6040).cumprod()
    eq_ew       = (1 + rew).cumprod()

    # Drawdown series for strategy
    rolling_max = eq_strategy.cummax()
    drawdown    = (eq_strategy - rolling_max) / rolling_max

    fig, axes = plt.subplots(3, 1, figsize=(16, 14))
    fig.suptitle("Regime-Shift | Phase 4: Walk-Forward Backtest Results",
                 fontsize=15, y=0.98)

    regime_colors = {0: "#16a34a", 1: "#d97706", 2: "#dc2626"}
    regime_alpha  = {0: 0.08, 1: 0.12, 2: 0.18}
    regimes = results_df["regime"].reindex(common)

    def shade_regimes(ax):
        if regimes is None or len(regimes) == 0:
            return
        curr = regimes.iloc[0]
        start = regimes.index[0]
        for date, reg in regimes.items():
            if reg != curr:
                ax.axvspan(start, date,
                           color=regime_colors[curr],
                           alpha=regime_alpha[curr], zorder=0)
                curr = reg
                start = date
        ax.axvspan(start, regimes.index[-1],
                   color=regime_colors[curr],
                   alpha=regime_alpha[curr], zorder=0)

    # ── Panel 1: Equity curves ────────────────────────────────────────────
    ax = axes[0]
    ax.plot(eq_strategy.index, eq_strategy.values,
            color="#2563eb", linewidth=2.0, label="Regime-Shift Strategy", zorder=3)
    ax.plot(eq_6040.index, eq_6040.values,
            color="#6b7280", linewidth=1.5, linestyle="--", label="60/40 Benchmark", zorder=2)
    ax.plot(eq_ew.index, eq_ew.values,
            color="#d97706", linewidth=1.5, linestyle=":", label="Equal Weight", zorder=2)

    shade_regimes(ax)
    ax.set_yscale("log")
    ax.set_title("Equity Curves — Growth of $1 (log scale)", fontsize=11)
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.2)

    # Add final value annotations
    for eq, color, name in [
        (eq_strategy, "#2563eb", "Strategy"),
        (eq_6040,     "#6b7280", "60/40"),
        (eq_ew,       "#d97706", "EqWt"),
    ]:
        ax.annotate(f"{name}: ${eq.iloc[-1]:.2f}",
                    xy=(eq.index[-1], eq.iloc[-1]),
                    xytext=(10, 0), textcoords="offset points",
                    fontsize=8, color=color, va="center")

    # ── Panel 2: Drawdown ─────────────────────────────────────────────────
    ax = axes[1]
    dd_6040 = (eq_6040 - eq_6040.cummax()) / eq_6040.cummax()
    ax.fill_between(drawdown.index, drawdown.values, 0,
                    color="#2563eb", alpha=0.4, label="Strategy")
    ax.plot(dd_6040.index, dd_6040.values,
            color="#6b7280", linewidth=1.2, linestyle="--", label="60/40", alpha=0.8)
    shade_regimes(ax)
    ax.set_title("Drawdown — peak-to-trough loss at each point in time", fontsize=11)
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.2)

    # ── Panel 3: Rolling 252-day Sharpe ───────────────────────────────────
    ax = axes[2]
    roll_sharpe = (strategy_rets.rolling(252).mean() /
                   strategy_rets.rolling(252).std() * np.sqrt(252))
    roll_sharpe_6040 = (r6040.rolling(252).mean() /
                        r6040.rolling(252).std() * np.sqrt(252))
    ax.plot(roll_sharpe.index, roll_sharpe.values,
            color="#2563eb", linewidth=1.5, label="Strategy rolling Sharpe")
    ax.plot(roll_sharpe_6040.index, roll_sharpe_6040.values,
            color="#6b7280", linewidth=1.2, linestyle="--", label="60/40 rolling Sharpe")
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.axhline(1, color="#16a34a", linewidth=0.8, linestyle=":", alpha=0.7, label="Sharpe=1")
    shade_regimes(ax)
    ax.set_title("Rolling 1-Year Sharpe Ratio", fontsize=11)
    ax.set_ylabel("Sharpe Ratio")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("Date")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✓ Chart saved → {save_path}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*55)
    print("REGIME-SHIFT | Phase 4: Walk-Forward Backtest")
    print("="*55 + "\n")

    # Load data
    print("📂  Loading data...")
    df = pd.read_csv("data/master_data.csv", index_col=0, parse_dates=True)
    print(f"   ✓ Loaded {len(df)} days")

    # Run walk-forward backtest
    print("\n🔄  Running walk-forward simulation...")
    print("   (This retrains the HMM from scratch every 3 months — takes 3-5 mins)\n")
    results_df = run_walk_forward(df, train_years=2, retrain_months=3)

    # Compute benchmark returns (aligned to backtest period)
    backtest_start = results_df.index[0]
    df_bt = df.loc[backtest_start:]
    ret_6040, ret_ew = compute_benchmark_returns(df_bt)

    strategy_rets = results_df["strategy_ret"]

    # Compute metrics for all three strategies
    print("\n📊  Computing performance metrics...")
    m_strategy = compute_metrics(strategy_rets,               label="Regime-Shift")
    m_6040     = compute_metrics(ret_6040.reindex(strategy_rets.index).fillna(0), label="60/40")
    m_ew       = compute_metrics(ret_ew.reindex(strategy_rets.index).fillna(0),   label="Equal Wt")

    print_metrics_table([m_strategy, m_6040, m_ew])

    # Save results
    results_df.to_csv("data/backtest_results.csv")
    print(f"\n   ✓ Results saved → data/backtest_results.csv")

    # Plot
    print("\n📊  Generating backtest chart...")
    plot_equity_curves(strategy_rets, ret_6040, ret_ew, results_df)

    print("\n✅  Phase 4 complete!")
    print("    Open reports/04_backtest.png — this is your hero chart.")
    print("    Next: python reports/tearsheet.py")