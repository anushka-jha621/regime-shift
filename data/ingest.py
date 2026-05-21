"""
REGIME-SHIFT | Phase 1: Data Ingestion
========================================
Goal: Download and clean all the raw data we need.
We pull TWO types of data:
  1. Asset prices  → what we'll INVEST in (SPY, TLT, GLD)
  2. Macro signals → what we'll use to DETECT regimes (VIX, yield spread)

Everything ends up in one clean DataFrame, saved as a CSV.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# ── Try to import fredapi; give a clear error if not installed ──────────────
try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False
    print("⚠  fredapi not installed. Run: pip install fredapi")
    print("   Macro features (VIX from FRED, yield spread) will be skipped.")

# ── FRED API key setup ───────────────────────────────────────────────────────
# The FRED API is FREE. Get your key at: https://fred.stlouisfed.org/docs/api/api_key.html
# After getting the key, either:
#   Option A: set it as an environment variable: export FRED_API_KEY="your_key_here"
#   Option B: paste it directly below (not recommended for shared code)
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")  # reads from environment


# ============================================================
# SECTION 1: ASSET PRICE DATA
# ============================================================
# We download daily adjusted closing prices for three ETFs.
# "Adjusted" means stock splits and dividends are already baked in — 
# so $100 in 2005 is comparable to $100 in 2024.

def download_assets(start="2005-01-01", end=None):
    """
    Downloads daily adjusted close prices for our three assets.

    WHY THESE THREE?
    - SPY = S&P 500 (US stocks). High return in bull markets, crashes hard.
    - TLT = 20yr US Treasury Bonds. Tends to RISE when stocks fall (safe haven).
    - GLD = Gold. Another safe haven. Holds value during crises and inflation.

    These three have low correlation — they don't all move the same direction.
    That's the whole point of diversification, and why the optimizer will
    shift between them based on which regime we detect.
    """
    print("📥  Downloading asset price data from Yahoo Finance...")
    
    tickers = ["SPY", "TLT", "GLD"]
    
    # yf.download gives us a DataFrame with MultiIndex columns.
    # We only want "Close" (adjusted close is now the default).
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True)
    prices = raw["Close"].copy()
    prices = prices.dropna(how="all")  # drop rows where ALL assets are missing    
    # Rename columns so it's obvious what each column is
    prices.columns = ["GLD_price", "SPY_price", "TLT_price"]
    prices = prices[["SPY_price", "TLT_price", "GLD_price"]]  # reorder
    
    print(f"   ✓ Got {len(prices)} days of price data from {prices.index[0].date()} to {prices.index[-1].date()}")
    return prices


def compute_returns(prices):
    """
    Converts price levels → daily log returns.

    WHY LOG RETURNS?
    - Regular return: (P_today - P_yesterday) / P_yesterday
    - Log return: log(P_today / P_yesterday)

    Log returns are preferred in finance because:
    1. They're additive over time: weekly return = sum of daily log returns
    2. They're more normally distributed (important for HMMs which assume Gaussian data)
    3. They're symmetric: a 50% loss followed by a 100% gain = 0, which is correct

    PRACTICAL INTUITION: a log return of 0.01 ≈ a 1% gain. For small moves,
    log returns and regular returns are almost identical. They diverge for big moves.
    """
    returns = np.log(prices / prices.shift(1))
    returns.columns = ["SPY_ret", "TLT_ret", "GLD_ret"]
    returns = returns.dropna(how="all")  # drop rows where ALL are NaN
    returns = returns.fillna(0)          # if TLT missing on a day, treat as 0 return
    
    print(f"   ✓ Computed daily log returns. Shape: {returns.shape}")
    return returns


# ============================================================
# SECTION 2: MACRO INDICATORS (REGIME SIGNALS)
# ============================================================
# These are NOT things we invest in. They're SIGNALS we use to figure out
# what kind of market environment we're in.

def download_macro_fred(start="2005-01-01"):
    """
    Downloads macro indicators from the Federal Reserve's FRED database.
    
    FRED SERIES WE USE:
    
    1. VIXCLS — CBOE Volatility Index (VIX)
       - Measures how much the market EXPECTS stocks to move in the next 30 days
       - VIX < 15: calm, bull market vibes
       - VIX 15-25: normal uncertainty
       - VIX > 30: fear/panic — likely Bear or Crisis regime
       - VIX > 50: full crisis (happened in 2008, 2020)
       - INTERVIEW TIP: VIX is called the "fear gauge" because it spikes when
         investors rush to buy put options (downside insurance)
    
    2. DGS10 — 10-year US Treasury yield
    3. DGS2  — 2-year US Treasury yield
       Spread = DGS10 - DGS2 (the "yield curve")
       - Normally positive: you get paid more for locking money up longer (normal)
       - Inverted (negative): short-term rates > long-term rates → recession signal
       - Every US recession since 1955 was preceded by yield curve inversion
       - INTERVIEW TIP: The yield curve inverts when the Fed raises short-term 
         rates to fight inflation while the market expects future rate cuts (slowdown)
    """
    if not FRED_AVAILABLE or not FRED_API_KEY:
        print("⚠  Skipping FRED download (no API key). Using VIX from Yahoo Finance instead.")
        return None
    
    print("📥  Downloading macro indicators from FRED...")
    fred = Fred(api_key=FRED_API_KEY)
    
    vix    = fred.get_series("VIXCLS",  observation_start=start).rename("VIX")
    dgs10  = fred.get_series("DGS10",   observation_start=start).rename("DGS10")
    dgs2   = fred.get_series("DGS2",    observation_start=start).rename("DGS2")
    
    macro = pd.DataFrame({"VIX": vix, "DGS10": dgs10, "DGS2": dgs2})
    macro["yield_spread"] = macro["DGS10"] - macro["DGS2"]  # positive = normal, negative = inverted
    
    print(f"   ✓ Got macro data. Shape: {macro.shape}")
    return macro[["VIX", "yield_spread"]]


def download_vix_yahoo(start="2005-01-01"):
    print("📥  Downloading VIX from Yahoo Finance (fallback)...")
    vix_raw = yf.download("^VIX", start=start, auto_adjust=True)
    vix = vix_raw[["Close"]].copy()
    vix.columns = ["VIX"]
    print(f"   ✓ Got VIX data. Shape: {vix.shape}")
    return vix


# ============================================================
# SECTION 3: MERGING & CLEANING
# ============================================================

def build_master_dataframe(returns, macro):
    """
    Merges asset returns with macro indicators into one clean DataFrame.

    The tricky part: different data sources have different calendars.
    - Stock markets are closed on weekends and US holidays
    - FRED data sometimes has gaps or different frequencies

    Strategy:
    - Use the returns DataFrame as the "spine" (it defines which days exist)
    - Left-join macro data onto it
    - Forward-fill any gaps in macro data (yesterday's VIX is the best
      estimate for today if today's data is missing)
    - Drop any remaining NaN rows (usually just the very beginning)
    """
    print("🔗  Merging datasets...")
    
    # Align on date index (both use DatetimeIndex)
    macro.index = pd.to_datetime(macro.index)
    returns.index = pd.to_datetime(returns.index)
    
    df = returns.join(macro, how="left")
    
    # Forward-fill macro data (VIX on Monday = VIX from Friday if Monday is missing)
    # Limit=5 means we only forward-fill up to 5 business days — beyond that, drop
    df[macro.columns] = df[macro.columns].ffill(limit=5)
    
    # Drop rows where any column is still NaN (typically the first few days)
    before = len(df)
    df = df.dropna()
    after = len(df)
    
    print(f"   ✓ Merged. Dropped {before - after} rows with missing data.")
    print(f"   ✓ Final dataset: {after} trading days | {df.shape[1]} features")
    print(f"   Date range: {df.index[0].date()} → {df.index[-1].date()}")
    
    return df


# ============================================================
# SECTION 4: EXPLORATORY VISUALIZATION
# ============================================================
# Always plot your data before modeling. This sanity-checks everything.
# You should see: VIX spikes in 2008-09, 2020. Yield curve inverts ~2006-07, 2019.

def plot_overview(prices, df, save_path="reports/01_data_overview.png"):
    """
    Plots 4 panels:
    1. Price levels for SPY, TLT, GLD (to see general trends)
    2. Daily returns distribution (to see if they look roughly Gaussian)
    3. VIX over time (should spike in crises)
    4. Yield spread (should go negative before recessions)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=False)
    fig.suptitle("Regime-Shift | Phase 1: Data Overview", fontsize=16, y=0.98)
    
    # ── Panel 1: Normalized prices ──────────────────────────────────────────
    # Normalize to 100 at start so we can compare assets on the same scale
    ax = axes[0]
    normalized = prices / prices.iloc[0] * 100
    aligned_prices = normalized.reindex(df.index)
    aligned_prices["SPY_price"].plot(ax=ax, color="#2563eb", label="SPY (stocks)", linewidth=1.5)
    aligned_prices["TLT_price"].plot(ax=ax, color="#059669", label="TLT (bonds)", linewidth=1.5)
    aligned_prices["GLD_price"].plot(ax=ax, color="#d97706", label="GLD (gold)", linewidth=1.5)
    ax.set_title("Asset Prices (normalized to 100 at start)", fontsize=11)
    ax.set_ylabel("Index (start = 100)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("")
    
    # ── Panel 2: SPY daily returns ───────────────────────────────────────────
    ax = axes[1]
    ax.fill_between(df.index, df["SPY_ret"], 0, 
                    where=df["SPY_ret"] >= 0, color="#059669", alpha=0.5, label="Up days")
    ax.fill_between(df.index, df["SPY_ret"], 0, 
                    where=df["SPY_ret"] < 0, color="#dc2626", alpha=0.5, label="Down days")
    ax.set_title("SPY Daily Log Returns", fontsize=11)
    ax.set_ylabel("Log Return")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("")
    
    # ── Panel 3: VIX ─────────────────────────────────────────────────────────
    ax = axes[2]
    vix_aligned = df["VIX"].reindex(df.index)
    ax.plot(df.index, vix_aligned, color="#7c3aed", linewidth=1, label="VIX")
    ax.axhline(20, color="#d97706", linewidth=1, linestyle="--", label="VIX=20 (elevated)")
    ax.axhline(30, color="#dc2626", linewidth=1, linestyle="--", label="VIX=30 (fear)")
    ax.fill_between(df.index, vix_aligned, 20, where=vix_aligned > 20, 
                    color="#dc2626", alpha=0.15)
    ax.set_title("VIX — The Fear Index", fontsize=11)
    ax.set_ylabel("VIX Level")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("")
    
    # ── Panel 4: Yield spread (only if available) ─────────────────────────
    ax = axes[3]
    if "yield_spread" in df.columns:
        ax.plot(df.index, df["yield_spread"], color="#0891b2", linewidth=1.2)
        ax.axhline(0, color="#dc2626", linewidth=1.5, linestyle="--", label="Inversion line")
        ax.fill_between(df.index, df["yield_spread"], 0, 
                        where=df["yield_spread"] < 0, color="#dc2626", alpha=0.2, 
                        label="Inverted (recession signal)")
        ax.set_title("Yield Curve Spread (10yr - 2yr Treasury)", fontsize=11)
        ax.set_ylabel("Spread (%)")
        ax.legend(loc="lower left", fontsize=9)
    else:
        ax.text(0.5, 0.5, "Yield spread not available\n(no FRED API key)", 
                transform=ax.transAxes, ha="center", va="center",
                color="gray", fontsize=11)
        ax.set_title("Yield Curve Spread (10yr - 2yr Treasury)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Date")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✓ Chart saved → {save_path}")


# ============================================================
# SECTION 5: DESCRIPTIVE STATISTICS
# ============================================================

def print_summary_stats(df):
    """
    Prints key statistics. Useful for understanding your data before modeling.
    
    INTERVIEW TIP: Be ready to explain what annualized Sharpe ratio means.
    Sharpe = (mean annual return - risk-free rate) / annual volatility
    A Sharpe > 1 is good. > 2 is excellent. The S&P500 historically ~0.5.
    """
    print("\n" + "="*55)
    print("SUMMARY STATISTICS")
    print("="*55)
    
    trading_days = 252  # approximate number of trading days per year
    
    for col in ["SPY_ret", "TLT_ret", "GLD_ret"]:
        asset = col.replace("_ret", "")
        mean_annual  = df[col].mean() * trading_days * 100
        vol_annual   = df[col].std() * np.sqrt(trading_days) * 100
        sharpe       = (df[col].mean() / df[col].std()) * np.sqrt(trading_days)
        worst_day    = df[col].min() * 100
        best_day     = df[col].max() * 100
        
        print(f"\n{asset}:")
        print(f"  Annual return (approx):  {mean_annual:+.1f}%")
        print(f"  Annual volatility:        {vol_annual:.1f}%")
        print(f"  Sharpe ratio:             {sharpe:.2f}")
        print(f"  Worst single day:         {worst_day:.2f}%")
        print(f"  Best single day:          {best_day:.2f}%")
    
    print("\nCorrelation matrix (daily returns):")
    print(df[["SPY_ret", "TLT_ret", "GLD_ret"]].corr().round(2))
    print("\nINTERPRET: Negative SPY-TLT correlation = bonds hedge stocks. That's diversification working.")


# ============================================================
# MAIN: RUN EVERYTHING
# ============================================================

if __name__ == "__main__":
    START_DATE = "2005-01-01"  # 2005 gives us 2008 crisis, 2020 COVID crash
    
    print("\n" + "="*55)
    print("REGIME-SHIFT | Phase 1: Data Ingestion")
    print("="*55 + "\n")
    
    # Step 1: Download asset prices
    prices = download_assets(start=START_DATE)
    
    # Step 2: Compute daily log returns
    returns = compute_returns(prices)
    
    # Step 3: Download macro indicators
    if FRED_AVAILABLE and FRED_API_KEY:
        macro = download_macro_fred(start=START_DATE)
    else:
        macro = download_vix_yahoo(start=START_DATE)
    
    # Step 4: Merge into master DataFrame
    df = build_master_dataframe(returns, macro)
    
    # Step 5: Print summary stats
    print_summary_stats(df)
    
    # Step 6: Save to CSV (this is the input for Phase 2)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/master_data.csv")
    prices.to_csv("data/prices.csv")
    print(f"\n✅  Data saved to data/master_data.csv")
    
    # Step 7: Plot overview
    plot_overview(prices, df, save_path="reports/01_data_overview.png")
    
    print("\n✅  Phase 1 complete! You now have clean data ready for HMM training.")
    print("    Next: python models/hmm_regime.py")
