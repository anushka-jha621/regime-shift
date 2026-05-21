"""
Data Ingestion
==============
Downloads and cleans all raw data needed for the project.
Two types of data:
  1. Asset prices  → what we invest in (SPY, TLT, GLD)
  2. Macro signals → what we use to detect regimes (VIX, yield spread)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False


FRED_API_KEY = os.environ.get("09c998f16e7bd34a1086ddd329fa7a6d", "")


def download_assets(start="2005-01-01", end=None):
    """
    Downloads daily adjusted close prices for SPY, TLT, and GLD.

    Why these three?
    - SPY = S&P 500. High return in bull markets, crashes hard.
    - TLT = 20yr Treasury Bonds. Tends to rise when stocks fall.
    - GLD = Gold. Holds value during crises and high inflation.

    Low correlation between them is the whole point — when one zigs,
    another zags. That's what makes regime-based reallocation meaningful.
    """
    print("Downloading asset prices...")
    tickers = ["SPY", "TLT", "GLD"]
    raw     = yf.download(tickers, start=start, end=end, auto_adjust=True)
    prices  = raw["Close"].copy().dropna(how="all")
    prices.columns = ["GLD_price", "SPY_price", "TLT_price"]
    prices  = prices[["SPY_price", "TLT_price", "GLD_price"]]
    print(f"  {len(prices)} days from {prices.index[0].date()} to {prices.index[-1].date()}")
    return prices


def compute_returns(prices):
    """
    Converts prices to daily log returns.

    Why log returns?
    - Additive over time: weekly return = sum of daily log returns
    - More Gaussian than simple returns (matters for the HMM)
    - Symmetric: a 50% loss + 100% gain = 0, which is mathematically correct
    """
    returns = np.log(prices / prices.shift(1))
    returns.columns = ["SPY_ret", "TLT_ret", "GLD_ret"]
    returns = returns.dropna(how="all").fillna(0)
    return returns


def download_macro_fred(start="2005-01-01"):
    """
    Downloads macro indicators from FRED.

    VIX — the fear index. Spikes when investors panic-buy put options.
    VIX < 15 = calm, VIX > 30 = fear, VIX > 50 = full crisis.

    Yield spread (10yr - 2yr Treasury):
    Normally positive. When it inverts (goes negative), it's a recession
    signal — every US recession since 1955 was preceded by inversion.
    """
    if not FRED_AVAILABLE or not FRED_API_KEY:
        return None

    print("Downloading macro indicators from FRED...")
    fred  = Fred(api_key=FRED_API_KEY)
    vix   = fred.get_series("VIXCLS", observation_start=start).rename("VIX")
    dgs10 = fred.get_series("DGS10",  observation_start=start).rename("DGS10")
    dgs2  = fred.get_series("DGS2",   observation_start=start).rename("DGS2")

    macro = pd.DataFrame({"VIX": vix, "DGS10": dgs10, "DGS2": dgs2})
    macro["yield_spread"] = macro["DGS10"] - macro["DGS2"]
    return macro[["VIX", "yield_spread"]]


def download_vix_yahoo(start="2005-01-01"):
    """Fallback if no FRED key — pulls VIX directly from Yahoo Finance."""
    print("Downloading VIX from Yahoo Finance...")
    vix_raw = yf.download("^VIX", start=start, auto_adjust=True)
    vix = vix_raw[["Close"]].copy()
    vix.columns = ["VIX"]
    return vix


def build_master_dataframe(returns, macro):
    """
    Merges asset returns with macro indicators.

    Stock markets and FRED have different calendars so we use returns
    as the spine and left-join macro onto it. Forward-fill any gaps
    (Friday's VIX carries over to Monday if Monday is missing).
    """
    macro.index   = pd.to_datetime(macro.index)
    returns.index = pd.to_datetime(returns.index)

    df = returns.join(macro, how="left")
    df[macro.columns] = df[macro.columns].ffill(limit=5)

    before = len(df)
    df     = df.dropna()
    print(f"  Dropped {before - len(df)} rows with missing data")
    print(f"  Final dataset: {len(df)} trading days, {df.shape[1]} features")
    return df


def plot_overview(prices, df, save_path="reports/01_data_overview.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=False)
    fig.suptitle("Data Overview", fontsize=16, y=0.98)

    # Normalized prices
    ax = axes[0]
    normalized     = prices / prices.iloc[0] * 100
    aligned_prices = normalized.reindex(df.index)
    aligned_prices["SPY_price"].plot(ax=ax, color="#2563eb", label="SPY (stocks)", linewidth=1.5)
    aligned_prices["TLT_price"].plot(ax=ax, color="#059669", label="TLT (bonds)",  linewidth=1.5)
    aligned_prices["GLD_price"].plot(ax=ax, color="#d97706", label="GLD (gold)",   linewidth=1.5)
    ax.set_title("Asset Prices (normalized to 100 at start)", fontsize=11)
    ax.set_ylabel("Index")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("")

    # SPY returns
    ax = axes[1]
    ax.fill_between(df.index, df["SPY_ret"], 0,
                    where=df["SPY_ret"] >= 0, color="#059669", alpha=0.5, label="Up days")
    ax.fill_between(df.index, df["SPY_ret"], 0,
                    where=df["SPY_ret"] < 0,  color="#dc2626", alpha=0.5, label="Down days")
    ax.set_title("SPY Daily Log Returns", fontsize=11)
    ax.set_ylabel("Log Return")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("")

    # VIX
    ax = axes[2]
    vix_aligned = df["VIX"].reindex(df.index)
    ax.plot(df.index, vix_aligned, color="#7c3aed", linewidth=1, label="VIX")
    ax.axhline(20, color="#d97706", linewidth=1, linestyle="--", label="VIX=20")
    ax.axhline(30, color="#dc2626", linewidth=1, linestyle="--", label="VIX=30")
    ax.fill_between(df.index, vix_aligned, 20, where=vix_aligned > 20,
                    color="#dc2626", alpha=0.15)
    ax.set_title("VIX — Fear Index", fontsize=11)
    ax.set_ylabel("VIX Level")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("")

    # Yield spread
    ax = axes[3]
    if "yield_spread" in df.columns:
        ax.plot(df.index, df["yield_spread"], color="#0891b2", linewidth=1.2)
        ax.axhline(0, color="#dc2626", linewidth=1.5, linestyle="--", label="Inversion line")
        ax.fill_between(df.index, df["yield_spread"], 0,
                        where=df["yield_spread"] < 0, color="#dc2626", alpha=0.2,
                        label="Inverted (recession signal)")
        ax.set_title("Yield Curve Spread (10yr - 2yr)", fontsize=11)
        ax.set_ylabel("Spread (%)")
        ax.legend(loc="lower left", fontsize=9)
    else:
        ax.text(0.5, 0.5, "Yield spread not available (no FRED API key)",
                transform=ax.transAxes, ha="center", va="center", color="gray", fontsize=11)
        ax.set_title("Yield Curve Spread (10yr - 2yr)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Date")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved → {save_path}")


def print_summary_stats(df):
    """Quick sanity check on the data before modeling."""
    print("\nSUMMARY STATISTICS")
    print("="*45)
    for col in ["SPY_ret", "TLT_ret", "GLD_ret"]:
        asset      = col.replace("_ret", "")
        mean_ann   = df[col].mean() * 252 * 100
        vol_ann    = df[col].std() * np.sqrt(252) * 100
        sharpe     = (df[col].mean() / df[col].std()) * np.sqrt(252)
        print(f"\n{asset}:")
        print(f"  Annual return:    {mean_ann:+.1f}%")
        print(f"  Annual vol:       {vol_ann:.1f}%")
        print(f"  Sharpe ratio:     {sharpe:.2f}")
        print(f"  Worst day:        {df[col].min()*100:.2f}%")
        print(f"  Best day:         {df[col].max()*100:.2f}%")

    print("\nCorrelation matrix:")
    print(df[["SPY_ret", "TLT_ret", "GLD_ret"]].corr().round(2))
    print("\nNegative SPY-TLT correlation = bonds hedge stocks.")


if __name__ == "__main__":
    START_DATE = "2005-01-01"

    prices  = download_assets(start=START_DATE)
    returns = compute_returns(prices)

    if FRED_AVAILABLE and FRED_API_KEY:
        macro = download_macro_fred(start=START_DATE)
    else:
        macro = download_vix_yahoo(start=START_DATE)

    df = build_master_dataframe(returns, macro)
    print_summary_stats(df)

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/master_data.csv")
    prices.to_csv("data/prices.csv")

    print("\nGenerating charts...")
    plot_overview(prices, df, save_path="reports/01_data_overview.png")
    print("Done.")