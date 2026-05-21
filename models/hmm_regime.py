"""
HMM Regime Detection
=====================
Trains a Hidden Markov Model on market data to automatically
discover and label market regimes: Bull, Bear, Crisis.

The HMM is unsupervised — you never tell it what a Bull market is.
You feed it returns and VIX, and it finds the clusters on its own.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import pickle
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


def build_features(df):
    """
    Builds the feature matrix the HMM trains on.

    Features:
    - SPY daily return: the most direct regime signal
    - 20-day rolling vol: volatility clusters — calm stays calm, chaos stays chaotic
    - VIX: forward-looking fear gauge from the options market
    - TLT return: bonds rise when stocks crash (flight to safety)
    - GLD return: gold holds value during crises and inflation

    20-day window is roughly one trading month — stable enough to be
    meaningful, fast enough to catch real regime changes.
    """
    features = pd.DataFrame(index=df.index)
    features["spy_ret"]     = df["SPY_ret"]
    features["spy_vol_20d"] = df["SPY_ret"].rolling(20).std() * np.sqrt(252)
    features["vix"]         = df["VIX"]
    features["tlt_ret"]     = df["TLT_ret"]
    features["gld_ret"]     = df["GLD_ret"]
    return features.dropna()


def scale_features(features):
    """
    Standardizes to zero mean, unit variance.

    VIX ranges 10-80, returns range -0.12 to +0.10. Without scaling,
    VIX dominates the HMM just because its numbers are bigger.
    We save the scaler — must apply the same transform to new data later.
    """
    scaler    = StandardScaler()
    scaled    = scaler.fit_transform(features)
    scaled_df = pd.DataFrame(scaled, index=features.index, columns=features.columns)
    return scaled_df, scaler


def train_hmm(scaled_features, n_states=3, n_iter=1000, random_state=42):
    """
    Trains a Gaussian HMM using the Baum-Welch algorithm.

    n_states=3 → Bull, Bear, Crisis. Could try 2 or 4 but 3 matches
    domain knowledge. In practice you'd pick n_states using BIC.

    covariance_type="full" → each state gets its own covariance matrix,
    so the model can learn that Crisis = high VIX AND negative returns
    (not just that each has high variance independently).

    Baum-Welch is an EM algorithm:
      E-step: given parameters, compute prob each day belongs to each state
      M-step: update parameters to maximize likelihood
      Repeat until convergence. Finds a local optimum, hence random_state.
    """
    print("Training HMM...")
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
        verbose=False
    )
    model.fit(scaled_features.values)
    print(f"  Converged: {model.monitor_.converged}")
    print(f"  Log-likelihood: {model.score(scaled_features.values):.2f}")
    return model


def decode_regimes(model, scaled_features):
    """
    Runs the Viterbi algorithm to find the most likely regime sequence.

    Viterbi finds the single best sequence of hidden states across the
    full history — not just the most likely state day by day. This matters
    because it respects transition probabilities: markets don't jump from
    Bull to Crisis in one day, so Viterbi won't label them that way unless
    the evidence is overwhelming.
    """
    print("Decoding regimes with Viterbi...")
    raw_states = model.predict(scaled_features.values)
    return pd.Series(raw_states, index=scaled_features.index, name="regime")


def interpret_and_relabel(states, df):
    """
    HMM state numbers are arbitrary (0/1/2 means nothing on its own).
    We sort states by average SPY return to get consistent labels:
      highest avg return → Bull (0)
      middle             → Bear (1)
      lowest             → Crisis (2)

    This is post-hoc interpretation of an unsupervised model —
    the model found the clusters, we give them names.
    """
    aligned = df.loc[states.index]
    summary = pd.DataFrame({
        "avg_spy_return": aligned.groupby(states)["SPY_ret"].mean() * 252,
        "avg_spy_vol":    aligned.groupby(states)["SPY_ret"].std() * np.sqrt(252),
        "avg_vix":        aligned.groupby(states)["VIX"].mean(),
        "day_count":      states.value_counts().sort_index(),
        "pct_of_days":    states.value_counts(normalize=True).sort_index() * 100
    })

    print("\nRaw state summary:")
    print(summary.round(3).to_string())

    state_order = summary["avg_spy_return"].sort_values(ascending=False).index.tolist()
    label_map   = {state_order[0]: 0, state_order[1]: 1, state_order[2]: 2}
    relabeled   = states.map(label_map)

    print("\nRelabeled regime summary:")
    for regime_id, name in {0: "Bull", 1: "Bear", 2: "Crisis"}.items():
        mask    = relabeled == regime_id
        avg_ret = aligned.loc[mask, "SPY_ret"].mean() * 252 * 100
        avg_vix = aligned.loc[mask, "VIX"].mean()
        pct     = mask.sum() / len(relabeled) * 100
        print(f"  {name:8s}: {mask.sum():4d} days ({pct:.1f}%) | "
              f"Avg return: {avg_ret:+.1f}% | Avg VIX: {avg_vix:.1f}")

    return relabeled, label_map


def print_transition_matrix(model, label_map):
    """
    Shows probability of switching regimes day-to-day.

    Diagonal should be > 0.95 — regimes are sticky, markets don't
    flip between Bull and Crisis randomly. Expected regime duration
    = 1 / (1 - diagonal). E.g. 0.98 diagonal → ~50 day average regime.
    """
    print("\nTransition matrix (row = current regime, col = next day):")
    print("             Bull    Bear   Crisis")
    raw_names = {v: k for k, v in label_map.items()}
    names     = ["Bull", "Bear", "Crisis"]
    for from_r in [0, 1, 2]:
        row = [model.transmat_[raw_names[from_r], raw_names[to_r]] for to_r in [0, 1, 2]]
        print(f"  {names[from_r]:8s}   " + "  ".join(f"{p:.3f}" for p in row))


def plot_regimes(prices, regimes, df, save_path="reports/02_regimes.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    spy_prices = prices["SPY_price"].reindex(regimes.index).ffill()
    colors = {0: "#16a34a", 1: "#d97706", 2: "#dc2626"}
    labels = {0: "Bull", 1: "Bear", 2: "Crisis"}
    alphas = {0: 0.15, 1: 0.20, 2: 0.25}

    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle("HMM Regime Detection", fontsize=15, y=0.98)

    def shade(ax):
        curr  = regimes.iloc[0]
        start = regimes.index[0]
        for date, regime in regimes.items():
            if regime != curr:
                ax.axvspan(start, date, color=colors[curr], alpha=alphas[curr], zorder=1)
                curr, start = regime, date
        ax.axvspan(start, regimes.index[-1], color=colors[curr], alpha=alphas[curr], zorder=1)

    ax = axes[0]
    ax.plot(spy_prices.index, spy_prices.values, color="#1e40af", linewidth=1.2, zorder=3)
    shade(ax)
    ax.set_title("SPY Price — shaded by detected regime", fontsize=11)
    ax.set_ylabel("Price ($)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.2, zorder=0)
    ax.legend(handles=[mpatches.Patch(color=colors[i], alpha=0.6, label=labels[i])
                       for i in [0, 1, 2]], loc="upper left", fontsize=10)

    ax = axes[1]
    vix_aligned = df["VIX"].reindex(regimes.index).ffill()
    ax.plot(vix_aligned.index, vix_aligned.values, color="#7c3aed", linewidth=0.8, zorder=3)
    ax.axhline(20, color="#d97706", linewidth=1, linestyle="--", alpha=0.7, label="VIX=20")
    ax.axhline(30, color="#dc2626", linewidth=1, linestyle="--", alpha=0.7, label="VIX=30")
    shade(ax)
    ax.set_title("VIX — spikes should align with Bear/Crisis shading", fontsize=11)
    ax.set_ylabel("VIX Level")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.2)

    ax = axes[2]
    for regime_id, color in colors.items():
        mask = regimes == regime_id
        ax.fill_between(regimes.index, regime_id, regime_id + 0.9,
                        where=mask, color=color, alpha=0.8, label=labels[regime_id])
    ax.set_yticks([0.45, 1.45, 2.45])
    ax.set_yticklabels(["Bull", "Bear", "Crisis"])
    ax.set_title("Detected regime each day", fontsize=11)
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.2, axis="x")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved → {save_path}")


def save_model(model, scaler, label_map, feature_cols, model_path="models/hmm_model.pkl"):
    """Saves the trained model so the backtester can load it without retraining."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    package = {"model": model, "scaler": scaler,
               "label_map": label_map, "feature_cols": feature_cols}
    with open(model_path, "wb") as f:
        pickle.dump(package, f)
    print(f"  Model saved → {model_path}")


if __name__ == "__main__":
    df     = pd.read_csv("data/master_data.csv", index_col=0, parse_dates=True)
    prices = pd.read_csv("data/prices.csv",       index_col=0, parse_dates=True)
    print(f"Loaded {len(df)} days of data")

    features                = build_features(df)
    scaled_features, scaler = scale_features(features)
    model                   = train_hmm(scaled_features, n_states=3)
    raw_states              = decode_regimes(model, scaled_features)
    regimes, label_map      = interpret_and_relabel(raw_states, df)

    print_transition_matrix(model, label_map)

    regimes.to_csv("data/regimes.csv", header=True)
    save_model(model, scaler, label_map, list(features.columns))

    print("\nGenerating charts...")
    plot_regimes(prices, regimes, df)
    print("Done.")