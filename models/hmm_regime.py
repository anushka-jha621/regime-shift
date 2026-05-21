"""
REGIME-SHIFT | Phase 2: HMM Regime Detection
==============================================
Goal: Train a Hidden Markov Model on our market data to automatically
discover and label market regimes (Bull / Bear / Crisis).

WHAT THIS FILE DOES, IN PLAIN ENGLISH:
1. Load the clean data from Phase 1
2. Build "features" — the signals the HMM will learn from
3. Train the HMM — it figures out the hidden regimes by itself
4. Label every historical day with a regime
5. Plot the regimes over the price chart so we can visually verify it makes sense

KEY CONCEPT — WHAT IS THE HMM ACTUALLY LEARNING?
The HMM is an UNSUPERVISED model. You never tell it "this is a Bull market."
You just feed it returns and VIX, and it discovers on its own that:
  - Some days cluster together (low vol, positive drift) → it calls this State 0
  - Some days cluster differently (high VIX, negative returns) → State 1
  - Some days are extreme (VIX > 40, big losses) → State 2

Your job is to INTERPRET those states after training — look at the stats
of each state and say "State 0 = Bull, State 1 = Bear, State 2 = Crisis."
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import pickle
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


# ============================================================
# SECTION 1: FEATURE ENGINEERING
# ============================================================
# Raw daily returns are noisy. We build richer features to help
# the HMM distinguish regimes more cleanly.

def build_features(df):
    """
    Creates the feature matrix the HMM will train on.

    WHY THESE FEATURES?

    1. SPY daily return
       The most direct signal. Bull markets have positive returns,
       Bear/Crisis have negative. But one day's return is noisy —
       a random bad day doesn't mean a regime change.

    2. Rolling 20-day volatility of SPY
       Volatility CLUSTERS — calm periods stay calm, volatile periods
       stay volatile. This is called "volatility clustering" and is one
       of the most well-documented facts in finance.
       FORMULA: std of last 20 daily returns × sqrt(252) to annualize
       WHY 20 days? Roughly one trading month. Long enough to be stable,
       short enough to react to regime changes.

    3. VIX level
       Forward-looking fear gauge. Captures what the OPTIONS market
       thinks about future volatility — a very different signal from
       realized (historical) volatility.

    4. TLT daily return
       Bonds are a regime signal themselves. In crises, money FLEES
       stocks and ENTERS bonds (flight to safety). So negative SPY + 
       positive TLT = classic crisis signature. The HMM will learn this.

    INTERVIEW TIP: "Feature engineering" is the process of creating
    informative inputs from raw data. The quality of your features
    determines the quality of your model — garbage in, garbage out.
    """
    features = pd.DataFrame(index=df.index)

    # Feature 1: SPY daily return (already in df)
    features["spy_ret"] = df["SPY_ret"]

    # Feature 2: 20-day rolling realized volatility (annualized)
    # std() of log returns × sqrt(252) = annualized vol
    features["spy_vol_20d"] = df["SPY_ret"].rolling(20).std() * np.sqrt(252)

    # Feature 3: VIX level (normalized by its own history to handle level changes)
    features["vix"] = df["VIX"]

    # Feature 4: TLT return (bond market signal)
    features["tlt_ret"] = df["TLT_ret"]

    # Feature 5: GLD return (gold as safe haven signal)
    features["gld_ret"] = df["GLD_ret"]

    # Drop NaN rows (first 20 days will have NaN rolling vol)
    features = features.dropna()

    print(f"   ✓ Built feature matrix: {features.shape[0]} days × {features.shape[1]} features")
    print(f"   Features: {list(features.columns)}")
    return features


def scale_features(features):
    """
    Standardizes features to zero mean and unit variance.

    WHY SCALE?
    VIX ranges from ~10 to ~80. SPY daily returns range from -0.12 to +0.10.
    If you feed both raw into the HMM, VIX will DOMINATE simply because
    its numbers are bigger. Scaling puts everything on equal footing.

    StandardScaler: for each feature, subtract mean, divide by std.
    Result: every feature has mean=0, std=1.

    IMPORTANT: We save the scaler so we can apply the SAME transformation
    to new data in live trading (you can't refit the scaler on new data —
    that would be look-ahead bias).
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    scaled_df = pd.DataFrame(scaled, index=features.index, columns=features.columns)
    print(f"   ✓ Features scaled (mean≈0, std≈1)")
    return scaled_df, scaler


# ============================================================
# SECTION 2: TRAINING THE HMM
# ============================================================

def train_hmm(scaled_features, n_states=3, n_iter=1000, random_state=42):
    """
    Trains a Gaussian HMM on the scaled feature matrix.

    PARAMETERS EXPLAINED:

    n_states=3
        We're telling the model: "assume there are 3 hidden regimes."
        Why 3? Bull, Bear, Crisis. You could try 2 or 4 — but 3 matches
        our domain knowledge about how markets behave.
        INTERVIEW TIP: Choosing n_states is a hyperparameter. In practice
        you'd try 2, 3, 4 and pick based on BIC (Bayesian Information
        Criterion) or domain knowledge.

    covariance_type="full"
        Each state gets its own full covariance matrix. This means the model
        can learn that in a Crisis state, SPY returns and VIX are strongly
        (negatively) correlated — not just that each has its own variance.
        Alternative: "diag" (diagonal) assumes features are independent
        within each state. Faster but less expressive.

    n_iter=1000
        The HMM trains using the Baum-Welch algorithm (a special case of
        EM — Expectation Maximization). It iterates until convergence.
        1000 iterations is plenty for our dataset size.

    HOW BAUM-WELCH WORKS (for interviews):
        E-step: Given current parameters, compute the probability that
                each day belongs to each state.
        M-step: Update the parameters (means, covariances, transitions)
                to maximize the likelihood of the observed data.
        Repeat until parameters stop changing (convergence).
        This is guaranteed to find a LOCAL optimum — not necessarily global.
        That's why we set random_state for reproducibility.
    """
    print(f"\n🧠  Training HMM with {n_states} states...")
    print(f"   This may take 10-30 seconds...")

    model = GaussianHMM(
        n_components=n_states,      # number of hidden states
        covariance_type="full",     # full covariance matrix per state
        n_iter=n_iter,              # max iterations for Baum-Welch
        random_state=random_state,  # for reproducibility
        verbose=False
    )

    # fit() runs the Baum-Welch algorithm
    # Input shape: (n_days, n_features)
    model.fit(scaled_features.values)

    print(f"   ✓ HMM trained! Converged: {model.monitor_.converged}")
    print(f"   Log-likelihood: {model.score(scaled_features.values):.2f}")

    return model


# ============================================================
# SECTION 3: DECODING REGIMES (VITERBI ALGORITHM)
# ============================================================

def decode_regimes(model, scaled_features):
    """
    Uses the Viterbi algorithm to find the most likely sequence of
    hidden states (regimes) for every day in our history.

    THE VITERBI ALGORITHM (interview gold):
    After training, the HMM knows the transition probabilities between
    states and what each state "looks like." But for any given day,
    there's uncertainty about which state it was in.

    Viterbi solves: "What is the SINGLE most likely sequence of states
    that generated our entire observed sequence?"

    It's a dynamic programming algorithm — it considers the full path,
    not just the most likely state day by day. This matters because:
    - The market doesn't jump from Bull to Crisis in one day (usually)
    - A regime change needs multiple consistent signals to be "confirmed"
    - Viterbi respects this by using transition probabilities as a prior

    ANALOGY: If you're tracking a car's GPS and get one weird reading
    putting it in the ocean, Viterbi says "that's probably noise, keep
    it on the road" because the transition "road → ocean" is very unlikely.
    """
    print("\n🔍  Running Viterbi algorithm to decode regime sequence...")

    # predict() runs Viterbi and returns the state index for each day
    raw_states = model.predict(scaled_features.values)

    states = pd.Series(raw_states, index=scaled_features.index, name="regime")
    print(f"   ✓ Decoded {len(states)} days of regime labels")
    return states


# ============================================================
# SECTION 4: INTERPRETING THE STATES
# ============================================================
# The HMM gives us states 0, 1, 2 — but which is Bull? Which is Crisis?
# We look at the average return and volatility of each state to find out.

def interpret_and_relabel(states, df):
    """
    The HMM labels states 0, 1, 2 — but the numbers are arbitrary.
    State 0 might be Bull on one run and Crisis on another (random init).

    We interpret each state by looking at:
    - Average daily SPY return in that state → higher = more bullish
    - Average VIX in that state → higher = more fearful

    Then we RELABEL: the state with highest avg return = Bull (0),
    middle = Bear (1), lowest (most negative + highest VIX) = Crisis (2).

    INTERVIEW TIP: This relabeling is a form of "post-hoc interpretation"
    of an unsupervised model. The model found the clusters; you give them
    meaning using domain knowledge.
    """
    print("\n📊  Interpreting regime states...")

    aligned = df.loc[states.index]
    summary = pd.DataFrame({
        "avg_spy_return":  aligned.groupby(states)["SPY_ret"].mean() * 252,  # annualized
        "avg_spy_vol":     aligned.groupby(states)["SPY_ret"].std() * np.sqrt(252),
        "avg_vix":         aligned.groupby(states)["VIX"].mean(),
        "day_count":       states.value_counts().sort_index(),
        "pct_of_days":     states.value_counts(normalize=True).sort_index() * 100
    })

    print("\n   RAW STATE SUMMARY (before relabeling):")
    print(summary.round(4).to_string())

    # Relabel: sort states by avg annual return (highest = Bull)
    state_order = summary["avg_spy_return"].sort_values(ascending=False).index.tolist()
    # state_order[0] = highest return state → Bull
    # state_order[1] = middle return state  → Bear
    # state_order[2] = lowest return state  → Crisis

    label_map = {
        state_order[0]: 0,  # → Bull
        state_order[1]: 1,  # → Bear
        state_order[2]: 2,  # → Crisis
    }
    relabeled = states.map(label_map)

    regime_names = {0: "Bull", 1: "Bear", 2: "Crisis"}
    print("\n   RELABELED REGIME SUMMARY:")
    for regime_id, name in regime_names.items():
        mask = relabeled == regime_id
        count = mask.sum()
        avg_ret = aligned.loc[mask, "SPY_ret"].mean() * 252 * 100
        avg_vix = aligned.loc[mask, "VIX"].mean()
        pct = count / len(relabeled) * 100
        print(f"   {name:8s}: {count:4d} days ({pct:.1f}%) | "
              f"Avg annual ret: {avg_ret:+.1f}% | Avg VIX: {avg_vix:.1f}")

    return relabeled, label_map


# ============================================================
# SECTION 5: TRANSITION MATRIX
# ============================================================

def print_transition_matrix(model, label_map):
    """
    The transition matrix tells you: given you're in regime X today,
    what's the probability of being in regime Y tomorrow?

    WHAT TO LOOK FOR:
    - Diagonal should be high (>0.95): regimes are "sticky" — markets
      don't jump between Bull and Crisis every other day
    - Off-diagonal Bull→Crisis should be very small
    - Off-diagonal Bear→Crisis should be larger than Bull→Crisis

    INTERVIEW TIP: The transition matrix is one of the key outputs of
    the HMM. It captures "regime persistence" — how long do regimes last?
    If the diagonal is 0.98, the expected regime duration is 1/(1-0.98) = 50 days.
    """
    print("\n📈  TRANSITION MATRIX (prob of switching regimes day-to-day):")
    print("    From \\ To →   Bull    Bear   Crisis")

    # Reorder the raw transition matrix to match our relabeling
    raw_names = {v: k for k, v in label_map.items()}  # new_label → old_label
    regime_names = ["Bull", "Bear", "Crisis"]

    for from_regime in [0, 1, 2]:
        raw_from = raw_names[from_regime]
        row = []
        for to_regime in [0, 1, 2]:
            raw_to = raw_names[to_regime]
            row.append(model.transmat_[raw_from, raw_to])
        print(f"    {regime_names[from_regime]:8s}        " +
              "  ".join(f"{p:.3f}" for p in row))

    print("\n    Read: row = current regime, column = next day's regime")
    print("    Diagonal = probability of STAYING in the same regime")


# ============================================================
# SECTION 6: VISUALIZATION
# ============================================================

def plot_regimes(prices, regimes, df, save_path="reports/02_regimes.png"):
    """
    The hero chart of Phase 2.
    Plots SPY price with colored background showing which regime was active.
    Green = Bull, Orange = Bear, Red = Crisis.

    If your HMM trained correctly, you should see:
    - Green (Bull) during 2005-2007, 2013-2019, 2020-2021 recovery
    - Orange/Red during 2008-2009 financial crisis
    - Red during March 2020 COVID crash
    - Orange during 2022 rate hike bear market
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Align prices with regime dates
    spy_prices = prices["SPY_price"].reindex(regimes.index).ffill()

    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle("Regime-Shift | Phase 2: HMM Regime Detection", fontsize=15, y=0.98)

    colors = {0: "#16a34a", 1: "#d97706", 2: "#dc2626"}  # Bull=green, Bear=amber, Crisis=red
    labels = {0: "Bull", 1: "Bear", 2: "Crisis"}
    alphas = {0: 0.15, 1: 0.20, 2: 0.25}

    # ── Panel 1: SPY Price with regime shading ────────────────────────────
    ax = axes[0]
    ax.plot(spy_prices.index, spy_prices.values, color="#1e40af", linewidth=1.2, zorder=3)

    # Shade background by regime
    _shade_regimes(ax, regimes, colors, alphas)

    ax.set_title("SPY Price — shaded by detected regime", fontsize=11)
    ax.set_ylabel("Price ($)")
    ax.set_yscale("log")  # log scale so 2008 crash and 2020 crash are visible
    ax.grid(True, alpha=0.2, zorder=0)

    legend_patches = [mpatches.Patch(color=colors[i], alpha=0.6, label=labels[i])
                      for i in [0, 1, 2]]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=10)

    # ── Panel 2: VIX with regime shading ─────────────────────────────────
    ax = axes[1]
    vix_aligned = df["VIX"].reindex(regimes.index).ffill()
    ax.plot(vix_aligned.index, vix_aligned.values, color="#7c3aed", linewidth=0.8, zorder=3)
    ax.axhline(20, color="#d97706", linewidth=1, linestyle="--", alpha=0.7, label="VIX=20")
    ax.axhline(30, color="#dc2626", linewidth=1, linestyle="--", alpha=0.7, label="VIX=30")
    _shade_regimes(ax, regimes, colors, alphas)
    ax.set_title("VIX (Fear Index) — should spike during Bear/Crisis regimes", fontsize=11)
    ax.set_ylabel("VIX Level")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.2)

    # ── Panel 3: Regime over time as discrete colored bars ────────────────
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
    print(f"\n   ✓ Chart saved → {save_path}")


def _shade_regimes(ax, regimes, colors, alphas):
    """Helper: shades the background of a chart by regime."""
    current_regime = regimes.iloc[0]
    start_date = regimes.index[0]

    for date, regime in regimes.items():
        if regime != current_regime:
            ax.axvspan(start_date, date,
                       color=colors[current_regime],
                       alpha=alphas[current_regime], zorder=1)
            current_regime = regime
            start_date = date

    # shade the final segment
    ax.axvspan(start_date, regimes.index[-1],
               color=colors[current_regime],
               alpha=alphas[current_regime], zorder=1)


# ============================================================
# SECTION 7: SAVE MODEL
# ============================================================

def save_model(model, scaler, label_map, features_cols,
               model_path="models/hmm_model.pkl"):
    """
    Saves the trained model to disk using pickle.

    WHY SAVE?
    - Training takes time. Save once, load many times.
    - The backtester (Phase 4) will load this model.
    - In a real system, you'd retrain weekly and save the new model.

    We save a dictionary with everything needed to make predictions:
    - model: the trained HMM
    - scaler: the fitted StandardScaler (MUST use same scaler on new data)
    - label_map: maps raw HMM state → Bull/Bear/Crisis
    - feature_cols: the list of features in the right order
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    package = {
        "model": model,
        "scaler": scaler,
        "label_map": label_map,
        "feature_cols": features_cols
    }
    with open(model_path, "wb") as f:
        pickle.dump(package, f)
    print(f"   ✓ Model saved → {model_path}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*55)
    print("REGIME-SHIFT | Phase 2: HMM Regime Detection")
    print("="*55 + "\n")

    # ── Load Phase 1 output ────────────────────────────────────────────────
    print("📂  Loading data from Phase 1...")
    df = pd.read_csv("data/master_data.csv", index_col=0, parse_dates=True)
    prices = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)
    print(f"   ✓ Loaded {len(df)} days of data")

    # ── Build features ─────────────────────────────────────────────────────
    print("\n🔧  Building features...")
    features = build_features(df)

    # ── Scale features ─────────────────────────────────────────────────────
    scaled_features, scaler = scale_features(features)

    # ── Train HMM ──────────────────────────────────────────────────────────
    model = train_hmm(scaled_features, n_states=3)

    # ── Decode regimes ─────────────────────────────────────────────────────
    raw_states = decode_regimes(model, scaled_features)

    # ── Interpret & relabel ────────────────────────────────────────────────
    regimes, label_map = interpret_and_relabel(raw_states, df)

    # ── Print transition matrix ────────────────────────────────────────────
    print_transition_matrix(model, label_map)

    # ── Save regimes to CSV (input for Phase 3 optimizer) ─────────────────
    regimes.to_csv("data/regimes.csv", header=True)
    print(f"\n   ✓ Regime labels saved → data/regimes.csv")

    # ── Save model ─────────────────────────────────────────────────────────
    save_model(model, scaler, label_map, list(features.columns))

    # ── Plot ───────────────────────────────────────────────────────────────
    print("\n📊  Generating regime chart...")
    plot_regimes(prices, regimes, df)

    print("\n✅  Phase 2 complete!")
    print("    Open reports/02_regimes.png to see your regime chart.")
    print("    Next: python models/optimize.py")