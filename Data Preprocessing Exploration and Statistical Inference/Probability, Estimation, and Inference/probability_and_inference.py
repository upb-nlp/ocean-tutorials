# ============================================================
# probability_and_inference.py -  Probability, Estimation and Inference
# ============================================================
#Exercises accompanying probability_and_inference.md.
#  Part I   — Probabilities, Probability Distributions, Bayes' Theorem
#  Part II  — Sampling Distributions (repeated sampling, bootstrap, permutation, CLT)
#  Part III — The German Tank Problem (four estimators, bias / variance / MSE)
#  Part IV  — The Overselling Problem (estimation → prediction → optimisation)
# All plots are saved to plots/ next to this script.
# Run: python probability_and_inference.py 
# Requirements: pip install numpy matplotlib scipy
# ============================================================


import os
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, binom

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)
random.seed(RNG_SEED)

BLUE   = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN  = "#2ca02c"
RED    = "#d62728"
PURPLE = "#9467bd"
GRAY   = "#aaaaaa"

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})


# ===========================================================================
# PART I — PROBABILITIES, PROBABILITY DISTRIBUTIONS, BAYES' THEOREM
# ===========================================================================

def part1():
    print("=" * 60)
    print("PART I — PROBABILITIES, DISTRIBUTIONS, BAYES")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 1. Basic probability rules
    # -----------------------------------------------------------------------
    print("\n--- 1. Basic probability rules ---")

    P_A = 0.40   # P(it rains)
    P_B = 0.30   # P(traffic jam)
    P_A_and_B = 0.12   # P(rains AND traffic jam)

    # Complement
    P_not_A = 1 - P_A
    print(f"P(A)       = {P_A}")
    print(f"P(not A)   = 1 - P(A) = {P_not_A}")

    # Addition rule
    P_A_or_B = P_A + P_B - P_A_and_B
    print(f"\nP(A or B)  = P(A) + P(B) - P(A∩B) = {P_A_or_B:.2f}")

    # Conditional probability
    P_B_given_A = P_A_and_B / P_A
    P_A_given_B = P_A_and_B / P_B
    print(f"\nP(B | A)   = P(A∩B) / P(A) = {P_B_given_A:.3f}")
    print(f"P(A | B)   = P(A∩B) / P(B) = {P_A_given_B:.3f}")

    # Independence check: P(A∩B) == P(A)*P(B)?
    independent = math.isclose(P_A_and_B, P_A * P_B, abs_tol=1e-9)
    print(f"\nAre A and B independent?  {independent}  "
          f"(P(A)*P(B) = {P_A * P_B:.2f},  P(A∩B) = {P_A_and_B})")

    # -----------------------------------------------------------------------
    # 2. Probability distributions — parameters and moments
    # -----------------------------------------------------------------------
    print("\n--- 2. Probability distributions ---")

    N_SAMPLES = 100_000

    # -- Bernoulli(p) ---------------------------------------------------------
    p_bern = 0.3
    theoretical_mean_bern = p_bern
    theoretical_var_bern  = p_bern * (1 - p_bern)

    samples_bern = rng.binomial(1, p_bern, N_SAMPLES)
    empirical_mean_bern = samples_bern.mean()
    empirical_var_bern  = samples_bern.var()

    print(f"\nBernoulli(p={p_bern})")
    print(f"  Theoretical:  E[X] = {theoretical_mean_bern:.4f},  "
          f"Var(X) = {theoretical_var_bern:.4f}")
    print(f"  Empirical:    E[X] = {empirical_mean_bern:.4f},  "
          f"Var(X) = {empirical_var_bern:.4f}")

    # -- Binomial(n, p) -------------------------------------------------------
    n_bin, p_bin = 20, 0.3
    theoretical_mean_bin = n_bin * p_bin
    theoretical_var_bin  = n_bin * p_bin * (1 - p_bin)

    samples_bin = rng.binomial(n_bin, p_bin, N_SAMPLES)
    empirical_mean_bin = samples_bin.mean()
    empirical_var_bin  = samples_bin.var()

    print(f"\nBinomial(n={n_bin}, p={p_bin})")
    print(f"  Theoretical:  E[X] = {theoretical_mean_bin:.4f},  "
          f"Var(X) = {theoretical_var_bin:.4f}")
    print(f"  Empirical:    E[X] = {empirical_mean_bin:.4f},  "
          f"Var(X) = {empirical_var_bin:.4f}")

    # -- Normal(mu, sigma) ----------------------------------------------------
    mu_norm, sigma_norm = 5.0, 2.0
    theoretical_mean_norm = mu_norm
    theoretical_var_norm  = sigma_norm ** 2

    samples_norm = rng.normal(mu_norm, sigma_norm, N_SAMPLES)
    empirical_mean_norm = samples_norm.mean()
    empirical_var_norm  = samples_norm.var()

    print(f"\nNormal(μ={mu_norm}, σ={sigma_norm})")
    print(f"  Theoretical:  E[X] = {theoretical_mean_norm:.4f},  "
          f"Var(X) = {theoretical_var_norm:.4f}")
    print(f"  Empirical:    E[X] = {empirical_mean_norm:.4f},  "
          f"Var(X) = {empirical_var_norm:.4f}")

    # -- Plot: all three distributions ----------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Bernoulli bar chart
    ax = axes[0]
    ax.bar([0, 1], [1 - p_bern, p_bern], color=[BLUE, RED],
           alpha=0.75, width=0.4, edgecolor="white")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["0 (failure)", "1 (success)"])
    ax.set_title(f"Bernoulli(p={p_bern})", fontweight="bold")
    ax.set_ylabel("P(X = k)")
    ax.text(0.97, 0.95, f"E[X]={theoretical_mean_bern:.2f}\nVar={theoretical_var_bern:.3f}",
            ha="right", va="top", transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRAY))

    # Binomial bar chart
    ax = axes[1]
    k_vals = np.arange(0, n_bin + 1)
    probs  = binom.pmf(k_vals, n_bin, p_bin)
    ax.bar(k_vals, probs, color=BLUE, alpha=0.75, edgecolor="white", width=0.7)
    ax.axvline(theoretical_mean_bin, color=RED, lw=2, ls="--",
               label=f"E[X] = {theoretical_mean_bin}")
    ax.set_title(f"Binomial(n={n_bin}, p={p_bin})", fontweight="bold")
    ax.set_xlabel("k")
    ax.set_ylabel("P(X = k)")
    ax.text(0.97, 0.95, f"E[X]={theoretical_mean_bin:.2f}\nVar={theoretical_var_bin:.3f}",
            ha="right", va="top", transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRAY))
    ax.legend(fontsize=8)

    # Normal density
    ax = axes[2]
    x = np.linspace(mu_norm - 4 * sigma_norm, mu_norm + 4 * sigma_norm, 300)
    ax.plot(x, norm.pdf(x, mu_norm, sigma_norm), color=BLUE, lw=2.5)
    ax.axvline(mu_norm, color=RED, lw=2, ls="--", label=f"E[X] = {mu_norm}")
    ax.fill_between(x, norm.pdf(x, mu_norm, sigma_norm),
                    where=(x >= mu_norm - sigma_norm) & (x <= mu_norm + sigma_norm),
                    alpha=0.2, color=BLUE, label="±1σ  (≈68%)")
    ax.set_title(f"Normal(μ={mu_norm}, σ={sigma_norm})", fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.text(0.97, 0.95, f"E[X]={theoretical_mean_norm:.2f}\nVar={theoretical_var_norm:.2f}",
            ha="right", va="top", transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRAY))
    ax.legend(fontsize=8)

    fig.suptitle("Probability Distributions — Parameters and Moments",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "part1_distributions.png"), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

    # -----------------------------------------------------------------------
    # 3. Bayes' theorem — fair vs biased coin
    # -----------------------------------------------------------------------
    print("\n--- 3. Bayes' Theorem — fair vs biased coin ---")
    print("""
Setup:
  We have two coins in a bag:
    Coin A (fair):   P(heads) = 0.5
    Coin B (biased): P(heads) = 0.8
  We pick one at random (prior: 50/50) and observe a sequence of flips.
  We want to know: given the observed flips, which coin did we pick?
""")

    prior_A = 0.5   # P(coin A)
    prior_B = 0.5   # P(coin B)
    p_heads_A = 0.5
    p_heads_B = 0.8

    # Simulate flips from coin B (so we know ground truth)
    true_coin = "B"
    true_p    = p_heads_B
    flips     = rng.binomial(1, true_p, 10)   # 1 = heads, 0 = tails

    print(f"True coin drawn: {true_coin}  (P(heads)={true_p})")
    print(f"Flips (1=H, 0=T): {flips.tolist()}")
    print(f"\n{'Flips seen':>12}  {'P(A|data)':>12}  {'P(B|data)':>12}  {'Winner':>8}")
    print("-" * 50)

    post_A, post_B = prior_A, prior_B
    for i, flip in enumerate(flips):
        # Likelihood of this flip under each hypothesis
        lik_A = p_heads_A if flip == 1 else (1 - p_heads_A)
        lik_B = p_heads_B if flip == 1 else (1 - p_heads_B)

        # Bayes update (unnormalised)
        unnorm_A = post_A * lik_A
        unnorm_B = post_B * lik_B
        normaliser = unnorm_A + unnorm_B

        post_A = unnorm_A / normaliser
        post_B = unnorm_B / normaliser

        winner = "A" if post_A > post_B else "B"
        flip_str = "H" if flip == 1 else "T"
        print(f"{flip_str:>12}  {post_A:>12.4f}  {post_B:>12.4f}  {winner:>8}")

    print(f"\nFinal posterior:  P(A|data) = {post_A:.4f},  P(B|data) = {post_B:.4f}")
    print(f"Conclusion: most likely coin = {'A' if post_A > post_B else 'B'}  "
          f"(true coin = {true_coin})")


# ===========================================================================
# PART II — SAMPLING DISTRIBUTIONS
# ===========================================================================

def part2():
    print("\n" + "=" * 60)
    print("PART II — SAMPLING DISTRIBUTIONS")
    print("=" * 60)

    # Population: two groups with different means
    MU1, MU2   = 10.0, 8.0
    SIG1, SIG2 = 2.5, 2.5
    N_OBS  = 30
    N_ITER = 5000

    # Draw one fixed sample (what an analyst would have)
    sample1 = rng.normal(MU1, SIG1, N_OBS)
    sample2 = rng.normal(MU2, SIG2, N_OBS)
    observed_diff = sample1.mean() - sample2.mean()

    print(f"\nPopulation 1: Normal(μ={MU1}, σ={SIG1})")
    print(f"Population 2: Normal(μ={MU2}, σ={SIG2})")
    print(f"True difference in means: {MU1 - MU2}")
    print(f"Observed difference in this sample: {observed_diff:.3f}")
    print(f"Sample size per group: n = {N_OBS}")

    # -----------------------------------------------------------------------
    # 1. Repeated sampling (Monte Carlo)
    # -----------------------------------------------------------------------
    print("\n--- 1. Repeated Sampling ---")
    print("Pretend we can draw fresh samples from the population many times.")

    diffs_repeat = []
    for _ in range(N_ITER):
        s1 = rng.normal(MU1, SIG1, N_OBS)
        s2 = rng.normal(MU2, SIG2, N_OBS)
        diffs_repeat.append(s1.mean() - s2.mean())

    diffs_repeat = np.array(diffs_repeat)
    theoretical_se = np.sqrt(SIG1**2 / N_OBS + SIG2**2 / N_OBS)

    print(f"  Mean of sampling distribution: {diffs_repeat.mean():.4f}  "
          f"(true: {MU1 - MU2})")
    print(f"  Std  of sampling distribution: {diffs_repeat.std():.4f}  "
          f"(theoretical SE: {theoretical_se:.4f})")

    # -----------------------------------------------------------------------
    # 2. Bootstrap
    # -----------------------------------------------------------------------
    print("\n--- 2. Bootstrap ---")
    print("Resample with replacement from the one sample we have.")

    diffs_boot = []
    for _ in range(N_ITER):
        b1 = rng.choice(sample1, N_OBS, replace=True)
        b2 = rng.choice(sample2, N_OBS, replace=True)
        diffs_boot.append(b1.mean() - b2.mean())

    diffs_boot = np.array(diffs_boot)
    print(f"  Mean of bootstrap distribution: {diffs_boot.mean():.4f}  "
          f"(observed diff: {observed_diff:.4f})")
    print(f"  Std  of bootstrap distribution: {diffs_boot.std():.4f}  "
          f"(true SE: {theoretical_se:.4f})")

    # -----------------------------------------------------------------------
    # 3. Permutation test
    # -----------------------------------------------------------------------
    print("\n--- 3. Permutation Test ---")
    print("Shuffle group labels to simulate: what if there were NO difference?")

    pooled = np.concatenate([sample1, sample2])
    diffs_perm = []
    for _ in range(N_ITER):
        shuffled = rng.permutation(pooled)
        diffs_perm.append(shuffled[:N_OBS].mean() - shuffled[N_OBS:].mean())

    diffs_perm = np.array(diffs_perm)
    print(f"  Mean of null distribution: {diffs_perm.mean():.4f}  (expect ≈ 0)")
    print(f"  Std  of null distribution: {diffs_perm.std():.4f}")

    # -----------------------------------------------------------------------
    # 4. Central Limit Theorem
    # -----------------------------------------------------------------------
    print("\n--- 4. Central Limit Theorem ---")
    print("Population: Exponential(λ=1)  — strongly non-normal")

    LAM     = 1.0
    TRUE_MU = 1 / LAM
    TRUE_SD = 1 / LAM
    sample_sizes = [2, 5, 15, 30, 100]
    N_CLT = 4000

    print(f"\n{'n':>6}  {'E[x̄]':>8}  {'Std(x̄)':>10}  {'Theoretical SE':>16}")
    clt = {}
    for n in sample_sizes:
        means_n = np.array([rng.exponential(1 / LAM, n).mean()
                            for _ in range(N_CLT)])
        clt[n] = means_n
        print(f"{n:>6}  {means_n.mean():>8.4f}  {means_n.std():>10.4f}  "
              f"{TRUE_SD / np.sqrt(n):>16.4f}")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    # Panel A: three sampling distributions
    true_diff = MU1 - MU2
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    configs = [
        (diffs_repeat, BLUE,   "Repeated Sampling"),
        (diffs_boot,   ORANGE, "Bootstrap"),
        (diffs_perm,   GREEN,  "Permutation (null)"),
    ]
    for ax, (diffs, col, title) in zip(axes, configs):
        ax.hist(diffs, bins=50, color=col, alpha=0.7, edgecolor="white", density=True)
        ax.axvline(diffs.mean(), color=RED,   lw=2,   ls="--", label="Dist Mean")
        ax.axvline(true_diff,    color=GREEN, lw=2,   ls="-",  label=f"True Diff ({true_diff:.1f})")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Difference in means  (x̄₁ − x̄₂)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    fig.suptitle("Sampling Distributions — Three Methods", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "part2_sampling_distributions.png"), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

    # Panel B: CLT convergence
    fig, axes = plt.subplots(1, len(sample_sizes), figsize=(14, 4), sharey=False)
    for ax, n in zip(axes, sample_sizes):
        se_n = TRUE_SD / np.sqrt(n)
        ax.hist(clt[n], bins=45, density=True, color=BLUE, alpha=0.65, edgecolor="white")
        x_r = np.linspace(TRUE_MU - 4.5 * se_n, TRUE_MU + 4.5 * se_n, 300)
        ax.plot(x_r, norm.pdf(x_r, TRUE_MU, se_n), color=RED, lw=2.2, label="Normal")
        ax.axvline(TRUE_MU, color=RED, lw=1.5, ls="--", alpha=0.5)
        ax.set_title(f"n = {n}", fontweight="bold")
        ax.set_xlabel("Sample mean x̄")
        if ax is axes[0]:
            ax.set_ylabel("Density")
        ax.legend(fontsize=7.5)

    fig.suptitle("Central Limit Theorem — Exponential population, varying n",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "part2_clt.png"), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


# ===========================================================================
# PART III — THE GERMAN TANK PROBLEM
# ===========================================================================

def part3():
    print("\n" + "=" * 60)
    print("PART III — THE GERMAN TANK PROBLEM")
    print("=" * 60)

    N_TANKS  = 237   # true total (the estimand — unknown to the estimator)
    M_SAMPLE = 20    # captured tanks (sample size)
    N_REPS   = 5000  # simulation repetitions

    population = list(range(1, N_TANKS + 1))

    # -----------------------------------------------------------------------
    # Estimator definitions
    # -----------------------------------------------------------------------
    def est_mle(s):
        """MLE: sample maximum."""
        return max(s)

    def est_mom(s):
        """MoM: 2·mean − 1."""
        return 2 * np.mean(s) - 1

    def est_mvue(s):
        """MVUE: x_max · (1 + 1/m) − 1."""
        return max(s) * (1 + 1 / len(s)) - 1

    def est_bayes(s):
        """Bayesian posterior mean under a flat (discrete uniform) prior."""
        xmax, m = max(s), len(s)
        return (xmax - 1) * (m - 1) / (m - 2) if m > 2 else float(xmax)

    estimators = {
        "MLE  (max)": est_mle,
        "MoM":        est_mom,
        "MVUE":       est_mvue,
        "Bayesian":   est_bayes,
    }

    # -----------------------------------------------------------------------
    # Run simulation
    # -----------------------------------------------------------------------
    print(f"\nTrue N (estimand) = {N_TANKS}")
    print(f"Sample size m     = {M_SAMPLE}")
    print(f"Repetitions       = {N_REPS}\n")

    results = {name: [] for name in estimators}
    for _ in range(N_REPS):
        sample = random.sample(population, M_SAMPLE)
        for name, fn in estimators.items():
            results[name].append(fn(sample))

    # -----------------------------------------------------------------------
    # Compute bias, variance, MSE
    # -----------------------------------------------------------------------
    print(f"{'Estimator':>14}  {'E[θ̂]':>8}  {'Bias':>8}  {'Variance':>10}  {'MSE':>10}")
    print("-" * 58)
    stats = {}
    for name, vals in results.items():
        arr  = np.array(vals)
        mean = arr.mean()
        bias = mean - N_TANKS
        var  = arr.var()
        mse  = ((arr - N_TANKS) ** 2).mean()
        stats[name] = dict(mean=mean, bias=bias, var=var, mse=mse)
        print(f"{name:>14}  {mean:>8.1f}  {bias:>+8.1f}  {var:>10.1f}  {mse:>10.1f}")

    print("\nNote: MSE = Bias² + Variance  (verify below)")
    for name, s in stats.items():
        check = s["bias"]**2 + s["var"]
        print(f"  {name:>14}:  Bias²+Var = {check:.1f},  MSE = {s['mse']:.1f}  ✓")

    # -----------------------------------------------------------------------
    # Single-sample demo
    # -----------------------------------------------------------------------
    demo = random.sample(population, M_SAMPLE)
    print(f"\nDemo — one analyst's sample (m={M_SAMPLE}):")
    print(f"  Observed serial numbers: {sorted(demo)}")
    print(f"  x_max = {max(demo)},  x_mean = {np.mean(demo):.1f}")
    for name, fn in estimators.items():
        print(f"  {name:>14} estimate: {fn(demo):.1f}")

    # -----------------------------------------------------------------------
    # Plot: four estimator distributions
    # -----------------------------------------------------------------------
    colors = [RED, BLUE, GREEN, PURPLE]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))

    for ax, (name, col) in zip(axes, zip(estimators, colors)):
        vals = np.array(results[name])
        s    = stats[name]
        ax.hist(vals, bins=45, color=col, alpha=0.70, edgecolor="white", density=True)
        ax.axvline(N_TANKS,    color="black", lw=2.5, ls="--", label=f"True N = {N_TANKS}")
        ax.axvline(s["mean"],  color=col,     lw=2,   ls=":",  label=f"E[θ̂] = {s['mean']:.0f}")
        ax.set_title(name, fontweight="bold", color=col)
        ax.set_xlabel("Estimate of N")
        if ax is axes[0]:
            ax.set_ylabel("Density")
        ax.text(0.5, 0.97,
                f"Bias = {s['bias']:+.1f}\nVar  = {s['var']:.0f}\nMSE  = {s['mse']:.0f}",
                ha="center", va="top", transform=ax.transAxes, fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col, alpha=0.85))
        ax.legend(fontsize=7.5, loc="lower left")

    fig.suptitle(
        f"German Tank Problem — Estimator Comparison  (N={N_TANKS}, m={M_SAMPLE})",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "part3_tank_estimators.png"), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


# ===========================================================================
# PART IV — THE OVERSELLING PROBLEM
# ===========================================================================

def part4():
    print("\n" + "=" * 60)
    print("PART IV — THE OVERSELLING PROBLEM")
    print("=" * 60)

    CAPACITY    = 400
    TRUE_P      = 0.05   # true no-show probability
    TICKET_PRICE = 100
    BUMP_COST   = 300
    N_HIST      = 500    # historical tickets to estimate p from

    # -----------------------------------------------------------------------
    # Stage 1 — Estimation: estimate p from historical data
    # -----------------------------------------------------------------------
    print("\n--- Stage 1: Estimation ---")
    print(f"Simulate {N_HIST} past tickets with true p_noshow = {TRUE_P}")

    historical = rng.binomial(1, TRUE_P, N_HIST)  # 1 = no-show
    n_noshows  = historical.sum()
    p_hat = n_noshows / N_HIST

    print(f"Observed no-shows: {n_noshows} / {N_HIST}")
    print(f"Estimated p̂ = {p_hat:.4f}  (true p = {TRUE_P})")

    # -----------------------------------------------------------------------
    # Stage 2 — Prediction: Binomial(n, p̂) for several ticket counts
    # -----------------------------------------------------------------------
    print("\n--- Stage 2: Prediction ---")
    ns_preview = [400, 410, 420]

    print(f"\n{'n tickets':>10}  {'E[no-shows]':>14}  {'P(no bumping)':>15}")
    for n in ns_preview:
        break_even = n - CAPACITY      # min no-shows to avoid bumping
        p_safe = binom.cdf(n - 1, n, 1 - p_hat)   # P(show-ups <= capacity)
        print(f"{n:>10}  {n * p_hat:>14.2f}  {p_safe:>15.4f}")

    # -----------------------------------------------------------------------
    # Stage 3 — Optimisation: find n that maximises expected revenue
    # -----------------------------------------------------------------------
    print("\n--- Stage 3: Optimisation ---")

    ticket_counts = np.arange(400, 501)

    def expected_revenue(n, p, capacity, price, bump):
        ev = 0.0
        for k in range(n + 1):
            prob   = binom.pmf(k, n, p)        # P(k no-shows)
            bumped = max((n - k) - capacity, 0)
            ev    += prob * (price * n - bump * bumped)
        return ev

    exp_revs = np.array([
        expected_revenue(n, p_hat, CAPACITY, TICKET_PRICE, BUMP_COST)
        for n in ticket_counts
    ])

    best_n   = ticket_counts[np.argmax(exp_revs)]
    best_rev = exp_revs.max()
    rev_400  = expected_revenue(CAPACITY, p_hat, CAPACITY, TICKET_PRICE, BUMP_COST)

    print(f"\nOptimal tickets to sell:   n* = {best_n}")
    print(f"Expected revenue at n*:    ${best_rev:,.2f}")
    print(f"Expected revenue at n=400: ${rev_400:,.2f}")
    print(f"Gain from overbooking:     ${best_rev - rev_400:,.2f}")

    # -----------------------------------------------------------------------
    # Monte Carlo verification at the optimal n
    # -----------------------------------------------------------------------
    print(f"\n[Monte Carlo] Simulating 10,000 flights at n={best_n}, p̂={p_hat:.4f}")
    sim_revs = []
    for _ in range(10_000):
        noshows = rng.binomial(best_n, p_hat)
        showups = best_n - noshows
        bumped  = max(showups - CAPACITY, 0)
        sim_revs.append(TICKET_PRICE * best_n - BUMP_COST * bumped)

    sim_revs = np.array(sim_revs)
    print(f"  Analytic expected revenue:  ${best_rev:,.2f}")
    print(f"  Simulated mean revenue:     ${sim_revs.mean():,.2f}")
    print(f"  Fraction of flights bumped: {(sim_revs < TICKET_PRICE * best_n).mean():.3f}")

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Right: expected revenue curve
    ax = axes[1]
    ax.scatter(ticket_counts, exp_revs, color=BLUE, s=18, alpha=0.8,
               label="Expected revenue")
    ax.scatter([best_n], [best_rev], color=RED, s=80, zorder=5,
               label=f"Optimal n = {best_n}")
    ax.axhline(best_rev, color=GRAY, ls=":", lw=1.2)
    ax.set_xlabel("Tickets sold (n)", fontsize=11)
    ax.set_ylabel("Expected revenue ($)", fontsize=11)
    ax.set_title("Stage 3 — Optimisation", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.text(best_n + 1.5, best_rev - 30,
            f"n*={best_n}\n${best_rev:,.0f}",
            fontsize=8.5, color=RED, va="top")

    # Left: Binomial PMF at optimal n — k on x-axis, coloured by revenue outcome
    ax2 = axes[0]
    import matplotlib.patches as mpatches

    k_all  = np.arange(0, best_n + 1)
    pmf    = binom.pmf(k_all, best_n, p_hat)
    mask   = pmf > 1e-5

    def rev_at_k(k):
        bumped = max((best_n - k) - CAPACITY, 0)
        return TICKET_PRICE * best_n - BUMP_COST * bumped

    k_vals     = k_all[mask]
    p_vals     = pmf[mask]
    rev_vals   = np.array([rev_at_k(int(k)) for k in k_vals])
    exp_rev_n  = float(np.sum(rev_vals * p_vals))
    break_even = best_n - CAPACITY   # min no-shows needed to avoid bumping

    for k, pk in zip(k_vals, p_vals):
        color = BLUE if k >= break_even else RED
        ax2.bar(k, pk, color=color, alpha=0.75, edgecolor="white", width=0.8)

    ax2.axvline(break_even - 0.5, color=GRAY, lw=1.2, ls=":",
                label=f"Break-even: {break_even} no-shows")
    ax2.axvline(best_n * p_hat, color=RED, lw=2, ls="--",
                label=f"E[X] = {best_n * p_hat:.1f}")

    ax2.set_xlabel("k  (no. no-shows)", fontsize=11)
    ax2.set_ylabel("P(X = k)", fontsize=11)
    ax2.set_title(f"Stage 2 — No-show distribution  (n={best_n})  |  E[Rev] = ${exp_rev_n:,.0f}",
                  fontsize=10, fontweight="bold")

    handles = [
        mpatches.Patch(color=BLUE, alpha=0.75, label="No bumping"),
        mpatches.Patch(color=RED,  alpha=0.75, label="Passengers bumped"),
        plt.Line2D([0], [0], color=RED, lw=2, ls="--", label=f"E[X] = {best_n * p_hat:.1f}"),
    ]
    ax2.legend(handles=handles, fontsize=8)

    fig.suptitle(
        f"Overselling Problem  —  capacity={CAPACITY}, "
        f"ticket=${TICKET_PRICE}, bump=${BUMP_COST}, p̂={p_hat:.4f}",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "part4_overselling.png"), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    part1()
    part2()
    part3()
    part4()
    print("\n" + "=" * 60)
    print("All parts complete.")
    print("=" * 60)
