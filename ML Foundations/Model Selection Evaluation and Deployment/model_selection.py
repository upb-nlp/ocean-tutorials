"""
Model Selection, Evaluation, and Deployment Readiness — companion tutorial.

Works on a controlled synthetic binary-classification problem
(make_classification: 2000 samples, 20 features, imbalanced ~15% positive,
with the first 8 features KNOWN to carry signal). Because we know the ground
truth about which features matter, the interpretability lab is verifiable.

Run all labs:
    python model_selection.py

Run a single lab:
    python model_selection.py --lab 1   # Cross-validation strategies
    python model_selection.py --lab 2   # Hyperparameter tuning
    python model_selection.py --lab 3   # Evaluation metrics in depth
    python model_selection.py --lab 4   # Bias-variance tradeoff
    python model_selection.py --lab 5   # Model interpretability
    python model_selection.py --lab 6   # Full model-selection pipeline
"""

import argparse
import os
import sys
import textwrap
import warnings

# Windows consoles default to cp1252, which cannot print the box-drawing chars.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUTPUT_DIR = "./outputs/model_selection"
SEED = 42


# ════════════════════════════════════════════════════════════════════════════
#  PACKAGE CHECK & SHARED UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def require(packages: list) -> None:
    import_names = {"scikit-learn": "sklearn"}
    missing = []
    for pkg in packages:
        try:
            __import__(import_names.get(pkg, pkg))
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"\n  Missing packages: {', '.join(missing)}")
        print(f"     Install with:  pip install {' '.join(missing)}\n")
        sys.exit(1)


def section(title: str) -> None:
    width = 70
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def subsection(title: str) -> None:
    print(f"\n  ── {title} " + "─" * max(0, 60 - len(title)))


def show_table(headers: list, rows: list, col_width: int = 18) -> None:
    fmt = "  " + "".join(f"{{:<{col_width}}}" for _ in headers)
    print(fmt.format(*headers))
    print("  " + "-" * (col_width * len(headers)))
    for row in rows:
        print(fmt.format(*[str(c)[: col_width - 1] for c in row]))


def get_plt():
    """pyplot with a non-interactive backend (save-to-file only)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save_fig(fig, name: str) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    get_plt().close(fig)
    print(f"    ✓ figure saved to {path}")


# Number of features that actually carry signal (informative + redundant).
N_INFORMATIVE = 6
N_REDUNDANT = 2
N_SIGNAL = N_INFORMATIVE + N_REDUNDANT   # first 8 columns


def make_data():
    """Controlled imbalanced binary problem with KNOWN signal features.

    make_classification with shuffle=False lays the columns out as
    [informative | redundant | noise], so we know feature 0..7 carry signal
    and 8..19 are pure noise — ground truth for the interpretability lab.
    """
    require(["scikit-learn", "numpy"])
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=2000, n_features=20,
        n_informative=N_INFORMATIVE, n_redundant=N_REDUNDANT, n_repeated=0,
        n_classes=2, weights=[0.85, 0.15], flip_y=0.02, class_sep=0.9,
        shuffle=False, random_state=SEED,
    )
    feature_names = [f"f{i:02d}" for i in range(X.shape[1])]
    return X, y, feature_names


# ════════════════════════════════════════════════════════════════════════════
#  LAB 1 — CROSS-VALIDATION STRATEGIES
# ════════════════════════════════════════════════════════════════════════════

def lab1_cross_validation():
    require(["scikit-learn", "numpy", "matplotlib"])
    import numpy as np
    from sklearn.model_selection import (
        KFold, StratifiedKFold, TimeSeriesSplit, cross_val_score)
    from sklearn.linear_model import LogisticRegression
    plt = get_plt()

    section("1 — CROSS-VALIDATION STRATEGIES")
    print(textwrap.dedent("""
      A single train/test split is one noisy number. Cross-validation reuses
      the data by rotating which part is held out, giving a mean ± spread that
      is far more trustworthy — IF the splitting scheme respects the structure
      of the data (class balance, time order, groups).
    """))

    X, y, _ = make_data()
    pos_rate = y.mean()
    print(f"  Dataset: {len(y)} samples, positive class = {pos_rate:.1%} (imbalanced)")

    # ── 1.1  Plain k-fold ────────────────────────────────────────────────────
    subsection("1.1  Plain k-fold")
    print("  Split into k equal folds; each takes a turn as the validation set.\n")
    model = LogisticRegression(max_iter=2000)
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(model, X, y, cv=kf, scoring="roc_auc")
    print(f"  5-fold ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"  Per-fold: {[round(float(s), 4) for s in scores]}")

    # ── 1.2  Stratified k-fold on imbalanced data ────────────────────────────
    subsection("1.2  Stratified k-fold — preserving class balance")
    print("  With a 15% positive rate, plain KFold can hand a fold very few")
    print("  positives just by chance. Stratified folds keep each fold's class")
    print("  ratio close to the whole dataset's.\n")

    def fold_pos_rates(cv):
        return [round(float(y[test].mean()), 3) for _, test in cv.split(X, y)]

    plain = KFold(n_splits=5, shuffle=True, random_state=SEED)
    strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    show_table(
        ["Scheme", "Positive rate in each of the 5 validation folds"],
        [
            ["KFold",           str(fold_pos_rates(plain))],
            ["StratifiedKFold", str(fold_pos_rates(strat))],
        ],
        col_width=18,
    )
    print(f"\n  Target rate = {pos_rate:.3f}. Stratified folds hug it far tighter →")
    print("  lower-variance, less biased estimates. Use it for ALL classification.")

    # ── 1.3  Time-series split ───────────────────────────────────────────────
    subsection("1.3  Time-series split — never train on the future")
    print(textwrap.dedent("""
      If samples are ordered in time (and the signal drifts), shuffling lets the
      model peek at future rows to predict the past — a leak that inflates CV
      scores. TimeSeriesSplit always trains on the PAST, validates on the FUTURE.
    """))

    # A drifting time series: the decision boundary slowly shifts over time.
    rng = np.random.default_rng(SEED)
    n = 1200
    t = np.linspace(0, 1, n)
    drift = 4 * t                                   # slowly moving threshold
    xt = rng.normal(size=n)
    yt = (xt + drift + rng.normal(scale=0.3, size=n) > 2).astype(int)
    Xt = np.c_[xt, t]                               # feature + time index

    shuffled = KFold(n_splits=5, shuffle=True, random_state=SEED)
    tss = TimeSeriesSplit(n_splits=5)
    s_shuf = cross_val_score(LogisticRegression(max_iter=1000), Xt, yt,
                             cv=shuffled, scoring="roc_auc")
    s_time = cross_val_score(LogisticRegression(max_iter=1000), Xt, yt,
                             cv=tss, scoring="roc_auc")
    print(f"  Shuffled KFold ROC-AUC:   {s_shuf.mean():.4f}  (optimistic — leaks the future)")
    print(f"  TimeSeriesSplit ROC-AUC:  {s_time.mean():.4f}  (honest — forward-only)")
    print("\n  The gap IS the leakage. On temporal data, the honest number is lower.")
    print("\n  Fold structure (train grows, validation always comes after it):")
    for i, (tr, te) in enumerate(tss.split(Xt), 1):
        print(f"    fold {i}: train=[0..{tr[-1]:>4}]  test=[{te[0]:>4}..{te[-1]:>4}]")

    # ── 1.4  Other schemes ───────────────────────────────────────────────────
    subsection("1.4  Choosing a scheme")
    show_table(
        ["Scheme", "Use when"],
        [
            ["KFold",            "i.i.d. regression / balanced data"],
            ["StratifiedKFold",  "classification (esp. imbalanced) — default"],
            ["TimeSeriesSplit",  "temporal / sequential data"],
            ["GroupKFold",       "repeated subjects (patients, users) — no leak"],
            ["LeaveOneOut",      "tiny datasets (expensive, high variance)"],
        ],
        col_width=18,
    )

    # ── figure: visualize the fold indices ───────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    _plot_cv_indices(KFold(5, shuffle=True, random_state=SEED), X, y, axes[0], "KFold")
    _plot_cv_indices(StratifiedKFold(5, shuffle=True, random_state=SEED), X, y,
                     axes[1], "StratifiedKFold")
    _plot_cv_indices(TimeSeriesSplit(5), X, y, axes[2], "TimeSeriesSplit")
    axes[2].set_xlabel("sample index")
    fig.suptitle("How each CV scheme assigns samples to folds", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "01_cv_strategies.png")


def _plot_cv_indices(cv, X, y, ax, title):
    import numpy as np
    n = len(y)
    for i, (tr, te) in enumerate(cv.split(X, y)):
        idx = np.zeros(n)
        idx[te] = 1                          # 1 = validation, 0 = train
        ax.scatter(range(n), [i + 0.5] * n, c=idx, marker="_", lw=8,
                   cmap="coolwarm", vmin=-0.2, vmax=1.2)
    ax.set_yticks([i + 0.5 for i in range(cv.get_n_splits())])
    ax.set_yticklabels([f"fold {i+1}" for i in range(cv.get_n_splits())])
    ax.set_title(f"{title}   (blue = train, red = validation)", fontsize=10, loc="left")
    ax.set_ylim(cv.get_n_splits(), 0)


# ════════════════════════════════════════════════════════════════════════════
#  LAB 2 — HYPERPARAMETER TUNING
# ════════════════════════════════════════════════════════════════════════════

def lab2_hyperparameter_tuning():
    require(["scikit-learn", "numpy", "matplotlib"])
    import numpy as np
    from scipy.stats import randint
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import (
        GridSearchCV, RandomizedSearchCV, StratifiedKFold)
    plt = get_plt()

    section("2 — HYPERPARAMETER TUNING")
    print(textwrap.dedent("""
      Model *parameters* are learned from data; *hyperparameters* (tree depth,
      #estimators, learning rate) are chosen by us. Tuning = search a space of
      hyperparameters, scoring each candidate by cross-validation, keep the best.
      Three strategies, increasing in sophistication: grid → random → Bayesian.
    """))

    X, y, _ = make_data()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    base = RandomForestClassifier(random_state=SEED, n_jobs=-1)

    # ── 2.1  Grid search ─────────────────────────────────────────────────────
    subsection("2.1  Grid search — exhaustive over a discrete grid")
    grid = {
        "n_estimators": [100, 200],
        "max_depth": [4, 8, None],
        "min_samples_leaf": [1, 5],
    }
    n_grid = np.prod([len(v) for v in grid.values()])
    print(f"  Grid = {int(n_grid)} combinations × {cv.get_n_splits()} folds "
          f"= {int(n_grid) * cv.get_n_splits()} fits.")
    print("  Exhaustive and reproducible, but cost explodes with each new axis.\n")
    gs = GridSearchCV(base, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gs.fit(X, y)
    print(f"  Best params: {gs.best_params_}")
    print(f"  Best CV ROC-AUC: {gs.best_score_:.4f}")

    # ── 2.2  Random search ───────────────────────────────────────────────────
    subsection("2.2  Random search — sample the space on a budget")
    print("  Sample n_iter random configs from (possibly continuous) ranges.")
    print("  Usually matches grid search at a fraction of the cost, because only")
    print("  a few hyperparameters actually matter (Bergstra & Bengio, 2012).\n")
    dist = {
        "n_estimators": randint(100, 400),
        "max_depth": randint(3, 20),
        "min_samples_leaf": randint(1, 10),
        "max_features": ["sqrt", "log2", None],
    }
    n_iter = 30
    rs = RandomizedSearchCV(base, dist, n_iter=n_iter, scoring="roc_auc",
                            cv=cv, n_jobs=-1, random_state=SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rs.fit(X, y)
    print(f"  {n_iter} configs × {cv.get_n_splits()} folds = {n_iter * cv.get_n_splits()} fits.")
    print(f"  Best params: {rs.best_params_}")
    print(f"  Best CV ROC-AUC: {rs.best_score_:.4f}")

    # running-max convergence of random search (order of sampling)
    rand_scores = rs.cv_results_["mean_test_score"]
    rand_curve = np.maximum.accumulate(rand_scores)

    # ── 2.3  Bayesian optimization (Optuna, optional) ────────────────────────
    subsection("2.3  Bayesian optimization — learn from past trials")
    print(textwrap.dedent("""
      Grid and random search are memoryless. Bayesian optimization builds a
      probabilistic model of score-vs-hyperparameters and spends each new trial
      where the expected improvement is highest — converging in fewer trials.
    """))
    bayes_curve = None
    bayes_best = None
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        from sklearn.model_selection import cross_val_score

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 400),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features",
                                                          ["sqrt", "log2", None]),
            }
            clf = RandomForestClassifier(random_state=SEED, n_jobs=-1, **params)
            return cross_val_score(clf, X, y, cv=cv, scoring="roc_auc",
                                   n_jobs=-1).mean()

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEED))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            study.optimize(objective, n_trials=n_iter, show_progress_bar=False)
        bayes_best = study.best_value
        vals = [t.value for t in study.trials if t.value is not None]
        bayes_curve = np.maximum.accumulate(vals)
        print(f"  {n_iter} TPE trials × {cv.get_n_splits()} folds.")
        print(f"  Best params: {study.best_params}")
        print(f"  Best CV ROC-AUC: {bayes_best:.4f}")
    except ImportError:
        print("  optuna not installed — skipping the Bayesian demo.")
        print("  Install with:  pip install optuna")

    # ── 2.4  Comparison ──────────────────────────────────────────────────────
    subsection("2.4  Strategy comparison")
    rows = [
        ["Grid search",   f"{int(n_grid)}", f"{gs.best_score_:.4f}", "exhaustive, explodes"],
        ["Random search", f"{n_iter}",      f"{rs.best_score_:.4f}", "cheap, strong baseline"],
    ]
    if bayes_best is not None:
        rows.append(["Bayesian (TPE)", f"{n_iter}", f"{bayes_best:.4f}",
                     "sample-efficient"])
    show_table(["Strategy", "#Configs", "Best CV AUC", "Character"], rows, col_width=17)

    # ── figure ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(rand_curve) + 1), rand_curve, marker="o", ms=4,
            label="Random search (best-so-far)")
    if bayes_curve is not None:
        ax.plot(range(1, len(bayes_curve) + 1), bayes_curve, marker="s", ms=4,
                label="Bayesian TPE (best-so-far)")
    ax.axhline(gs.best_score_, ls="--", color="grey", label="Grid-search best")
    ax.set_xlabel("trials evaluated")
    ax.set_ylabel("best CV ROC-AUC so far")
    ax.set_title("Search convergence: fewer trials to the same score")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "02_tuning_search.png")


# ════════════════════════════════════════════════════════════════════════════
#  LAB 3 — EVALUATION METRICS IN DEPTH
# ════════════════════════════════════════════════════════════════════════════

def lab3_metrics():
    require(["scikit-learn", "numpy", "matplotlib"])
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.calibration import CalibratedClassifierCV, calibration_curve
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        precision_recall_curve, roc_curve, auc, average_precision_score,
        precision_score, recall_score, f1_score, brier_score_loss,
        confusion_matrix)
    plt = get_plt()

    section("3 — EVALUATION METRICS IN DEPTH")
    print(textwrap.dedent("""
      Accuracy hides everything that matters on imbalanced data. This lab digs
      into the precision/recall tradeoff, ROC vs PR curves, and calibration —
      whether a predicted probability of 0.8 actually means 80% of the time.
    """))

    X, y, _ = make_data()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y)

    clf = GradientBoostingClassifier(random_state=SEED)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]

    # ── 3.1  Accuracy is a trap ──────────────────────────────────────────────
    subsection("3.1  Why accuracy misleads on imbalanced data")
    base_acc = max(y_te.mean(), 1 - y_te.mean())
    default_pred = (proba >= 0.5).astype(int)
    print(f"  Always-predict-majority accuracy: {base_acc:.3f}")
    print(f"  Model accuracy @0.5:              {(default_pred == y_te).mean():.3f}")
    print("  A tiny accuracy gain can still mean catching very few positives —")
    print("  which is the whole point on fraud / disease / churn problems.")

    # ── 3.2  Precision / recall tradeoff via threshold ───────────────────────
    subsection("3.2  The precision/recall tradeoff is a threshold choice")
    print(textwrap.dedent("""
      precision = TP / (TP + FP)   \"of those I flagged, how many were right?\"
      recall    = TP / (TP + FN)   \"of all real positives, how many did I catch?\"
      Raising the threshold buys precision at the cost of recall, and vice-versa.
    """))
    show_table(
        ["Threshold", "Precision", "Recall", "F1", "#Flagged"],
        [[f"{t:.2f}",
          f"{precision_score(y_te, proba >= t, zero_division=0):.3f}",
          f"{recall_score(y_te, proba >= t, zero_division=0):.3f}",
          f"{f1_score(y_te, proba >= t, zero_division=0):.3f}",
          int((proba >= t).sum())]
         for t in [0.1, 0.3, 0.5, 0.7, 0.9]],
        col_width=13,
    )

    # choose a threshold hitting a target recall (e.g. 0.90)
    prec, rec, thr = precision_recall_curve(y_te, proba)
    target_recall = 0.90
    ok = np.where(rec[:-1] >= target_recall)[0]
    if len(ok):
        best = ok[np.argmax(prec[:-1][ok])]
        print(f"\n  To guarantee ≥{target_recall:.0%} recall, set threshold ≈ {thr[best]:.3f}")
        print(f"  → precision there is {prec[best]:.3f}. This is a business decision,")
        print("    not a modelling one: pick the point on the curve you can live with.")

    # ── 3.3  ROC vs PR ───────────────────────────────────────────────────────
    subsection("3.3  ROC-AUC vs Average Precision (PR-AUC)")
    fpr, tpr, _ = roc_curve(y_te, proba)
    roc_auc = auc(fpr, tpr)
    ap = average_precision_score(y_te, proba)
    print(f"  ROC-AUC:           {roc_auc:.4f}")
    print(f"  Average precision: {ap:.4f}  (baseline = positive rate = {y_te.mean():.3f})")
    print(textwrap.dedent("""
      ROC-AUC can look flattering under heavy imbalance because true-negatives
      dominate the false-positive-rate. The PR curve ignores true-negatives, so
      Average Precision is the more honest headline for rare-positive problems.
    """))

    # ── 3.4  Calibration ─────────────────────────────────────────────────────
    subsection("3.4  Calibration — do the probabilities mean anything?")
    print("  Gaussian NB is famously over-confident. We compare it raw vs")
    print("  isotonic-calibrated using the reliability curve and Brier score.\n")
    nb = GaussianNB().fit(X_tr, y_tr)
    nb_p = nb.predict_proba(X_te)[:, 1]
    cal = CalibratedClassifierCV(GaussianNB(), method="isotonic", cv=5)
    cal.fit(X_tr, y_tr)
    cal_p = cal.predict_proba(X_te)[:, 1]
    show_table(
        ["Model", "Brier score (↓ better)"],
        [
            ["Gaussian NB (raw)",       f"{brier_score_loss(y_te, nb_p):.4f}"],
            ["Gaussian NB (isotonic)",  f"{brier_score_loss(y_te, cal_p):.4f}"],
            ["Gradient Boosting",       f"{brier_score_loss(y_te, proba):.4f}"],
        ],
        col_width=26,
    )
    print("\n  Lower Brier = probabilities closer to reality. Calibration matters")
    print("  whenever a downstream decision uses the probability, not just the label.")

    # ── figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    # ROC
    axes[0, 0].plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    axes[0, 0].plot([0, 1], [0, 1], "--", color="grey")
    axes[0, 0].set_title("ROC curve"); axes[0, 0].set_xlabel("FPR")
    axes[0, 0].set_ylabel("TPR"); axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)
    # PR
    axes[0, 1].plot(rec, prec, label=f"AP = {ap:.3f}")
    axes[0, 1].axhline(y_te.mean(), ls="--", color="grey", label="baseline")
    axes[0, 1].set_title("Precision-Recall curve"); axes[0, 1].set_xlabel("Recall")
    axes[0, 1].set_ylabel("Precision"); axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)
    # threshold sweep
    axes[1, 0].plot(thr, prec[:-1], label="precision")
    axes[1, 0].plot(thr, rec[:-1], label="recall")
    axes[1, 0].set_title("Precision & recall vs threshold")
    axes[1, 0].set_xlabel("decision threshold"); axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)
    # calibration
    for name, p in [("NB raw", nb_p), ("NB isotonic", cal_p), ("GBoost", proba)]:
        frac, mean_pred = calibration_curve(y_te, p, n_bins=10, strategy="quantile")
        axes[1, 1].plot(mean_pred, frac, marker="o", ms=4, label=name)
    axes[1, 1].plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    axes[1, 1].set_title("Calibration (reliability) curve")
    axes[1, 1].set_xlabel("predicted probability")
    axes[1, 1].set_ylabel("observed frequency")
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)
    fig.suptitle("Evaluation metrics in depth", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "03_metrics_in_depth.png")


# ════════════════════════════════════════════════════════════════════════════
#  LAB 4 — BIAS-VARIANCE TRADEOFF
# ════════════════════════════════════════════════════════════════════════════

def lab4_bias_variance():
    require(["scikit-learn", "numpy", "matplotlib"])
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import validation_curve, learning_curve
    plt = get_plt()

    section("4 — BIAS-VARIANCE TRADEOFF")
    print(textwrap.dedent("""
      Expected test error decomposes into three pieces:

        E[(y − f̂)²]  =  bias²   +   variance   +   irreducible noise
                        (too simple) (too sensitive) (can't be removed)

      Underfitting = high bias; overfitting = high variance. The art of model
      selection is finding the complexity that minimises their sum.
    """))

    # ── 4.1  Empirical decomposition on a regression toy ─────────────────────
    subsection("4.1  Measuring bias² and variance directly (polynomial fits)")
    print("  Fit polynomials of rising degree to a noisy sine, over many resampled")
    print("  training sets, and measure how the predictions vary at a fixed point.\n")

    rng = np.random.default_rng(SEED)
    true_fn = lambda x: np.sin(2 * np.pi * x)
    noise_sd = 0.25
    x_test = np.linspace(0, 1, 100)
    y_true = true_fn(x_test)

    degrees = [1, 3, 5, 9, 15]
    n_sets, n_train = 200, 50
    rows = []
    preds_by_degree = {}
    for d in degrees:
        preds = np.zeros((n_sets, len(x_test)))
        for b in range(n_sets):
            xb = rng.uniform(0, 1, n_train)
            yb = true_fn(xb) + rng.normal(0, noise_sd, n_train)
            # Polynomial.fit rescales x to [-1, 1] → well-conditioned even at
            # high degree (np.polyfit on raw [0,1] blows up for degree ≥ 9).
            series = np.polynomial.Polynomial.fit(xb, yb, d)
            preds[b] = series(x_test)
        mean_pred = preds.mean(axis=0)
        bias2 = np.mean((mean_pred - y_true) ** 2)
        var = np.mean(preds.var(axis=0))
        total = bias2 + var + noise_sd ** 2
        rows.append([f"degree {d:>2}", f"{bias2:.4f}", f"{var:.4f}",
                     f"{noise_sd**2:.4f}", f"{total:.4f}"])
        preds_by_degree[d] = (mean_pred, preds)
    show_table(["Complexity", "Bias²", "Variance", "Noise", "≈ Total error"],
               rows, col_width=14)
    print("\n  Bias falls and variance rises with degree — total error is U-shaped.")

    # ── 4.2  Validation curve on the real classifier ─────────────────────────
    subsection("4.2  Validation curve — the U-shape on real data")
    X, y, _ = make_data()
    depths = [1, 2, 3, 5, 8, 12, 20, None]
    depth_vals = [d if d is not None else 30 for d in depths]
    train_sc, val_sc = validation_curve(
        DecisionTreeClassifier(random_state=SEED), X, y,
        param_name="max_depth", param_range=depth_vals,
        cv=5, scoring="roc_auc", n_jobs=-1)
    print("  Decision tree, sweeping max_depth (train vs 5-fold CV ROC-AUC):\n")
    show_table(
        ["max_depth", "Train AUC", "CV AUC", "Gap (overfit)"],
        [[str(d), f"{train_sc[i].mean():.3f}", f"{val_sc[i].mean():.3f}",
          f"{train_sc[i].mean() - val_sc[i].mean():+.3f}"]
         for i, d in enumerate(depths)],
        col_width=14,
    )
    print("\n  Shallow → both scores low (high bias). Deep → train≈1.0 but CV drops")
    print("  and the gap widens (high variance). Best generalisation is in between.")

    # ── 4.3  Learning curve — does more data help? ───────────────────────────
    subsection("4.3  Learning curve — more data vs more model")
    sizes, tr_lc, val_lc = learning_curve(
        RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
        X, y, train_sizes=np.linspace(0.1, 1.0, 6), cv=5,
        scoring="roc_auc", n_jobs=-1, shuffle=True, random_state=SEED)
    print("  Random forest, ROC-AUC as the training set grows:\n")
    show_table(
        ["Train size", "Train AUC", "CV AUC"],
        [[int(sizes[i]), f"{tr_lc[i].mean():.3f}", f"{val_lc[i].mean():.3f}"]
         for i in range(len(sizes))],
        col_width=14,
    )
    print(textwrap.dedent("""
      Reading learning curves:
      • Curves converge LOW & together  → high bias: add features / complexity.
      • A wide, persistent GAP          → high variance: add data / regularise.
    """))

    # ── figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    # (a) fits at three complexities
    for d, color in [(1, "#1f77b4"), (5, "#2ca02c"), (15, "#d62728")]:
        mean_pred, _ = preds_by_degree[d]
        axes[0].plot(x_test, mean_pred, color=color, label=f"degree {d}")
    axes[0].plot(x_test, y_true, "k--", lw=2, label="truth")
    axes[0].set_ylim(-1.8, 1.8)   # high-degree fits overshoot at the edges
    axes[0].set_title("(a) Under- vs over-fitting")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    # (b) validation curve
    tr_mean, val_mean = train_sc.mean(1), val_sc.mean(1)
    axes[1].plot(depth_vals, tr_mean, marker="o", label="train")
    axes[1].plot(depth_vals, val_mean, marker="s", label="CV")
    axes[1].fill_between(depth_vals, val_mean, tr_mean, alpha=0.15, color="red")
    axes[1].set_title("(b) Validation curve (tree depth)")
    axes[1].set_xlabel("max_depth"); axes[1].set_ylabel("ROC-AUC")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    # (c) learning curve
    axes[2].plot(sizes, tr_lc.mean(1), marker="o", label="train")
    axes[2].plot(sizes, val_lc.mean(1), marker="s", label="CV")
    axes[2].set_title("(c) Learning curve (random forest)")
    axes[2].set_xlabel("training samples"); axes[2].set_ylabel("ROC-AUC")
    axes[2].legend(); axes[2].grid(alpha=0.3)
    fig.suptitle("Bias-variance tradeoff", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "04_bias_variance.png")


# ════════════════════════════════════════════════════════════════════════════
#  LAB 5 — MODEL INTERPRETABILITY
# ════════════════════════════════════════════════════════════════════════════

def lab5_interpretability():
    require(["scikit-learn", "numpy", "matplotlib"])
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance, PartialDependenceDisplay
    from sklearn.model_selection import train_test_split
    plt = get_plt()

    section("5 — MODEL INTERPRETABILITY")
    print(textwrap.dedent(f"""
      A model you can't explain is a model you can't trust, debug, or defend to
      a regulator. We know the ground truth here: features f00–f{N_SIGNAL-1:02d} carry the
      signal, f{N_SIGNAL:02d}–f19 are pure noise — so we can check whether each method
      actually recovers the real drivers.
    """))

    X, y, names = make_data()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y)
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    rf.fit(X_tr, y_tr)

    def top(scores, k=8):
        order = np.argsort(scores)[::-1][:k]
        return [(names[i], round(float(scores[i]), 4)) for i in order]

    # ── 5.1  Impurity-based importance (built-in, but biased) ────────────────
    subsection("5.1  Built-in (impurity) feature importance")
    imp = rf.feature_importances_
    print("  Fast and free, but BIASED toward high-cardinality / continuous")
    print("  features and computed on TRAINING data — read it with caution.\n")
    for name, val in top(imp):
        flag = "signal" if int(name[1:]) < N_SIGNAL else "NOISE ⚠"
        print(f"    {name}  {val:.4f}   ({flag})")

    # ── 5.2  Permutation importance (model-agnostic, on test) ────────────────
    subsection("5.2  Permutation importance (model-agnostic, held-out)")
    print("  Shuffle one feature, measure the drop in TEST score. Works for any")
    print("  model and reflects real predictive value, not training artefacts.\n")
    perm = permutation_importance(rf, X_te, y_te, n_repeats=10,
                                  random_state=SEED, scoring="roc_auc", n_jobs=-1)
    for name, val in top(perm.importances_mean):
        flag = "signal" if int(name[1:]) < N_SIGNAL else "NOISE ⚠"
        print(f"    {name}  {val:.4f}   ({flag})")

    imp_hits = sum(1 for n, _ in top(imp) if int(n[1:]) < N_SIGNAL)
    perm_hits = sum(1 for n, _ in top(perm.importances_mean) if int(n[1:]) < N_SIGNAL)
    print(f"\n  Signal features in the top 8 — impurity: {imp_hits}/8, "
          f"permutation: {perm_hits}/8.")

    # ── 5.3  SHAP (optional) ─────────────────────────────────────────────────
    subsection("5.3  SHAP — unified local + global attributions")
    print(textwrap.dedent("""
      SHAP assigns each feature a signed contribution to EACH individual
      prediction (game-theoretic Shapley values), then averages |value| for a
      global view that is consistent and locally faithful.
    """))
    shap_mean = None
    try:
        import shap
        expl = shap.TreeExplainer(rf)
        sample = X_te[:200]
        sv = expl.shap_values(sample)
        # Normalise across shap versions: list per class, or (n, feat, class).
        if isinstance(sv, list):
            sv = sv[1]
        elif np.ndim(sv) == 3:
            sv = sv[:, :, 1]
        shap_mean = np.abs(sv).mean(axis=0)
        print("  Mean |SHAP value| (global importance, top 8):\n")
        for name, val in top(shap_mean):
            flag = "signal" if int(name[1:]) < N_SIGNAL else "NOISE ⚠"
            print(f"    {name}  {val:.4f}   ({flag})")
    except ImportError:
        print("  shap not installed — skipping (permutation importance covers the")
        print("  same global need). Install with:  pip install shap")
    except Exception as e:                       # shap/numpy version friction
        print(f"  shap present but errored ({type(e).__name__}); skipping SHAP plot.")

    # ── 5.4  Partial dependence — the shape of an effect ─────────────────────
    subsection("5.4  Partial dependence — HOW a feature moves the prediction")
    top_feat = int(np.argmax(perm.importances_mean))
    print(f"  Most important feature by permutation: {names[top_feat]}.")
    print("  Partial dependence shows the average predicted probability as that")
    print("  feature varies — direction and shape, not just magnitude.")

    # ── figure ───────────────────────────────────────────────────────────────
    ncols = 3 if shap_mean is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5.5))
    order = np.argsort(perm.importances_mean)[::-1][:8]
    colors = ["#2ca02c" if i < N_SIGNAL else "#d62728" for i in order]
    axes[0].barh(range(8), perm.importances_mean[order][::-1],
                 color=colors[::-1])
    axes[0].set_yticks(range(8))
    axes[0].set_yticklabels([names[i] for i in order[::-1]])
    axes[0].set_title("Permutation importance\n(green=signal, red=noise)")
    axes[0].set_xlabel("mean AUC drop")
    PartialDependenceDisplay.from_estimator(rf, X_te, [top_feat],
                                            feature_names=names, ax=axes[1])
    axes[1].set_title(f"Partial dependence — {names[top_feat]}")
    if shap_mean is not None:
        s_order = np.argsort(shap_mean)[::-1][:8]
        s_colors = ["#2ca02c" if i < N_SIGNAL else "#d62728" for i in s_order]
        axes[2].barh(range(8), shap_mean[s_order][::-1], color=s_colors[::-1])
        axes[2].set_yticks(range(8))
        axes[2].set_yticklabels([names[i] for i in s_order[::-1]])
        axes[2].set_title("Mean |SHAP value|")
        axes[2].set_xlabel("mean |SHAP|")
    fig.suptitle("Model interpretability", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "05_interpretability.png")


# ════════════════════════════════════════════════════════════════════════════
#  LAB 6 — FULL MODEL-SELECTION PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def lab6_pipeline():
    require(["scikit-learn", "numpy", "matplotlib"])
    import time
    import pickle
    import numpy as np
    from scipy.stats import randint, loguniform
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import (
        train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score)
    from sklearn.inspection import permutation_importance
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, f1_score, brier_score_loss,
        roc_curve, precision_recall_curve, confusion_matrix, classification_report)
    plt = get_plt()

    section("6 — FULL MODEL-SELECTION PIPELINE")
    print(textwrap.dedent("""
      Everything from Labs 1–5 combined into the real workflow:

        split → tune each candidate (random search, stratified CV)
              → compare by cross-validation → pick winner
              → evaluate ONCE on the untouched test set
              → interpret → check deployment readiness → select.
    """))

    X, y, names = make_data()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    print(f"  Train: {len(y_tr)}   Test (held out): {len(y_te)}   "
          f"positive rate: {y.mean():.1%}")

    # ── 6.1  Candidates + search spaces ──────────────────────────────────────
    subsection("6.1  Candidate models + hyperparameter spaces")
    candidates = {
        "Logistic Regression": (
            Pipeline([("scale", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=2000))]),
            {"clf__C": loguniform(1e-2, 1e2)},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=SEED, n_jobs=-1),
            {"n_estimators": randint(100, 400), "max_depth": randint(3, 20),
             "min_samples_leaf": randint(1, 10)},
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=SEED),
            {"n_estimators": randint(100, 300), "max_depth": randint(2, 5),
             "learning_rate": loguniform(1e-2, 3e-1)},
        ),
    }
    print(f"  {len(candidates)} model families, tuned by RandomizedSearchCV")
    print("  (scoring = average_precision, the right metric for imbalance).")

    # ── 6.2  Tune each candidate ─────────────────────────────────────────────
    subsection("6.2  Tune each candidate (random search on TRAIN only)")
    tuned, rows = {}, []
    for name, (est, space) in candidates.items():
        rs = RandomizedSearchCV(est, space, n_iter=20, scoring="average_precision",
                                cv=cv, n_jobs=-1, random_state=SEED)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rs.fit(X_tr, y_tr)
        tuned[name] = rs.best_estimator_
        rows.append([name, f"{rs.best_score_:.4f}"])
    show_table(["Model", "Best CV avg-precision"], rows, col_width=24)

    # ── 6.3  Compare tuned models by cross-validation → pick ─────────────────
    subsection("6.3  Compare tuned models by cross-validation")
    cmp_rows, cv_means = [], {}
    for name, est in tuned.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ap = cross_val_score(est, X_tr, y_tr, cv=cv,
                                 scoring="average_precision", n_jobs=-1)
            roc = cross_val_score(est, X_tr, y_tr, cv=cv,
                                  scoring="roc_auc", n_jobs=-1)
        cv_means[name] = ap.mean()
        cmp_rows.append([name, f"{ap.mean():.4f} ± {ap.std():.3f}",
                         f"{roc.mean():.4f}"])
    show_table(["Model", "CV avg-precision", "CV ROC-AUC"], cmp_rows, col_width=24)
    winner = max(cv_means, key=cv_means.get)
    best_model = tuned[winner]
    print(f"\n  ► Selected: {winner}  (highest CV average precision)")

    # ── 6.4  Final, one-shot test-set evaluation ─────────────────────────────
    subsection("6.4  Final evaluation on the untouched test set")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best_model.fit(X_tr, y_tr)
    proba = best_model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    print(f"  ROC-AUC:           {roc_auc_score(y_te, proba):.4f}")
    print(f"  Average precision: {average_precision_score(y_te, proba):.4f}")
    print(f"  F1 @0.5:           {f1_score(y_te, pred):.4f}")
    print(f"  Brier score:       {brier_score_loss(y_te, proba):.4f}")
    print("\n" + textwrap.indent(
        classification_report(y_te, pred, target_names=["negative", "positive"],
                              zero_division=0), "  "))

    # ── 6.5  Interpret the selected model ────────────────────────────────────
    subsection("6.5  Interpret the selected model (permutation importance)")
    perm = permutation_importance(best_model, X_te, y_te, n_repeats=10,
                                  random_state=SEED, scoring="average_precision",
                                  n_jobs=-1)
    order = np.argsort(perm.importances_mean)[::-1][:6]
    for i in order:
        flag = "signal" if i < N_SIGNAL else "NOISE ⚠"
        print(f"    {names[i]}  {perm.importances_mean[i]:.4f}   ({flag})")

    # ── 6.6  Deployment readiness ────────────────────────────────────────────
    subsection("6.6  Deployment readiness checks")
    blob = pickle.dumps(best_model)
    t0 = time.perf_counter()
    for _ in range(5):
        best_model.predict_proba(X_te)
    latency_ms = (time.perf_counter() - t0) / 5 / len(X_te) * 1e3
    frac, mean_pred = calibration_curve(y_te, proba, n_bins=8, strategy="quantile")
    ece = float(np.mean(np.abs(frac - mean_pred)))
    show_table(
        ["Check", "Value", "Verdict"],
        [
            ["Serialized size", f"{len(blob)/1024:.1f} KB",
             "ok" if len(blob) < 5_000_000 else "large"],
            ["Latency / sample", f"{latency_ms:.3f} ms",
             "ok" if latency_ms < 1 else "review"],
            ["Calibration error", f"{ece:.3f}",
             "ok" if ece < 0.1 else "calibrate"],
            ["Test avg-precision", f"{average_precision_score(y_te, proba):.3f}",
             "vs baseline " + f"{y_te.mean():.3f}"],
        ],
        col_width=20,
    )
    print(textwrap.dedent("""
      A model is deployment-ready only when, beyond accuracy, it is: fast enough,
      small enough, well-calibrated if probabilities are consumed, monitored for
      input drift, reproducible (pinned data + seed + code), and documented. Ship
      the whole Pipeline (preprocessing + model) as one artefact — never a bare
      estimator that expects features someone else has to remember to scale.
    """))

    # ── figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    # model comparison
    mods = list(cv_means)
    axes[0, 0].barh(range(len(mods)), [cv_means[m] for m in mods],
                    color=["#d62728" if m == winner else "#4c72b0" for m in mods])
    axes[0, 0].set_yticks(range(len(mods))); axes[0, 0].set_yticklabels(mods)
    axes[0, 0].set_title(f"CV avg-precision (winner: {winner})")
    axes[0, 0].set_xlabel("average precision")
    # ROC + PR of winner
    fpr, tpr, _ = roc_curve(y_te, proba)
    axes[0, 1].plot(fpr, tpr, label=f"ROC AUC={roc_auc_score(y_te, proba):.3f}")
    prec, rec, _ = precision_recall_curve(y_te, proba)
    axes[0, 1].plot(rec, prec, label=f"PR AP={average_precision_score(y_te, proba):.3f}")
    axes[0, 1].plot([0, 1], [1, 0], "--", color="grey", lw=0.8)
    axes[0, 1].set_title("Test ROC & PR curves"); axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    # calibration
    axes[1, 0].plot(mean_pred, frac, marker="o", label=f"ECE={ece:.3f}")
    axes[1, 0].plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    axes[1, 0].set_title("Calibration of selected model")
    axes[1, 0].set_xlabel("predicted prob"); axes[1, 0].set_ylabel("observed freq")
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)
    # importance
    ord6 = order[::-1]
    axes[1, 1].barh(range(len(ord6)), perm.importances_mean[ord6],
                    color=["#2ca02c" if i < N_SIGNAL else "#d62728" for i in ord6])
    axes[1, 1].set_yticks(range(len(ord6)))
    axes[1, 1].set_yticklabels([names[i] for i in ord6])
    axes[1, 1].set_title("Permutation importance (green=signal)")
    axes[1, 1].set_xlabel("avg-precision drop")
    fig.suptitle(f"Model-selection pipeline (selected: {winner})", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "06_model_selection_pipeline.png")


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Model Selection, Evaluation & Deployment Readiness Tutorial")
    parser.add_argument(
        "--lab", type=int, choices=[1, 2, 3, 4, 5, 6],
        help="Run a specific lab (1=Cross-validation, 2=Tuning, 3=Metrics, "
             "4=Bias-variance, 5=Interpretability, 6=Full pipeline)")
    args = parser.parse_args()

    print("\n" + "█" * 70)
    print("  MODEL SELECTION, EVALUATION & DEPLOYMENT READINESS  ")
    print("█" * 70)
    print("""
  Labs:
    1 → Cross-validation strategies      (k-fold, stratified, time-series)
    2 → Hyperparameter tuning            (grid, random, Bayesian/Optuna)
    3 → Evaluation metrics in depth      (P/R tradeoff, ROC/PR, calibration)
    4 → Bias-variance tradeoff           (decomposition, validation & learning curves)
    5 → Model interpretability           (impurity, permutation, SHAP, PDP)
    6 → Full model-selection pipeline    (tune → compare → evaluate → interpret → ship)

  Every lab saves a figure to ./outputs/model_selection/
    """)

    labs = {
        1: lab1_cross_validation,
        2: lab2_hyperparameter_tuning,
        3: lab3_metrics,
        4: lab4_bias_variance,
        5: lab5_interpretability,
        6: lab6_pipeline,
    }

    if args.lab is not None:
        labs[args.lab]()
    else:
        for fn in labs.values():
            fn()

    print(f"\n  All figures saved under {OUTPUT_DIR}/\n")


if __name__ == "__main__":
    main()
