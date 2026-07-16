"""
Supervised Learning: Regression — companion tutorial script.

Trains and compares a battery of regressors on the California Housing dataset
(20 640 samples, 8 features, continuous target = median house value of a
district).  Lab 2 also reproduces the polynomial-curve-fitting demo from the
lecture's Colab notebook on the synthetic sqrt(x)*sin(x) signal.

Run all labs:
    python supervised_regression.py

Run a single lab:
    python supervised_regression.py --lab 1   # Data, splits, evaluation
    python supervised_regression.py --lab 2   # Linear & Polynomial regression
    python supervised_regression.py --lab 3   # Ridge, Lasso, Elastic Net
    python supervised_regression.py --lab 4   # Decision Tree, RF, Gradient Boosting
    python supervised_regression.py --lab 5   # Final benchmark + visualisations
"""

import argparse
import os
import sys
import textwrap
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# Force UTF-8 stdout/stderr on Windows (cp1252 default chokes on box-drawing chars)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ════════════════════════════════════════════════════════════════════════════
#  PACKAGE CHECK
# ════════════════════════════════════════════════════════════════════════════

def require(packages: List[str]) -> None:
    missing = []
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"\n  Missing packages: {', '.join(missing)}")
        print(f"     Install with:  pip install {' '.join(missing)}\n")
        sys.exit(1)


# ════════════════════════════════════════════════════════════════════════════
#  SHARED UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def section(title: str) -> None:
    width = 70
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def subsection(title: str) -> None:
    print(f"\n  ── {title} " + "─" * max(0, 60 - len(title)))


def show_table(headers: List[str], rows: List[List], col_width: int = 18) -> None:
    fmt = "  " + "".join(f"{{:<{col_width}}}" for _ in headers)
    print(fmt.format(*headers))
    print("  " + "-" * (col_width * len(headers)))
    for row in rows:
        print(fmt.format(*[str(c)[: col_width - 1] for c in row]))


OUTPUT_DIR = "./outputs/regression"
SEED = 42


# ════════════════════════════════════════════════════════════════════════════
#  DATA LOADING & PREPROCESSING
# ════════════════════════════════════════════════════════════════════════════

def load_and_prep_data():
    """Load California Housing, split 60/20/20 and standardize."""
    require(["sklearn", "numpy"])

    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    data = fetch_california_housing()
    X, y = data.data, data.target  # y is median house value in 100k USD

    # 60 % train, 20 % validation, 20 % test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=SEED,
    )

    # Standardize using only train statistics
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Also keep the full set scaled for cross-validation later
    X_full_s = StandardScaler().fit_transform(X)

    return {
        "train":         (X_train_s, y_train),
        "val":           (X_val_s,   y_val),
        "test":          (X_test_s,  y_test),
        "full":          (X_full_s,  y),
        "feature_names": list(data.feature_names),
        "target_name":   "MedHouseVal (100k USD)",
    }


def evaluate_model(model, X, y) -> Dict[str, float]:
    """Return RMSE, MAE, R² of a fitted model on (X, y)."""
    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    preds = model.predict(X)
    mse = float(mean_squared_error(y, preds))
    return {
        "RMSE": round(float(np.sqrt(mse)),               4),
        "MAE":  round(float(mean_absolute_error(y, preds)), 4),
        "R2":   round(float(r2_score(y, preds)),         4),
    }


# ════════════════════════════════════════════════════════════════════════════
#  LAB 1 — DATA, SPLITS, EVALUATION METRICS
# ════════════════════════════════════════════════════════════════════════════

def lab1_basics():
    section("1 — DATA, SPLITS & EVALUATION METRICS")
    print(textwrap.dedent("""
      Dataset: California Housing
        - 20 640 samples
        - 8 numerical features (MedInc, HouseAge, AveRooms, …)
        - Continuous target: median house value of a district (in 100k USD)

      We split 60 % / 20 % / 20 % (train / val / test).
      Features are standardized using the training set only.
    """))

    require(["numpy"])
    import numpy as np

    dataset = load_and_prep_data()
    X_train, y_train = dataset["train"]
    X_val,   y_val   = dataset["val"]
    X_test,  y_test  = dataset["test"]

    subsection("1.1  Set sizes")
    show_table(
        ["Set", "Size", "y mean", "y std"],
        [
            ["Train",      len(y_train), f"{y_train.mean():.3f}", f"{y_train.std():.3f}"],
            ["Validation", len(y_val),   f"{y_val.mean():.3f}",   f"{y_val.std():.3f}"],
            ["Test",       len(y_test),  f"{y_test.mean():.3f}",  f"{y_test.std():.3f}"],
        ],
    )

    subsection("1.2  A baseline — always predict the training mean")
    mean_pred = float(y_train.mean())
    baseline_preds = np.full_like(y_test, mean_pred)
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    rmse = float(np.sqrt(mean_squared_error(y_test, baseline_preds)))
    mae  = float(mean_absolute_error(y_test, baseline_preds))
    r2   = float(r2_score(y_test, baseline_preds))
    print(f"  Constant prediction:   {mean_pred:.4f}")
    print(f"  Baseline test RMSE:    {rmse:.4f}")
    print(f"  Baseline test MAE:     {mae:.4f}")
    print(f"  Baseline test R²:      {r2:.4f}   (≈ 0 by construction)")
    print("  Any useful regressor must beat this.")

    subsection("1.3  Why we report several metrics")
    print(textwrap.dedent("""
      RMSE   — squared errors → penalises large mistakes more
      MAE    — absolute errors → robust to outliers, same units as y
      R²     — fraction of variance explained; 1 = perfect, 0 = mean baseline
      MAPE   — relative error; great for forecasting, breaks when y ≈ 0

      A model that wins on RMSE but loses on MAE is biased toward
      avoiding *catastrophic* errors at the cost of many small ones.
    """))


# ════════════════════════════════════════════════════════════════════════════
#  LAB 2 — LINEAR AND POLYNOMIAL REGRESSION
# ════════════════════════════════════════════════════════════════════════════

def lab2_linear_polynomial():
    section("2 — LINEAR AND POLYNOMIAL REGRESSION")

    require(["numpy", "matplotlib"])
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.metrics import mean_squared_error

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 2.1  Plain linear regression on California Housing ──────────────────
    subsection("2.1  Linear regression on California Housing")
    dataset = load_and_prep_data()
    X_train, y_train = dataset["train"]
    X_val,   y_val   = dataset["val"]

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    m_tr = evaluate_model(lr, X_train, y_train)
    m_va = evaluate_model(lr, X_val,   y_val)
    print(f"  Train  →  RMSE={m_tr['RMSE']}  MAE={m_tr['MAE']}  R²={m_tr['R2']}")
    print(f"  Val    →  RMSE={m_va['RMSE']}  MAE={m_va['MAE']}  R²={m_va['R2']}")

    print("\n  Learned coefficients (standardised features):")
    show_table(
        ["Feature", "Weight"],
        [[name, f"{w:+.4f}"] for name, w in zip(dataset["feature_names"], lr.coef_)],
        col_width=22,
    )
    print(f"  Bias (intercept): {lr.intercept_:+.4f}")

    # ── 2.2  Synthetic polynomial demo ──────
    subsection("2.2  Polynomial regression on sqrt(x)*sin(x)")
    print("  Reproduces the lecture's Colab example with np.vander.")
    print("  Fits LinearRegression and Ridge on degrees [1, 4, 8, 12, 16].\n")

    rng = np.random.RandomState(SEED)

    def f(x, noise_amount):
        y = np.sqrt(x) * np.sin(x)
        noise = rng.normal(0, 1, len(x))
        return y + noise_amount * noise

    X = np.linspace(0, 20, 100)
    y = f(X, noise_amount=0.5)

    degrees = [1, 4, 8, 12, 16]
    fig, axes = plt.subplots(1, len(degrees), figsize=(5 * len(degrees), 5),
                             sharey=True)

    print("  Degree |   Linear-MSE   |   Ridge-MSE")
    print("  -------|----------------|---------------")
    for ax, degree in zip(axes, degrees):
        X_plot = np.linspace(0, 20, 100)
        y_plot = f(X_plot, noise_amount=0)  # ground truth, no noise

        # Ridge (polynomial) — suppress ill-conditioning warnings for high degrees
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kr = Ridge()
            kr.fit(np.vander(X, degree + 1), y)
            y_ridge = kr.predict(np.vander(X_plot, degree + 1))
            mse_ridge = mean_squared_error(y_plot, y_ridge)

            # Plain linear regression on polynomial features
            lr_poly = LinearRegression()
            lr_poly.fit(np.vander(X, degree + 1), y)
            y_lin = lr_poly.predict(np.vander(X_plot, degree + 1))
            mse_lin = mean_squared_error(y_plot, y_lin)

        print(f"  {degree:>6} | {mse_lin:>14.3f} | {mse_ridge:>13.3f}")

        ax.scatter(X, y, s=14, alpha=0.6, label="noisy samples")
        ax.plot(X_plot, y_plot, color="gold", linewidth=2, label="ground truth")
        ax.plot(X_plot, y_lin,  color="red",  linewidth=1.6, label="Linear Reg")
        ax.plot(X_plot, y_ridge, color="green", linewidth=1.6, label="Ridge")
        ax.set_title(f"degree = {degree}")
        ax.set_xlabel("X")
        ax.set_ylim(-5, 5)
        if ax is axes[0]:
            ax.set_ylabel("y")
        ax.legend(loc="lower left", fontsize=8)

    plt.suptitle("Polynomial fits on  y = √x · sin(x) + noise", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "polynomial_fits.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  polynomial_fits.png → {out}")

    print(textwrap.dedent("""
      Observations:
        - degree = 1 underfits — a straight line cannot follow sqrt(x)·sin(x).
        - degree = 4 tracks the trend reasonably.
        - degree = 8 / 12 fit very well in the middle.
        - degree = 16 oscillates wildly near the edges (Runge phenomenon).
        - Ridge dampens the oscillations vs plain LinearRegression.
    """))


# ════════════════════════════════════════════════════════════════════════════
#  LAB 3 — REGULARISATION (RIDGE / LASSO / ELASTIC NET)
# ════════════════════════════════════════════════════════════════════════════

def lab3_regularization():
    section("3 — REGULARISATION: RIDGE, LASSO, ELASTIC NET")

    require(["numpy", "matplotlib"])
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset = load_and_prep_data()
    X_train, y_train = dataset["train"]
    X_val,   y_val   = dataset["val"]

    # ── 3.1  α sweep on Ridge / Lasso / Elastic Net ─────────────────────────
    subsection("3.1  α sweep on California Housing (polynomial degree 2)")
    print("  We build degree-2 polynomial features (8 → 45 features),")
    print("  then fit Ridge / Lasso / ElasticNet for a range of α.\n")

    alphas = np.logspace(-3, 3, 13)
    methods = [
        ("Ridge",       lambda a: Ridge(alpha=a)),
        ("Lasso",       lambda a: Lasso(alpha=a, max_iter=20000)),
        ("ElasticNet",  lambda a: ElasticNet(alpha=a, l1_ratio=0.5, max_iter=20000)),
    ]

    sweep_results = {name: [] for name, _ in methods}
    for name, make in methods:
        for a in alphas:
            pipe = make_pipeline(
                PolynomialFeatures(degree=2, include_bias=False),
                make(a),
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe.fit(X_train, y_train)
            m = evaluate_model(pipe, X_val, y_val)
            sweep_results[name].append(m["RMSE"])

    # Best α per method
    print("  Best α per method (by validation RMSE):")
    show_table(
        ["Method", "best α", "val RMSE", "val R²"],
        [
            [
                name,
                f"{alphas[int(np.argmin(rmses))]:.3g}",
                f"{min(rmses):.4f}",
                "—",
            ]
            for name, rmses in sweep_results.items()
        ],
        col_width=14,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, rmses in sweep_results.items():
        ax.plot(alphas, rmses, marker="o", label=name)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\alpha$  (regularisation strength)")
    ax.set_ylabel("Validation RMSE")
    ax.set_title("Validation RMSE vs α — polynomial-2 features")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_sweep = os.path.join(OUTPUT_DIR, "alpha_sweep.png")
    plt.savefig(out_sweep, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  alpha_sweep.png → {out_sweep}")

    # ── 3.2  Coefficient paths on linear features ───────────────────────────
    subsection("3.2  Coefficient paths — how weights shrink with α")
    feature_names = dataset["feature_names"]
    n_feat = X_train.shape[1]
    path_alphas = np.logspace(-3, 2, 30)
    ridge_paths = np.zeros((len(path_alphas), n_feat))
    lasso_paths = np.zeros_like(ridge_paths)
    for i, a in enumerate(path_alphas):
        r = Ridge(alpha=a).fit(X_train, y_train)
        ridge_paths[i] = r.coef_
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            la = Lasso(alpha=a, max_iter=20000).fit(X_train, y_train)
        lasso_paths[i] = la.coef_

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for j, name in enumerate(feature_names):
        axes[0].plot(path_alphas, ridge_paths[:, j], label=name)
        axes[1].plot(path_alphas, lasso_paths[:, j], label=name)
    for ax, title in zip(axes, ["Ridge", "Lasso"]):
        ax.set_xscale("log")
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_xlabel(r"$\alpha$")
        ax.set_title(f"{title} coefficient path")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("coefficient")
    axes[0].legend(fontsize=8, loc="upper right")
    plt.suptitle("Coefficient paths on California Housing (standardised features)",
                 fontweight="bold")
    plt.tight_layout()
    out_paths = os.path.join(OUTPUT_DIR, "regularization_paths.png")
    plt.savefig(out_paths, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  regularization_paths.png → {out_paths}")

    # ── 3.3  Lasso as a feature selector ────────────────────────────────────
    subsection("3.3  Lasso sets coefficients to *exactly* zero")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        la = Lasso(alpha=0.05, max_iter=20000).fit(X_train, y_train)
    nz = int((np.abs(la.coef_) > 1e-8).sum())
    print(f"  At α=0.05, Lasso keeps {nz} / {n_feat} features non-zero:")
    show_table(
        ["Feature", "Weight"],
        [[name, f"{w:+.4f}" if abs(w) > 1e-8 else "  0  (dropped)"]
         for name, w in zip(feature_names, la.coef_)],
        col_width=24,
    )

    # ── 3.4  Elastic Net l1_ratio sweep ─────────────────────────────────────
    subsection("3.4  Elastic Net — sweep l1_ratio at fixed α")
    print("  l1_ratio = 0  → pure Ridge;   l1_ratio = 1  → pure Lasso\n")
    rows = []
    for l1r in [0.0, 0.25, 0.5, 0.75, 1.0]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            en = ElasticNet(alpha=0.01, l1_ratio=l1r, max_iter=20000)
            en.fit(X_train, y_train)
        m = evaluate_model(en, X_val, y_val)
        nz = int((np.abs(en.coef_) > 1e-8).sum())
        rows.append([f"l1_ratio={l1r}", m["RMSE"], m["R2"], nz])
    show_table(
        ["Hyperparameter", "Val RMSE", "Val R²", "#non-zero coefs"],
        rows, col_width=20,
    )


# ════════════════════════════════════════════════════════════════════════════
#  LAB 4 — TREE-BASED REGRESSION
# ════════════════════════════════════════════════════════════════════════════

def lab4_trees():
    section("4 — TREE-BASED REGRESSION")

    require(["numpy"])
    import numpy as np
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import (
        RandomForestRegressor,
        GradientBoostingRegressor,
    )

    dataset = load_and_prep_data()
    X_train, y_train = dataset["train"]
    X_val,   y_val   = dataset["val"]

    # ── 4.1  Single tree — depth sweep ──────────────────────────────────────
    subsection("4.1  Decision Tree — picking depth by validation RMSE")
    print("  Shallow trees underfit; deep trees memorise the training set.\n")
    rows = []
    best_d, best_rmse = None, float("inf")
    for d in [2, 3, 5, 7, 10, 15, 20, None]:
        dt = DecisionTreeRegressor(max_depth=d, random_state=SEED)
        dt.fit(X_train, y_train)
        m_tr = evaluate_model(dt, X_train, y_train)
        m_va = evaluate_model(dt, X_val,   y_val)
        rows.append([
            f"max_depth={d}", m_tr["RMSE"], m_va["RMSE"], m_va["R2"],
        ])
        if m_va["RMSE"] < best_rmse:
            best_rmse, best_d = m_va["RMSE"], d
    show_table(
        ["Hyperparameter", "Train RMSE", "Val RMSE", "Val R²"],
        rows, col_width=18,
    )
    print(f"\n  Best max_depth by val RMSE: {best_d}  (val RMSE = {best_rmse:.4f})")

    # ── 4.2  Random Forest — #trees sweep ───────────────────────────────────
    subsection("4.2  Random Forest — effect of #trees")
    rows = []
    for T in [10, 50, 100, 200, 500]:
        rf = RandomForestRegressor(n_estimators=T, random_state=SEED, n_jobs=-1)
        rf.fit(X_train, y_train)
        m = evaluate_model(rf, X_val, y_val)
        rows.append([f"n_estimators={T}", m["RMSE"], m["MAE"], m["R2"]])
    show_table(
        ["Hyperparameter", "Val RMSE", "Val MAE", "Val R²"],
        rows, col_width=18,
    )

    # ── 4.3  Random Forest — feature importances ────────────────────────────
    subsection("4.3  Random Forest — top feature importances")
    rf = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    order = np.argsort(importances)[::-1]
    feats = dataset["feature_names"]
    show_table(
        ["Rank", "Feature", "Importance"],
        [[i + 1, feats[idx], f"{importances[idx]:.4f}"] for i, idx in enumerate(order)],
        col_width=22,
    )

    # ── 4.4  Gradient Boosting — depth × learning-rate ──────────────────────
    subsection("4.4  Gradient Boosting — depth × learning rate")
    rows = []
    for depth in [2, 3, 5]:
        for lr in [0.01, 0.05, 0.1]:
            gb = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=depth,
                learning_rate=lr,
                random_state=SEED,
            )
            gb.fit(X_train, y_train)
            m = evaluate_model(gb, X_val, y_val)
            rows.append([f"depth={depth}, lr={lr}", m["RMSE"], m["R2"]])
    show_table(
        ["Hyperparameter", "Val RMSE", "Val R²"],
        rows, col_width=24,
    )

    # ── 4.5  XGBoost (optional) ─────────────────────────────────────────────
    subsection("4.5  XGBoost")
    try:
        import xgboost as xgb
        xgb_reg = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            random_state=SEED,
            n_jobs=-1,
        )
        xgb_reg.fit(X_train, y_train)
        m = evaluate_model(xgb_reg, X_val, y_val)
        print("  300 trees, depth=5, lr=0.1")
        print(f"  Val RMSE: {m['RMSE']}")
        print(f"  Val MAE:  {m['MAE']}")
        print(f"  Val R²:   {m['R2']}")
    except ImportError:
        print("  xgboost not installed — skipping.")
        print("  Install with: pip install xgboost")


# ════════════════════════════════════════════════════════════════════════════
#  LAB 5 — FINAL BENCHMARK ON THE TEST SET + VISUALISATIONS
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class RegressorResult:
    name:      str
    metrics:   Dict[str, float]
    y_pred:    "np.ndarray" = None  # noqa: F821
    cv_scores: "np.ndarray" = field(default=None)  # noqa: F821


def build_all_regressors():
    """Return the list of (name, model) pairs used in the final benchmark."""
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import (
        RandomForestRegressor,
        GradientBoostingRegressor,
    )

    regressors = [
        ("Linear",            LinearRegression()),
        ("Ridge (α=1)",       Ridge(alpha=1.0)),
        ("Lasso (α=0.01)",    Lasso(alpha=0.01, max_iter=20000)),
        ("ElasticNet",        ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=20000)),
        ("Decision Tree",     DecisionTreeRegressor(max_depth=8, random_state=SEED)),
        ("Random Forest",     RandomForestRegressor(n_estimators=300,
                                                    random_state=SEED, n_jobs=-1)),
        ("Gradient Boosting", GradientBoostingRegressor(n_estimators=300,
                                                        max_depth=3,
                                                        learning_rate=0.1,
                                                        random_state=SEED)),
    ]

    try:
        import xgboost as xgb
        regressors.append((
            "XGBoost",
            xgb.XGBRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.1,
                random_state=SEED, n_jobs=-1,
            ),
        ))
    except ImportError:
        pass

    return regressors


def lab5_final_benchmark():
    section("5 — FINAL BENCHMARK ON THE TEST SET + VISUALISATIONS")
    print(textwrap.dedent("""
      We fit every regressor on the training set, evaluate once on the
      held-out test set, run 5-fold cross-validation on the full
      dataset, and save a suite of comparison plots.
    """))

    require(["numpy", "matplotlib", "sklearn"])
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.model_selection import KFold, cross_val_score, learning_curve
    from sklearn.metrics import mean_squared_error

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset = load_and_prep_data()
    X_train, y_train = dataset["train"]
    X_test,  y_test  = dataset["test"]
    X_full,  y_full  = dataset["full"]

    regressors = build_all_regressors()

    # ── 5.1  Train + evaluate on test ───────────────────────────────────────
    subsection("5.1  Train on train, evaluate on test")
    results: List[RegressorResult] = []
    rows = []
    for name, reg in regressors:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reg.fit(X_train, y_train)
        m = evaluate_model(reg, X_test, y_test)
        y_pred = reg.predict(X_test)
        results.append(RegressorResult(name, m, y_pred))
        rows.append([name, m["RMSE"], m["MAE"], m["R2"]])

    show_table(
        ["Regressor", "RMSE", "MAE", "R²"],
        sorted(rows, key=lambda r: r[1]),  # ascending RMSE
        col_width=20,
    )

    # ── 5.2  5-fold cross-validation ────────────────────────────────────────
    subsection("5.2  5-fold cross-validation (R²)")
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_rows = []
    for r, (name, reg) in zip(results, regressors):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r.cv_scores = cross_val_score(
                reg, X_full, y_full, cv=kf, scoring="r2", n_jobs=-1,
            )
        cv_rows.append([
            name,
            f"{r.cv_scores.mean():.4f}",
            f"{r.cv_scores.std():.4f}",
            f"[{r.cv_scores.min():.3f}, {r.cv_scores.max():.3f}]",
        ])
    show_table(
        ["Regressor", "CV mean R²", "CV std", "Range"],
        sorted(cv_rows, key=lambda r: -float(r[1])),
        col_width=22,
    )

    # ── 5.3  Visualisations ─────────────────────────────────────────────────
    subsection("5.3  Saving visualisations")
    plot_metrics_bar(results, os.path.join(OUTPUT_DIR, "metrics_comparison.png"))
    plot_predictions_vs_actual(
        results, y_test, os.path.join(OUTPUT_DIR, "predictions_vs_actual.png"),
    )
    plot_residuals(
        results, y_test, os.path.join(OUTPUT_DIR, "residuals.png"),
    )
    plot_feature_importance(
        regressors, dataset["feature_names"],
        os.path.join(OUTPUT_DIR, "feature_importance.png"),
    )
    plot_cv_box(results, os.path.join(OUTPUT_DIR, "cv_results.png"))
    plot_learning_curve(
        X_train, y_train,
        os.path.join(OUTPUT_DIR, "learning_curve.png"),
    )

    print(f"\n  All plots saved to {OUTPUT_DIR}/")


# ════════════════════════════════════════════════════════════════════════════
#  VISUALISATIONS
# ════════════════════════════════════════════════════════════════════════════

def plot_metrics_bar(results, save_path):
    import numpy as np
    import matplotlib.pyplot as plt

    names = [r.name for r in results]
    x = np.arange(len(names))
    width = 0.27

    fig, ax_left = plt.subplots(figsize=(13, 6))
    ax_right = ax_left.twinx()

    ax_left.bar(x - width, [r.metrics["RMSE"] for r in results],
                width=width, color="#1f77b4", label="RMSE")
    ax_left.bar(x,         [r.metrics["MAE"]  for r in results],
                width=width, color="#ff7f0e", label="MAE")
    ax_right.bar(x + width, [r.metrics["R2"]   for r in results],
                 width=width, color="#2ca02c", label="R² (right axis)")

    ax_left.set_xticks(x)
    ax_left.set_xticklabels(names, rotation=30, ha="right")
    ax_left.set_ylabel("RMSE / MAE")
    ax_right.set_ylabel("R²")
    ax_right.set_ylim(0, 1)
    ax_left.set_title("Test-set metrics across regressors")

    lines1, labels1 = ax_left.get_legend_handles_labels()
    lines2, labels2 = ax_right.get_legend_handles_labels()
    ax_left.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax_left.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  metrics_comparison.png    → {save_path}")


def plot_predictions_vs_actual(results, y_test, save_path):
    import numpy as np
    import matplotlib.pyplot as plt

    n = len(results)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    lo, hi = float(min(y_test)), float(max(y_test))

    for ax, r in zip(axes, results):
        ax.scatter(y_test, r.y_pred, s=6, alpha=0.3)
        ax.plot([lo, hi], [lo, hi], "--", color="red", linewidth=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("actual")
        ax.set_ylabel("predicted")
        ax.set_title(f"{r.name}\nR²={r.metrics['R2']:.3f}  RMSE={r.metrics['RMSE']:.3f}",
                     fontsize=10)
    for ax in axes[len(results):]:
        ax.set_visible(False)
    plt.suptitle("Predictions vs Actual (test set)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  predictions_vs_actual.png → {save_path}")


def plot_residuals(results, y_test, save_path):
    import numpy as np
    import matplotlib.pyplot as plt

    n = len(results)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()

    for ax, r in zip(axes, results):
        residuals = y_test - r.y_pred
        ax.scatter(r.y_pred, residuals, s=6, alpha=0.3)
        ax.axhline(0, color="red", linestyle="--", linewidth=1)
        ax.set_xlabel("predicted")
        ax.set_ylabel("residual (y − ŷ)")
        ax.set_title(r.name, fontsize=10)
    for ax in axes[len(results):]:
        ax.set_visible(False)
    plt.suptitle("Residual plots (test set)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  residuals.png             → {save_path}")


def plot_feature_importance(regressors, feature_names, save_path):
    import numpy as np
    import matplotlib.pyplot as plt

    tree_models = [
        (name, reg) for name, reg in regressors
        if hasattr(reg, "feature_importances_")
    ]
    if not tree_models:
        return

    ncols = min(3, len(tree_models))
    nrows = (len(tree_models) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for ax, (name, reg) in zip(axes, tree_models):
        importances = reg.feature_importances_
        idx = np.argsort(importances)[::-1]
        ax.barh(range(len(idx)), importances[idx][::-1], color="steelblue")
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels([feature_names[i] for i in idx[::-1]], fontsize=8)
        ax.set_title(f"{name}", fontsize=10)
        ax.set_xlabel("importance")

    for ax in axes[len(tree_models):]:
        ax.set_visible(False)
    plt.suptitle("Feature importances (tree-based regressors)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  feature_importance.png    → {save_path}")


def plot_cv_box(results, save_path):
    import matplotlib.pyplot as plt

    data = [r.cv_scores for r in results if r.cv_scores is not None]
    labels = [r.name for r in results if r.cv_scores is not None]

    fig, ax = plt.subplots(figsize=(11, 6))
    try:
        ax.boxplot(data, tick_labels=labels, showmeans=True)
    except TypeError:
        ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_ylabel("5-fold CV R²")
    ax.set_title("Cross-validation R² distribution")
    plt.xticks(rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  cv_results.png            → {save_path}")


def plot_learning_curve(X_train, y_train, save_path):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import learning_curve

    sizes, train_scores, val_scores = learning_curve(
        GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                  learning_rate=0.1, random_state=SEED),
        X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 8),
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        random_state=SEED,
    )
    train_rmse = -train_scores.mean(axis=1)
    val_rmse   = -val_scores.mean(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, train_rmse, marker="o", label="train RMSE")
    ax.plot(sizes, val_rmse,   marker="o", label="val RMSE")
    ax.set_xlabel("training set size")
    ax.set_ylabel("RMSE")
    ax.set_title("Learning curve — Gradient Boosting (5-fold CV)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  learning_curve.png        → {save_path}")


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Supervised Regression Tutorial")
    parser.add_argument(
        "--lab", type=int, choices=[1, 2, 3, 4, 5],
        help=(
            "Run a specific lab "
            "(1=Data/eval, 2=Linear/Polynomial, 3=Ridge/Lasso/ElasticNet, "
            "4=Trees/RF/GB, 5=Final benchmark + plots)"
        ),
    )
    args = parser.parse_args()

    print("\n" + "█" * 70)
    print("  SUPERVISED LEARNING: REGRESSION  ")
    print("█" * 70)
    print("""
  Labs:
    1 → Data, train/val/test splits, evaluation metrics
    2 → Linear & Polynomial regression
    3 → Ridge, Lasso, Elastic Net  (coefficient paths, α sweeps)
    4 → Decision Tree, Random Forest, Gradient Boosting, XGBoost
    5 → Final benchmark on the test set + visualisations
    """)

    labs = {
        1: lab1_basics,
        2: lab2_linear_polynomial,
        3: lab3_regularization,
        4: lab4_trees,
        5: lab5_final_benchmark,
    }

    if args.lab is not None:
        labs[args.lab]()
    else:
        for fn in labs.values():
            fn()


if __name__ == "__main__":
    main()
