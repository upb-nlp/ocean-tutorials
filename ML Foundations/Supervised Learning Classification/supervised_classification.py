"""
Supervised Learning: Classification — companion tutorial script.

Trains and compares a battery of classifiers on the Breast Cancer Wisconsin
dataset (569 samples, 30 features, binary).  Each lab matches a section of the
README.

Run all labs:
    python supervised_classification.py

Run a single lab:
    python supervised_classification.py --lab 1   # Data, splits, evaluation
    python supervised_classification.py --lab 2   # kNN, Naive Bayes, Decision Tree
    python supervised_classification.py --lab 3   # SVM, MLP
    python supervised_classification.py --lab 4   # Random Forest, AdaBoost, GB, XGBoost
    python supervised_classification.py --lab 5   # Final benchmark + visualisations
"""

import argparse
import os
import sys
import textwrap
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

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


OUTPUT_DIR = "./outputs/classification"
SEED = 42


# ════════════════════════════════════════════════════════════════════════════
#  DATA LOADING & PREPROCESSING
# ════════════════════════════════════════════════════════════════════════════

def load_and_prep_data():
    """Load Breast Cancer Wisconsin, split 60/20/20 and standardize."""
    require(["sklearn", "numpy"])

    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    data = load_breast_cancer()
    X, y = data.data, data.target

    # 60 % train, 20 % validation, 20 % test (stratified)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=SEED, stratify=y_temp
    )

    # Standardize using only train statistics
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Also keep the full set scaled for cross-validation later
    X_full_s = scaler.fit_transform(X)

    return {
        "train":         (X_train_s, y_train),
        "val":           (X_val_s, y_val),
        "test":          (X_test_s, y_test),
        "full":          (X_full_s, y),
        "feature_names": list(data.feature_names),
        "target_names":  list(data.target_names),
    }


def evaluate_model(model, X, y) -> Dict[str, float]:
    """Return accuracy, precision, recall, F1 of a fitted model on (X, y)."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score
    )
    preds = model.predict(X)
    return {
        "Accuracy":  round(float(accuracy_score(y, preds)),  4),
        "Precision": round(float(precision_score(y, preds, zero_division=0)), 4),
        "Recall":    round(float(recall_score(y, preds,    zero_division=0)), 4),
        "F1":        round(float(f1_score(y, preds,        zero_division=0)), 4),
    }


# ════════════════════════════════════════════════════════════════════════════
#  LAB 1 — DATA, SPLITS, EVALUATION METRICS
# ════════════════════════════════════════════════════════════════════════════

def lab1_basics():
    section("1 — DATA, SPLITS & EVALUATION METRICS")
    print(textwrap.dedent("""
      Dataset: Breast Cancer Wisconsin
        - 569 samples
        - 30 numerical features (radius, texture, perimeter, …)
        - Binary target: 0 = malignant, 1 = benign

      We split 60 % / 20 % / 20 % (train / val / test), stratified by class.
      Features are standardized using the training set only — crucial for kNN,
      SVM and MLP.
    """))

    require(["numpy"])
    import numpy as np
    from sklearn.metrics import (
        confusion_matrix, classification_report,
    )

    dataset = load_and_prep_data()
    X_train, y_train = dataset["train"]
    X_val,   y_val   = dataset["val"]
    X_test,  y_test  = dataset["test"]

    subsection("1.1  Set sizes")
    show_table(
        ["Set", "Size", "Class 0 (malign.)", "Class 1 (benign)"],
        [
            ["Train",      len(y_train), int((y_train == 0).sum()), int((y_train == 1).sum())],
            ["Validation", len(y_val),   int((y_val   == 0).sum()), int((y_val   == 1).sum())],
            ["Test",       len(y_test),  int((y_test  == 0).sum()), int((y_test  == 1).sum())],
        ],
    )

    subsection("1.2  A baseline — always predict the majority class")
    majority = int(np.bincount(y_train).argmax())
    baseline_preds = np.full_like(y_test, majority)
    baseline_acc = float((baseline_preds == y_test).mean())
    print(f"  Majority class:        {majority} ({dataset['target_names'][majority]})")
    print(f"  Baseline test accuracy: {baseline_acc:.4f}")
    print("  Any useful classifier must beat this number.")

    subsection("1.3  Confusion matrix on the baseline")
    cm = confusion_matrix(y_test, baseline_preds)
    print("                Predicted 0    Predicted 1")
    print(f"  Actual 0      {cm[0, 0]:>10}   {cm[0, 1]:>10}")
    print(f"  Actual 1      {cm[1, 0]:>10}   {cm[1, 1]:>10}")

    subsection("1.4  Why accuracy alone is misleading")
    print("  A classifier that always says \"benign\" on this dataset already")
    print(f"  scores {baseline_acc:.1%} accuracy — but its recall on the malignant class is 0,")
    print("  i.e. every single cancerous sample is missed.")
    print("  → always look at precision / recall / F1 alongside accuracy.")

    subsection("1.5  Full classification report (baseline)")
    print(classification_report(
        y_test, baseline_preds,
        target_names=dataset["target_names"], zero_division=0,
    ))


# ════════════════════════════════════════════════════════════════════════════
#  LAB 2 — KNN, NAÏVE BAYES, DECISION TREE
# ════════════════════════════════════════════════════════════════════════════

def lab2_simple_classifiers():
    section("2 — KNN, NAÏVE BAYES & DECISION TREE")

    require(["numpy"])
    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier, export_text

    dataset = load_and_prep_data()
    X_train, y_train = dataset["train"]
    X_val,   y_val   = dataset["val"]

    # ── 2.1  kNN — sweep k on the validation set ─────────────────────────────
    subsection("2.1  k-Nearest Neighbours — picking k by validation accuracy")
    print("  kNN has *no training*: prediction time scans the training set.")
    print("  We sweep odd k values and pick the one with the best val accuracy.\n")

    knn_rows = []
    best_k, best_acc = None, -1.0
    for k in [1, 3, 5, 7, 9, 11, 15, 21]:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        metrics = evaluate_model(knn, X_val, y_val)
        knn_rows.append([f"k = {k}", metrics["Accuracy"], metrics["F1"]])
        if metrics["Accuracy"] > best_acc:
            best_acc, best_k = metrics["Accuracy"], k

    show_table(["Hyperparameter", "Val Accuracy", "Val F1"], knn_rows)
    print(f"\n  Best k by validation accuracy: {best_k}  (acc = {best_acc:.4f})")

    # ── 2.2  Distance-weighted kNN ───────────────────────────────────────────
    subsection("2.2  Distance-weighted vs uniform voting")
    print("  weights='distance' lets closer neighbours count more than far ones.\n")
    for w in ["uniform", "distance"]:
        knn = KNeighborsClassifier(n_neighbors=best_k, weights=w)
        knn.fit(X_train, y_train)
        m = evaluate_model(knn, X_val, y_val)
        print(f"  weights={w:<10}  acc={m['Accuracy']:.4f}  F1={m['F1']:.4f}")

    # ── 2.3  Gaussian Naïve Bayes ───────────────────────────────────────────
    subsection("2.3  Gaussian Naïve Bayes")
    print("  Models each P(x_j | c) as a Gaussian.")
    print("  No hyperparameters to tune — pure closed-form fit.\n")
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    m = evaluate_model(gnb, X_val, y_val)
    print(f"  Val Accuracy: {m['Accuracy']}")
    print(f"  Val F1:       {m['F1']}")

    # Show how its per-class means differ for a few features
    print("\n  Class-conditional Gaussian means for the first 5 features:")
    feat = dataset["feature_names"][:5]
    means = gnb.theta_   # shape (n_classes, n_features)
    show_table(
        ["Feature", f"μ | {dataset['target_names'][0]}", f"μ | {dataset['target_names'][1]}"],
        [[feat[i], round(float(means[0, i]), 3), round(float(means[1, i]), 3)] for i in range(5)],
        col_width=22,
    )

    # ── 2.4  Decision Tree — depth sweep ─────────────────────────────────────
    subsection("2.4  Decision Tree — picking depth by validation accuracy")
    print("  Shallow trees underfit; deep trees memorize the training set.")
    print("  We sweep max_depth and report training vs validation accuracy.\n")

    dt_rows = []
    best_d, best_d_acc = None, -1.0
    for d in [1, 2, 3, 4, 5, 7, 10, None]:
        dt = DecisionTreeClassifier(max_depth=d, random_state=SEED)
        dt.fit(X_train, y_train)
        train_m = evaluate_model(dt, X_train, y_train)
        val_m   = evaluate_model(dt, X_val,   y_val)
        dt_rows.append([
            f"max_depth = {d}",
            train_m["Accuracy"],
            val_m["Accuracy"],
            val_m["F1"],
        ])
        if val_m["Accuracy"] > best_d_acc:
            best_d_acc, best_d = val_m["Accuracy"], d
    show_table(
        ["Hyperparameter", "Train Acc", "Val Acc", "Val F1"],
        dt_rows,
        col_width=18,
    )
    print(f"\n  Best max_depth by val accuracy: {best_d}")

    # ── 2.5  Inspect the best small tree ─────────────────────────────────────
    subsection("2.5  The if–then rules of a small tree (max_depth=3)")
    dt = DecisionTreeClassifier(max_depth=3, random_state=SEED)
    dt.fit(X_train, y_train)
    rules = export_text(
        dt,
        feature_names=dataset["feature_names"],
        max_depth=3,
    )
    print(rules)

    # ── 2.6  Gini vs Entropy ─────────────────────────────────────────────────
    subsection("2.6  Gini vs entropy as the split criterion")
    for crit in ["gini", "entropy"]:
        dt = DecisionTreeClassifier(criterion=crit, max_depth=best_d, random_state=SEED)
        dt.fit(X_train, y_train)
        m = evaluate_model(dt, X_val, y_val)
        print(f"  criterion={crit:<8}  val_acc={m['Accuracy']:.4f}  val_F1={m['F1']:.4f}")


# ════════════════════════════════════════════════════════════════════════════
#  LAB 3 — SVM & MLP
# ════════════════════════════════════════════════════════════════════════════

def lab3_svm_mlp():
    section("3 — SUPPORT VECTOR MACHINES & MULTILAYER PERCEPTRON")

    require(["numpy"])
    import numpy as np
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier

    dataset = load_and_prep_data()
    X_train, y_train = dataset["train"]
    X_val,   y_val   = dataset["val"]

    # ── 3.1  SVM — kernel comparison ────────────────────────────────────────
    subsection("3.1  SVM with different kernels (default C, γ)")
    svm_rows = []
    for kernel in ["linear", "rbf", "poly", "sigmoid"]:
        svm = SVC(kernel=kernel, random_state=SEED)
        svm.fit(X_train, y_train)
        m = evaluate_model(svm, X_val, y_val)
        svm_rows.append([
            kernel,
            m["Accuracy"], m["Precision"], m["Recall"], m["F1"],
            int(svm.support_.shape[0]),
        ])
    show_table(
        ["Kernel", "Acc", "Prec", "Recall", "F1", "#SV"],
        svm_rows,
        col_width=14,
    )

    # ── 3.2  RBF SVM — sweep C (regularisation) ─────────────────────────────
    subsection("3.2  Effect of C — the soft-margin penalty")
    print("  Small C → wider margin, more tolerance for misclassification.")
    print("  Large C → narrow margin, fewer misclassifications.\n")
    c_rows = []
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        svm = SVC(kernel="rbf", C=C, random_state=SEED)
        svm.fit(X_train, y_train)
        m_tr = evaluate_model(svm, X_train, y_train)
        m_va = evaluate_model(svm, X_val,   y_val)
        c_rows.append([f"C = {C}", m_tr["Accuracy"], m_va["Accuracy"], m_va["F1"]])
    show_table(
        ["Hyperparameter", "Train Acc", "Val Acc", "Val F1"],
        c_rows, col_width=18,
    )

    # ── 3.3  Sweep γ on the RBF kernel ──────────────────────────────────────
    subsection("3.3  Effect of γ on the RBF kernel")
    print("  Larger γ → tighter, more local kernel → risk of overfitting.\n")
    g_rows = []
    for gamma in [0.001, 0.01, 0.1, 1.0, 10.0]:
        svm = SVC(kernel="rbf", gamma=gamma, random_state=SEED)
        svm.fit(X_train, y_train)
        m_tr = evaluate_model(svm, X_train, y_train)
        m_va = evaluate_model(svm, X_val,   y_val)
        g_rows.append([f"γ = {gamma}", m_tr["Accuracy"], m_va["Accuracy"], m_va["F1"]])
    show_table(
        ["Hyperparameter", "Train Acc", "Val Acc", "Val F1"],
        g_rows, col_width=18,
    )

    # ── 3.4  MLP — architecture sweep ───────────────────────────────────────
    subsection("3.4  MLP — different hidden layer architectures")
    print("  A feed-forward neural net trained with Adam.")
    print("  Standardised inputs are required to converge.\n")
    archs = [
        (16,),
        (32,),
        (32, 16),
        (64, 32),
        (128, 64, 32),
    ]
    mlp_rows = []
    for arch in archs:
        mlp = MLPClassifier(
            hidden_layer_sizes=arch,
            max_iter=500,
            random_state=SEED,
            early_stopping=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mlp.fit(X_train, y_train)
        m = evaluate_model(mlp, X_val, y_val)
        mlp_rows.append([
            f"layers={arch}",
            m["Accuracy"], m["F1"],
            int(mlp.n_iter_),
        ])
    show_table(
        ["Architecture", "Val Acc", "Val F1", "Epochs"],
        mlp_rows, col_width=22,
    )

    # ── 3.5  Activation and optimiser ───────────────────────────────────────
    subsection("3.5  MLP — activation and optimiser sweep")
    combos = [
        ("relu",     "adam"),
        ("relu",     "sgd"),
        ("tanh",     "adam"),
        ("logistic", "adam"),
    ]
    rows = []
    for act, opt in combos:
        mlp = MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation=act,
            solver=opt,
            max_iter=500,
            random_state=SEED,
            early_stopping=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mlp.fit(X_train, y_train)
        m = evaluate_model(mlp, X_val, y_val)
        rows.append([act, opt, m["Accuracy"], m["F1"]])
    show_table(
        ["Activation", "Optimiser", "Val Acc", "Val F1"],
        rows, col_width=14,
    )


# ════════════════════════════════════════════════════════════════════════════
#  LAB 4 — ENSEMBLES
# ════════════════════════════════════════════════════════════════════════════

def lab4_ensembles():
    section("4 — ENSEMBLE METHODS")

    require(["numpy"])
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import (
        RandomForestClassifier,
        AdaBoostClassifier,
        GradientBoostingClassifier,
    )

    dataset = load_and_prep_data()
    X_train, y_train = dataset["train"]
    X_val,   y_val   = dataset["val"]

    # ── 4.1  A single tree (baseline) ───────────────────────────────────────
    subsection("4.1  Single decision tree (baseline)")
    dt = DecisionTreeClassifier(max_depth=3, random_state=SEED)
    dt.fit(X_train, y_train)
    m = evaluate_model(dt, X_val, y_val)
    print(f"  max_depth=3   val_acc={m['Accuracy']:.4f}  val_F1={m['F1']:.4f}")

    # ── 4.2  Random Forest — # trees sweep ──────────────────────────────────
    subsection("4.2  Random Forest — effect of #trees")
    rows = []
    for T in [10, 50, 100, 200, 500]:
        rf = RandomForestClassifier(n_estimators=T, random_state=SEED, n_jobs=-1)
        rf.fit(X_train, y_train)
        m = evaluate_model(rf, X_val, y_val)
        rows.append([f"n_estimators={T}", m["Accuracy"], m["F1"]])
    show_table(["Hyperparameter", "Val Acc", "Val F1"], rows, col_width=20)

    # ── 4.3  Random Forest — feature importances ─────────────────────────────
    subsection("4.3  Random Forest — top-10 feature importances")
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    order = np.argsort(importances)[::-1][:10]
    feats = dataset["feature_names"]
    show_table(
        ["Rank", "Feature", "Importance"],
        [[i + 1, feats[idx], f"{importances[idx]:.4f}"] for i, idx in enumerate(order)],
        col_width=28,
    )

    # ── 4.4  AdaBoost — # rounds sweep ──────────────────────────────────────
    subsection("4.4  AdaBoost — boosting rounds")
    rows = []
    for T in [10, 50, 100, 200]:
        ada = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=SEED),
            n_estimators=T,
            random_state=SEED,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ada.fit(X_train, y_train)
        m = evaluate_model(ada, X_val, y_val)
        rows.append([f"n_estimators={T}", m["Accuracy"], m["F1"]])
    show_table(["Hyperparameter", "Val Acc", "Val F1"], rows, col_width=20)

    # ── 4.5  Gradient Boosting — learning-rate / depth ──────────────────────
    subsection("4.5  Gradient Boosting — depth × learning rate")
    rows = []
    for depth in [1, 3, 5]:
        for lr in [0.01, 0.1, 1.0]:
            gb = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=depth,
                learning_rate=lr,
                random_state=SEED,
            )
            gb.fit(X_train, y_train)
            m = evaluate_model(gb, X_val, y_val)
            rows.append([f"depth={depth}, lr={lr}", m["Accuracy"], m["F1"]])
    show_table(["Hyperparameter", "Val Acc", "Val F1"], rows, col_width=24)

    # ── 4.6  XGBoost (optional) ─────────────────────────────────────────────
    subsection("4.6  XGBoost")
    try:
        import xgboost as xgb
        xgb_clf = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=-1,
        )
        xgb_clf.fit(X_train, y_train)
        m = evaluate_model(xgb_clf, X_val, y_val)
        print(f"  300 trees, depth=4, lr=0.1")
        print(f"  Val Accuracy: {m['Accuracy']}")
        print(f"  Val F1:       {m['F1']}")
    except ImportError:
        print("  xgboost not installed — skipping.")
        print("  Install with: pip install xgboost")


# ════════════════════════════════════════════════════════════════════════════
#  LAB 5 — FINAL BENCHMARK ON THE TEST SET + VISUALISATIONS
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class ClassifierResult:
    name:      str
    metrics:   Dict[str, float]
    y_pred:    "np.ndarray" = None  # noqa: F821
    y_score:   "np.ndarray" = None  # noqa: F821
    cv_scores: "np.ndarray" = field(default=None)  # noqa: F821


def build_all_classifiers():
    """Return the list of (name, model) pairs used in the final benchmark."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import (
        RandomForestClassifier,
        AdaBoostClassifier,
        GradientBoostingClassifier,
    )

    classifiers = [
        ("kNN (k=7)",          KNeighborsClassifier(n_neighbors=7)),
        ("Gaussian NB",        GaussianNB()),
        ("Decision Tree",      DecisionTreeClassifier(max_depth=4, random_state=SEED)),
        ("Linear SVM",         SVC(kernel="linear", probability=True, random_state=SEED)),
        ("RBF SVM",            SVC(kernel="rbf",    probability=True, random_state=SEED)),
        ("MLP (32,16)",        MLPClassifier(hidden_layer_sizes=(32, 16),
                                             max_iter=500, random_state=SEED,
                                             early_stopping=True)),
        ("Random Forest",      RandomForestClassifier(n_estimators=300,
                                                     random_state=SEED, n_jobs=-1)),
        ("AdaBoost",           AdaBoostClassifier(
                                  estimator=DecisionTreeClassifier(max_depth=1,
                                                                   random_state=SEED),
                                  n_estimators=200, random_state=SEED)),
        ("Gradient Boosting",  GradientBoostingClassifier(n_estimators=200,
                                                         max_depth=3,
                                                         learning_rate=0.1,
                                                         random_state=SEED)),
    ]

    try:
        import xgboost as xgb
        classifiers.append((
            "XGBoost",
            xgb.XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.1,
                eval_metric="logloss", random_state=SEED, n_jobs=-1,
            ),
        ))
    except ImportError:
        pass

    return classifiers


def lab5_final_benchmark():
    section("5 — FINAL BENCHMARK ON THE TEST SET + VISUALISATIONS")
    print(textwrap.dedent("""
      We fit every classifier on the training set, evaluate once on the
      held-out test set, run 5-fold stratified cross-validation on the full
      dataset, and save a suite of comparison plots.
    """))

    require(["numpy", "matplotlib", "sklearn"])
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import (
        confusion_matrix, roc_curve, auc,
    )
    from sklearn.decomposition import PCA

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset = load_and_prep_data()
    X_train, y_train = dataset["train"]
    X_test,  y_test  = dataset["test"]
    X_full,  y_full  = dataset["full"]

    classifiers = build_all_classifiers()

    # ── 5.1  Train + evaluate on test ───────────────────────────────────────
    subsection("5.1  Train on train, evaluate on test")
    results: List[ClassifierResult] = []
    rows = []
    for name, clf in classifiers:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_train, y_train)
        m = evaluate_model(clf, X_test, y_test)
        y_pred = clf.predict(X_test)
        # Probability or decision score for ROC
        try:
            y_score = clf.predict_proba(X_test)[:, 1]
        except Exception:
            try:
                y_score = clf.decision_function(X_test)
            except Exception:
                y_score = y_pred.astype(float)
        results.append(ClassifierResult(name, m, y_pred, y_score))
        rows.append([name, m["Accuracy"], m["Precision"], m["Recall"], m["F1"]])

    show_table(
        ["Classifier", "Accuracy", "Precision", "Recall", "F1"],
        sorted(rows, key=lambda r: -r[1]),
        col_width=18,
    )

    # ── 5.2  5-fold stratified cross-validation ─────────────────────────────
    subsection("5.2  5-fold stratified cross-validation (accuracy)")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_rows = []
    for r, (name, clf) in zip(results, classifiers):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r.cv_scores = cross_val_score(
                clf, X_full, y_full, cv=skf, scoring="accuracy", n_jobs=-1,
            )
        cv_rows.append([
            name,
            f"{r.cv_scores.mean():.4f}",
            f"{r.cv_scores.std():.4f}",
            f"[{r.cv_scores.min():.3f}, {r.cv_scores.max():.3f}]",
        ])
    show_table(
        ["Classifier", "CV mean", "CV std", "Range"],
        sorted(cv_rows, key=lambda r: -float(r[1])),
        col_width=20,
    )

    # ── 5.3  Bar chart of test metrics ──────────────────────────────────────
    subsection("5.3  Saving visualisations")
    plot_metrics_bar(results, os.path.join(OUTPUT_DIR, "metrics_comparison.png"))
    plot_confusion_matrices(
        results, y_test, dataset["target_names"],
        os.path.join(OUTPUT_DIR, "confusion_matrices.png"),
    )
    plot_roc_curves(results, y_test, os.path.join(OUTPUT_DIR, "roc_curves.png"))
    plot_cv_box(results, os.path.join(OUTPUT_DIR, "cv_results.png"))
    plot_feature_importance(
        classifiers, dataset["feature_names"],
        os.path.join(OUTPUT_DIR, "feature_importance.png"),
    )
    plot_decision_boundaries(
        classifiers, X_train, y_train,
        os.path.join(OUTPUT_DIR, "decision_boundaries.png"),
    )

    print(f"\n  All plots saved to {OUTPUT_DIR}/")


# ════════════════════════════════════════════════════════════════════════════
#  VISUALISATIONS
# ════════════════════════════════════════════════════════════════════════════

def plot_metrics_bar(results, save_path):
    import numpy as np
    import matplotlib.pyplot as plt

    names   = [r.name for r in results]
    metrics_order = ["Accuracy", "Precision", "Recall", "F1"]
    width   = 0.2
    x       = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, key in enumerate(metrics_order):
        ax.bar(
            x + i * width - 1.5 * width,
            [r.metrics[key] for r in results],
            width=width, label=key,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0.7, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Test-set metrics across classifiers")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  metrics_comparison.png    → {save_path}")


def plot_confusion_matrices(results, y_test, target_names, save_path):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    n = len(results)
    ncols = 5
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for ax, r in zip(axes, results):
        cm = confusion_matrix(y_test, r.y_pred)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(target_names, rotation=20, fontsize=8)
        ax.set_yticklabels(target_names, fontsize=8)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                colour = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(
                    j, i, str(cm[i, j]),
                    ha="center", va="center", color=colour, fontsize=11,
                )
        ax.set_title(f"{r.name}\nacc={r.metrics['Accuracy']:.3f}", fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    for ax in axes[len(results):]:
        ax.set_visible(False)
    plt.suptitle("Confusion matrices", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  confusion_matrices.png    → {save_path}")


def plot_roc_curves(results, y_test, save_path):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=(8, 7))
    for r in results:
        fpr, tpr, _ = roc_curve(y_test, r.y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=1.4, label=f"{r.name}  AUC={roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves on the test set")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  roc_curves.png            → {save_path}")


def plot_cv_box(results, save_path):
    import matplotlib.pyplot as plt

    data = [r.cv_scores for r in results if r.cv_scores is not None]
    labels = [r.name for r in results if r.cv_scores is not None]

    fig, ax = plt.subplots(figsize=(11, 6))
    # `labels=` was renamed to `tick_labels=` in matplotlib 3.9; fall back for older versions.
    try:
        ax.boxplot(data, tick_labels=labels, showmeans=True)
    except TypeError:
        ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_ylabel("5-fold CV accuracy")
    ax.set_title("Cross-validation accuracy distribution")
    plt.xticks(rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  cv_results.png            → {save_path}")


def plot_feature_importance(classifiers, feature_names, save_path):
    import numpy as np
    import matplotlib.pyplot as plt

    tree_models = [
        (name, clf) for name, clf in classifiers
        if hasattr(clf, "feature_importances_")
    ]
    if not tree_models:
        return

    ncols = min(3, len(tree_models))
    nrows = (len(tree_models) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()

    for ax, (name, clf) in zip(axes, tree_models):
        importances = clf.feature_importances_
        idx = np.argsort(importances)[::-1][:10]
        ax.barh(range(len(idx)), importances[idx][::-1], color="steelblue")
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels([feature_names[i] for i in idx[::-1]], fontsize=8)
        ax.set_title(f"{name} — top 10", fontsize=10)
        ax.set_xlabel("importance")

    for ax in axes[len(tree_models):]:
        ax.set_visible(False)
    plt.suptitle("Feature importances (tree ensembles)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  feature_importance.png    → {save_path}")


def plot_decision_boundaries(classifiers, X_train, y_train, save_path):
    """PCA-2D decision boundaries for every classifier."""
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from sklearn.base import clone
    from sklearn.decomposition import PCA

    # Project the training set to 2D via PCA
    pca = PCA(n_components=2, random_state=SEED)
    X_2d = pca.fit_transform(X_train)

    # Grid for contour plot
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250),
                         np.linspace(y_min, y_max, 250))
    grid = np.c_[xx.ravel(), yy.ravel()]

    cmap_bg = ListedColormap(["#FFAAAA", "#AAAAFF"])
    cmap_pt = ListedColormap(["#CC0000", "#0000CC"])

    n = len(classifiers)
    ncols = 5
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()

    for ax, (name, original) in zip(axes, classifiers):
        clf = clone(original)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_2d, y_train)
        Z = clf.predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cmap_bg, alpha=0.5)
        ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_train,
                   cmap=cmap_pt, s=12, edgecolors="k", linewidths=0.2)
        ax.set_title(name, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[n:]:
        ax.set_visible(False)
    plt.suptitle("Decision boundaries in PCA-2D space", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  decision_boundaries.png   → {save_path}")


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Supervised Classification Tutorial")
    parser.add_argument(
        "--lab", type=int, choices=[1, 2, 3, 4, 5],
        help=(
            "Run a specific lab "
            "(1=Data/eval, 2=kNN/NB/DT, 3=SVM/MLP, 4=Ensembles, "
            "5=Final benchmark + plots)"
        ),
    )
    args = parser.parse_args()

    print("\n" + "█" * 70)
    print("  SUPERVISED LEARNING: CLASSIFICATION  ")
    print("█" * 70)
    print("""
  Labs:
    1 → Data, train/val/test splits, evaluation metrics
    2 → kNN, Naïve Bayes, Decision Tree
    3 → SVM (kernels, C, γ) and MLP (architecture, activation)
    4 → Random Forest, AdaBoost, Gradient Boosting, XGBoost
    5 → Final benchmark on the test set + visualisations
    """)

    labs = {
        1: lab1_basics,
        2: lab2_simple_classifiers,
        3: lab3_svm_mlp,
        4: lab4_ensembles,
        5: lab5_final_benchmark,
    }

    if args.lab is not None:
        labs[args.lab]()
    else:
        for fn in labs.values():
            fn()


if __name__ == "__main__":
    main()
