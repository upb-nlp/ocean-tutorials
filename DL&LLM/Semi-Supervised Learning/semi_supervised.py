"""
semi_supervised.py
==================
Labels are expensive; unlabeled data is cheap. Semi-supervised learning (SSL)
and active learning both attack the same problem — squeezing more accuracy out
of a small labeled set — from two directions:

    SSL             exploit the unlabeled data you already have
    Active learning choose the FEW points most worth labeling next

    Stage 1  Setup           tiny labeled set + large unlabeled pool; a
                             supervised baseline to beat
    Stage 2  Self-training   the model pseudo-labels its own confident guesses
    Stage 3  Co-training     two feature "views" teach each other
    Stage 4  Label spreading graph-based propagation through the data manifold
    Stage 5  Active learning query the most UNCERTAIN points (vs random)
    Stage 6  Putting it together — combine active learning + self-training

Everything runs offline on scikit-learn's `digits` dataset (and `make_moons`
for the graph visualization). Figures are written to ./images/.

Run:
    python semi_supervised.py
    python semi_supervised.py --stage 5
"""

import os
import sys
import argparse

import numpy as np


IMAGES_DIR = "images"
SEED = 42


# =============================================================================
# UTILITIES
# =============================================================================
def require(packages):
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


def section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def get_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save_fig(fig, name):
    os.makedirs(IMAGES_DIR, exist_ok=True)
    path = os.path.join(IMAGES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    get_plt().close(fig)
    print(f"    figure saved to {path}")


def make_self_training(base, **kw):
    """SelfTrainingClassifier's estimator kwarg was renamed across versions."""
    from sklearn.semi_supervised import SelfTrainingClassifier
    try:
        return SelfTrainingClassifier(estimator=base, **kw)
    except TypeError:
        return SelfTrainingClassifier(base_estimator=base, **kw)


# =============================================================================
# DATA — a small labeled set hidden inside a large unlabeled pool
# =============================================================================
def load_digits_ssl(n_labeled_per_class=4):
    """Load digits, split off a test set, then keep only a few labels in train.

    Returns the train features, a semi-supervised label vector (unlabeled = -1),
    the TRUE train labels (for oracle/active learning), the labeled mask, and a
    held-out test set.
    """
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    digits = load_digits()
    X = digits.data / 16.0                      # pixels 0..16 → 0..1
    y = digits.target

    X_tr, X_test, y_tr, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEED)

    rng = np.random.default_rng(SEED)
    labeled_idx = []
    for c in np.unique(y_tr):
        idx = np.where(y_tr == c)[0]
        labeled_idx.extend(rng.choice(idx, size=n_labeled_per_class, replace=False))
    labeled_idx = np.array(sorted(labeled_idx))

    labeled_mask = np.zeros(len(y_tr), dtype=bool)
    labeled_mask[labeled_idx] = True

    y_semi = np.full(len(y_tr), -1)             # -1 = unlabeled
    y_semi[labeled_mask] = y_tr[labeled_mask]

    return X_tr, y_semi, y_tr, labeled_mask, X_test, y_test


# =============================================================================
# STAGE 1 — SETUP & SUPERVISED BASELINE
# =============================================================================
def stage1_setup():
    section("STAGE 1 — SETUP: a few labels, a lot of unlabeled data")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    X_tr, y_semi, y_tr, mask, X_test, y_test = load_digits_ssl()
    n_lab, n_unlab = mask.sum(), (~mask).sum()
    print(f"\n  digits: {len(y_tr)} train + {len(y_test)} test, 10 classes")
    print(f"  Labeled:   {n_lab}   ({n_lab // 10} per class)")
    print(f"  Unlabeled: {n_unlab}  (labels hidden — marked -1)")

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_tr[mask], y_tr[mask])
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"\n  Supervised baseline (labeled data only): test accuracy = {acc:.3f}")
    print("  Every method below tries to beat this using the unlabeled pool")
    print("  or by labeling a few well-chosen points.")
    return acc


# =============================================================================
# STAGE 2 — SELF-TRAINING
# =============================================================================
def stage2_self_training(baseline=None):
    section("STAGE 2 — SELF-TRAINING (the model pseudo-labels itself)")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    plt = get_plt()

    X_tr, y_semi, y_tr, mask, X_test, y_test = load_digits_ssl()
    base_clf = LogisticRegression(max_iter=2000)
    base_clf.fit(X_tr[mask], y_tr[mask])
    base_acc = accuracy_score(y_test, base_clf.predict(X_test))

    print("\n  Idea: train on the labels you have, predict the unlabeled points,")
    print("  add the CONFIDENT predictions as 'pseudo-labels', retrain, repeat.")
    print("  The confidence threshold trades coverage against error:\n")

    thresholds = [0.60, 0.70, 0.80, 0.90, 0.95]
    accs, n_pseudo = [], []
    for th in thresholds:
        st = make_self_training(LogisticRegression(max_iter=2000), threshold=th)
        st.fit(X_tr, y_semi)
        acc = accuracy_score(y_test, st.predict(X_test))
        pseudo = int((st.transduction_ != -1).sum() - mask.sum())
        accs.append(acc)
        n_pseudo.append(pseudo)
        print(f"    threshold={th:.2f}   pseudo-labeled {pseudo:>4} points   "
              f"test acc={acc:.3f}")

    best = int(np.argmax(accs))
    print(f"\n  Baseline (no self-training):  {base_acc:.3f}")
    print(f"  Best self-training @{thresholds[best]:.2f}: {accs[best]:.3f}  "
          f"({accs[best] - base_acc:+.3f})")
    print("  A low threshold adds many but noisier pseudo-labels (confirmation")
    print("  bias); a high threshold is safer but adds fewer.")

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.plot(thresholds, accs, "o-", color="#1f77b4", label="self-training acc")
    ax1.axhline(base_acc, ls="--", color="grey", label="supervised baseline")
    ax1.set_xlabel("confidence threshold"); ax1.set_ylabel("test accuracy")
    ax1.legend(loc="lower left"); ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.bar(thresholds, n_pseudo, width=0.03, alpha=0.25, color="#2ca02c")
    ax2.set_ylabel("# pseudo-labels added", color="#2ca02c")
    ax1.set_title("Self-training: threshold vs accuracy and coverage")
    save_fig(fig, "self_training.png")
    return accs[best]


# =============================================================================
# STAGE 3 — CO-TRAINING (two views teach each other)
# =============================================================================
def stage3_co_training():
    section("STAGE 3 — CO-TRAINING (two feature views teach each other)")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    plt = get_plt()

    X_tr, y_semi, y_tr, mask, X_test, y_test = load_digits_ssl()
    # Two "views": left half of each 8x8 image vs right half.
    img = X_tr.reshape(-1, 8, 8)
    viewA = img[:, :, :4].reshape(len(X_tr), -1)      # left columns
    viewB = img[:, :, 4:].reshape(len(X_tr), -1)      # right columns
    tImg = X_test.reshape(-1, 8, 8)
    tA, tB = tImg[:, :, :4].reshape(len(X_test), -1), tImg[:, :, 4:].reshape(len(X_test), -1)

    print("\n  Split each image into a LEFT view and a RIGHT view. Two classifiers,")
    print("  one per view, take turns adding their most confident guesses to the")
    print("  shared labeled pool — each teaching the other what it is sure about.\n")

    labeled = mask.copy()
    pseudo_y = y_semi.copy()
    rng = np.random.default_rng(SEED)
    iterations, accs = [], []

    for it in range(0, 9):
        if it > 0:
            clfA = LogisticRegression(max_iter=1000).fit(viewA[labeled], pseudo_y[labeled])
            clfB = LogisticRegression(max_iter=1000).fit(viewB[labeled], pseudo_y[labeled])
            unlab = np.where(~labeled)[0]
            if len(unlab):
                for clf, view in [(clfA, viewA), (clfB, viewB)]:
                    proba = clf.predict_proba(view[unlab])
                    conf = proba.max(axis=1)
                    pred = clf.classes_[proba.argmax(axis=1)]
                    # take this view's 10 most-confident unlabeled points
                    take = unlab[np.argsort(conf)[::-1][:10]]
                    labeled[take] = True
                    pseudo_y[take] = pred[np.argsort(conf)[::-1][:10]]

        # measure a full-feature classifier trained on the current labeled pool
        clf = LogisticRegression(max_iter=2000).fit(X_tr[labeled], pseudo_y[labeled])
        acc = accuracy_score(y_test, clf.predict(X_test))
        iterations.append(it); accs.append(acc)
        print(f"    iter {it}:  labeled pool = {labeled.sum():>4}   test acc = {acc:.3f}")

    print(f"\n  Co-training grew {mask.sum()} labels into {labeled.sum()} and moved")
    print(f"  accuracy {accs[0]:.3f} → {accs[-1]:.3f} ({accs[-1]-accs[0]:+.3f}).")
    print("  Co-training works best when the two views are each sufficient and")
    print("  conditionally independent — here, two halves of the same digit.")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(iterations, accs, "o-", color="#9467bd")
    ax.axhline(accs[0], ls="--", color="grey", label="start (labeled only)")
    ax.set_xlabel("co-training iteration"); ax.set_ylabel("test accuracy")
    ax.set_title("Co-training: accuracy as the two views expand the labels")
    ax.legend(); ax.grid(alpha=0.3)
    save_fig(fig, "co_training.png")
    return accs[-1]


# =============================================================================
# STAGE 4 — LABEL PROPAGATION / SPREADING (graph-based)
# =============================================================================
def stage4_label_spreading():
    section("STAGE 4 — LABEL SPREADING (propagate through the data graph)")
    from sklearn.semi_supervised import LabelSpreading
    from sklearn.metrics import accuracy_score
    from sklearn.datasets import make_moons
    plt = get_plt()

    # (a) quantitative on digits
    X_tr, y_semi, y_tr, mask, X_test, y_test = load_digits_ssl()
    ls = LabelSpreading(kernel="knn", n_neighbors=7)
    ls.fit(X_tr, y_semi)
    # transductive accuracy on the (originally unlabeled) train points
    trans_acc = accuracy_score(y_tr[~mask], ls.transduction_[~mask])
    ind_acc = accuracy_score(y_test, ls.predict(X_test))
    print("\n  Build a graph connecting near-neighbour points, then let the few")
    print("  known labels 'spread' along the edges until every node is labeled.\n")
    print(f"  Transductive accuracy on the unlabeled train points: {trans_acc:.3f}")
    print(f"  Inductive accuracy on the held-out test set:         {ind_acc:.3f}")
    print("  Graph methods shine when classes form smooth, connected clusters.")

    # (b) visualization on two moons
    Xm, ym = make_moons(n_samples=250, noise=0.08, random_state=SEED)
    y_seed = np.full(len(ym), -1)
    # one labeled seed per moon
    y_seed[np.where(ym == 0)[0][0]] = 0
    y_seed[np.where(ym == 1)[0][0]] = 1
    lsm = LabelSpreading(kernel="knn", n_neighbors=10)
    lsm.fit(Xm, y_seed)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    seed_mask = y_seed != -1
    axes[0].scatter(Xm[~seed_mask, 0], Xm[~seed_mask, 1], c="lightgrey", s=25)
    axes[0].scatter(Xm[seed_mask, 0], Xm[seed_mask, 1],
                    c=y_seed[seed_mask], cmap="coolwarm", s=200,
                    edgecolors="k", marker="*")
    axes[0].set_title("Start: 2 labeled seeds (one per moon)")
    axes[1].scatter(Xm[:, 0], Xm[:, 1], c=lsm.transduction_, cmap="coolwarm", s=25)
    axes[1].scatter(Xm[seed_mask, 0], Xm[seed_mask, 1],
                    c=y_seed[seed_mask], cmap="coolwarm", s=200,
                    edgecolors="k", marker="*")
    axes[1].set_title("After spreading: whole manifold labeled")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Label spreading propagates 2 labels across the data manifold",
                 fontsize=13)
    save_fig(fig, "label_propagation.png")
    return ind_acc


# =============================================================================
# STAGE 5 — ACTIVE LEARNING (label the points that matter most)
# =============================================================================
def _al_loop(strategy, X_pool, y_pool, X_test, y_test, seed_idx, budget, batch):
    """Iteratively label points chosen by `strategy` and track test accuracy."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    labeled = list(seed_idx)
    pool = set(range(len(X_pool))) - set(labeled)
    rng = np.random.default_rng(SEED)
    sizes, accs = [], []

    while True:
        clf = LogisticRegression(max_iter=2000).fit(X_pool[labeled], y_pool[labeled])
        accs.append(accuracy_score(y_test, clf.predict(X_test)))
        sizes.append(len(labeled))
        if len(labeled) >= budget or not pool:
            break
        pool_idx = np.array(sorted(pool))
        if strategy == "random":
            chosen = rng.choice(pool_idx, size=min(batch, len(pool_idx)), replace=False)
        else:  # uncertainty: smallest margin between top-2 class probabilities
            proba = clf.predict_proba(X_pool[pool_idx])
            part = np.sort(proba, axis=1)
            margin = part[:, -1] - part[:, -2]
            chosen = pool_idx[np.argsort(margin)[:batch]]
        for c in chosen:
            labeled.append(int(c)); pool.discard(int(c))
    return sizes, accs


def stage5_active_learning():
    section("STAGE 5 — ACTIVE LEARNING (query the most uncertain points)")
    plt = get_plt()

    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_digits
    digits = load_digits()
    X, y = digits.data / 16.0, digits.target
    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEED)

    rng = np.random.default_rng(SEED)
    seed_idx = [int(rng.choice(np.where(y_pool == c)[0])) for c in np.unique(y_pool)]

    print("\n  Instead of labeling random points, label the ones the model is")
    print("  most UNSURE about (smallest margin between its top two classes).")
    print("  Same labeling budget, smarter choices.\n")

    s_rand, a_rand = _al_loop("random", X_pool, y_pool, X_test, y_test,
                              seed_idx, budget=150, batch=10)
    s_unc, a_unc = _al_loop("uncertainty", X_pool, y_pool, X_test, y_test,
                            seed_idx, budget=150, batch=10)

    for n, ar, au in zip(s_rand, a_rand, a_unc):
        print(f"    {n:>3} labels   random={ar:.3f}   uncertainty={au:.3f}")
    print(f"\n  At {s_unc[-1]} labels: uncertainty {a_unc[-1]:.3f} vs random "
          f"{a_rand[-1]:.3f} ({a_unc[-1]-a_rand[-1]:+.3f}).")
    print("  Active learning reaches the same accuracy with far fewer labels —")
    print("  it spends the labeling budget where the model is confused.")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(s_rand, a_rand, "o-", color="grey", label="random sampling")
    ax.plot(s_unc, a_unc, "s-", color="#d62728", label="uncertainty sampling")
    ax.set_xlabel("# labeled points"); ax.set_ylabel("test accuracy")
    ax.set_title("Active learning vs random labeling")
    ax.legend(); ax.grid(alpha=0.3)
    save_fig(fig, "active_learning.png")
    return a_unc[-1]


# =============================================================================
# STAGE 6 — PUTTING IT TOGETHER
# =============================================================================
def stage6_combine():
    section("STAGE 6 — PUTTING IT TOGETHER (active learning + self-training)")
    from sklearn.linear_model import LogisticRegression
    from sklearn.semi_supervised import LabelSpreading
    from sklearn.metrics import accuracy_score
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    plt = get_plt()

    digits = load_digits()
    X, y = digits.data / 16.0, digits.target
    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEED)
    rng = np.random.default_rng(SEED)

    BUDGET = 80
    print(f"\n  A fixed labeling budget of {BUDGET} points. Which recipe wins?\n")

    # 1. baseline — 80 RANDOM labels, supervised
    rand_idx = list(rng.choice(len(X_pool), size=BUDGET, replace=False))
    base = LogisticRegression(max_iter=2000).fit(X_pool[rand_idx], y_pool[rand_idx])
    acc_base = accuracy_score(y_test, base.predict(X_test))

    # 2. random labels + self-training on the rest
    y_semi = np.full(len(y_pool), -1); y_semi[rand_idx] = y_pool[rand_idx]
    st = make_self_training(LogisticRegression(max_iter=2000), threshold=0.9).fit(X_pool, y_semi)
    acc_self = accuracy_score(y_test, st.predict(X_test))

    # 3. random labels + label spreading
    ls = LabelSpreading(kernel="knn", n_neighbors=7).fit(X_pool, y_semi)
    acc_spread = accuracy_score(y_test, ls.predict(X_test))

    # 4. ACTIVE learning to pick the 80 labels
    seed_idx = [int(rng.choice(np.where(y_pool == c)[0])) for c in np.unique(y_pool)]
    s, a = _al_loop("uncertainty", X_pool, y_pool, X_test, y_test,
                    seed_idx, budget=BUDGET, batch=10)
    # reconstruct the actively-labeled set by replaying the loop's final classifier
    acc_active = a[-1]

    # 5. active labels + self-training on the rest
    #    (re-run active loop to recover the chosen indices)
    labeled = list(seed_idx); pool = set(range(len(X_pool))) - set(labeled)
    while len(labeled) < BUDGET and pool:
        clf = LogisticRegression(max_iter=2000).fit(X_pool[labeled], y_pool[labeled])
        pool_idx = np.array(sorted(pool))
        proba = clf.predict_proba(X_pool[pool_idx])
        part = np.sort(proba, axis=1); margin = part[:, -1] - part[:, -2]
        chosen = pool_idx[np.argsort(margin)[:10]]
        for c in chosen:
            labeled.append(int(c)); pool.discard(int(c))
    y_semi2 = np.full(len(y_pool), -1); y_semi2[labeled] = y_pool[labeled]
    st2 = make_self_training(LogisticRegression(max_iter=2000), threshold=0.9).fit(X_pool, y_semi2)
    acc_active_self = accuracy_score(y_test, st2.predict(X_test))

    methods = ["Random\n(supervised)", "Random\n+ self-train", "Random\n+ spread",
               "Active\n(supervised)", "Active\n+ self-train"]
    scores = [acc_base, acc_self, acc_spread, acc_active, acc_active_self]
    for m, sc in zip(methods, scores):
        print(f"    {m.replace(chr(10), ' '):<26} {sc:.3f}")
    best = int(np.argmax(scores))
    print(f"\n  Winner: {methods[best].replace(chr(10),' ')} at {scores[best]:.3f}.")
    print("  Choosing WHAT to label (active) and exploiting what you didn't")
    print("  (self-training) are complementary — together they beat either alone.")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#7f7f7f", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    bars = ax.bar(methods, scores, color=colors)
    bars[best].set_edgecolor("black"); bars[best].set_linewidth(2.5)
    ax.set_ylabel("test accuracy")
    ax.set_ylim(min(scores) - 0.05, 1.0)
    ax.set_title(f"Same {BUDGET}-label budget, five recipes")
    for b, sc in zip(bars, scores):
        ax.text(b.get_x() + b.get_width() / 2, sc + 0.005, f"{sc:.3f}",
                ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, "ssl_comparison.png")


# =============================================================================
# MAIN
# =============================================================================
def main():
    require(["scikit-learn", "numpy", "matplotlib"])
    parser = argparse.ArgumentParser(description="Semi-supervised & active learning")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4, 5, 6],
                        help="Run a single stage (default: all).")
    args = parser.parse_args()

    print("\n" + "#" * 70)
    print("  SEMI-SUPERVISED & ACTIVE LEARNING")
    print("#" * 70)

    stages = {
        1: stage1_setup,
        2: stage2_self_training,
        3: stage3_co_training,
        4: stage4_label_spreading,
        5: stage5_active_learning,
        6: stage6_combine,
    }
    to_run = [args.stage] if args.stage else [1, 2, 3, 4, 5, 6]
    for s in to_run:
        stages[s]()

    print(f"\n  Figures written to ./{IMAGES_DIR}/\n")


if __name__ == "__main__":
    main()
