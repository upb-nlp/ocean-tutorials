"""
Association Rules and Sequential Patterns - companion tutorial script.

Mines frequent itemsets, association rules, class association rules, and
sequential patterns on the small example databases.

Run all labs:
    python association_rules.py

Run a single lab:
    python association_rules.py --lab 1   # Items, transactions, support, confidence
    python association_rules.py --lab 2   # Apriori - mlxtend
    python association_rules.py --lab 3   # FP-Growth - mlxtend
    python association_rules.py --lab 4   # Class Association Rules (CARs)
    python association_rules.py --lab 5   # Sequential Patterns + visualisations
"""

import argparse
import os
import sys
import time
import textwrap
import warnings
from collections import Counter, defaultdict
from itertools import combinations
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
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def subsection(title: str) -> None:
    print(f"\n  -- {title} " + "-" * max(0, 60 - len(title)))


def show_table(headers: List[str], rows: List[List], col_width: int = 18) -> None:
    fmt = "  " + "".join(f"{{:<{col_width}}}" for _ in headers)
    print(fmt.format(*headers))
    print("  " + "-" * (col_width * len(headers)))
    for row in rows:
        print(fmt.format(*[str(c)[: col_width - 1] for c in row]))


OUTPUT_DIR = "./outputs/association_rules"


# ════════════════════════════════════════════════════════════════════════════
#  EXAMPLE DATASETS
# ════════════════════════════════════════════════════════════════════════════

# "Documents" database (6 docs containing keywords) used to illustrate
# support, confidence, lift, and conviction in lab 1.
DOCS_DATASET = [
    ["rule", "tree", "classification"],
    ["relation", "tuple", "join", "algebra", "recommendation"],
    ["variable", "loop", "procedure", "rule"],
    ["clustering", "rule", "tree", "recommendation"],
    ["join", "relation", "selection", "projection", "classification"],
    ["rule", "tree", "recommendation"],
]
DOCS_IDS = [f"Doc{i+1}" for i in range(len(DOCS_DATASET))]

# Same documents, now class-labelled - used for Class Association Rules.
DOCS_LABELS = ["datamining", "database", "programming",
               "datamining", "database", "datamining"]

# 9 transactions over items I1..I5 - the textbook Apriori walk-through
# (lab 2): minsup_count = 2, builds frequent 1, 2, and 3-itemsets, then
# generates rules from {I1, I2, I3}.
T_DATASET = [
    ["I1", "I2", "I5"],
    ["I2", "I4"],
    ["I2", "I3"],
    ["I1", "I2", "I4"],
    ["I1", "I3"],
    ["I2", "I3"],
    ["I1", "I3"],
    ["I1", "I2", "I3", "I5"],
    ["I1", "I2", "I3"],
]

# 5 transactions over items A..E - small ToDo example (identical to the
# colab notebook) used in lab 2 and lab 3.
ABCDE_DATASET = [
    ["A", "B"],
    ["A", "B", "D"],
    ["B", "D"],
    ["B", "C", "D", "E"],
    ["A", "B", "C", "D"],
]

# 10 transactions over items A..E - FP-Growth walk-through in lab 3.
FP_DATASET = [
    ["A", "B"],
    ["B", "C", "D"],
    ["A", "C", "D", "E"],
    ["A", "D", "E"],
    ["A", "B", "C"],
    ["A", "B", "C", "D"],
    ["A"],
    ["A", "B", "C"],
    ["A", "B", "D"],
    ["B", "C", "E"],
]

# 5 sequences for the sequential-pattern mining example in lab 5.
# Each sequence is a list of events; each event is a list of items.
SEQ_DATASET = [
    [["A"],         ["B"],      ["C"]],
    [["A", "B"],    ["C"],      ["A", "D"]],
    [["A", "B", "C"], ["B", "C", "E"]],
    [["A", "D"],    ["B", "C"], ["A", "E"]],
    [["B"],         ["E"]],
]

# 4 sequences for the ToDo sequence example in lab 5.
TODO_SEQ_DATASET = [
    [["A"], ["A", "B", "C"], ["A", "C"], ["D"], ["C"]],
    [["A", "D"], ["C"], ["B", "C"], ["A"]],
    [["E"], ["A", "B"], ["D"], ["C"], ["B"]],
    [["A"], ["C"], ["B"], ["C"]],
]


# ════════════════════════════════════════════════════════════════════════════
#  LOW-LEVEL HELPERS - pure-python support and rule generation
# ════════════════════════════════════════════════════════════════════════════

def count_support(itemset, transactions):
    """Number of transactions containing every item in `itemset`."""
    its = set(itemset)
    return sum(1 for t in transactions if its.issubset(t))


def support(itemset, transactions):
    """Fractional support of an itemset."""
    return count_support(itemset, transactions) / len(transactions)


def confidence(X, Y, transactions):
    """Confidence of the rule X → Y."""
    sxy = count_support(set(X) | set(Y), transactions)
    sx  = count_support(X, transactions)
    return sxy / sx if sx else 0.0


def lift(X, Y, transactions):
    """Lift of the rule X → Y."""
    sy = support(Y, transactions)
    return confidence(X, Y, transactions) / sy if sy else 0.0


def conviction(X, Y, transactions):
    """Conviction of the rule X → Y."""
    sy = support(Y, transactions)
    c  = confidence(X, Y, transactions)
    return (1 - sy) / (1 - c) if c < 1 else float("inf")


# ════════════════════════════════════════════════════════════════════════════
#  LAB 1 - ITEMS, TRANSACTIONS, SUPPORT, CONFIDENCE
# ════════════════════════════════════════════════════════════════════════════

def lab1_basics():
    section("1 - ITEMS, TRANSACTIONS, SUPPORT & CONFIDENCE")
    print(textwrap.dedent("""
      Goal: introduce the building blocks of frequent itemset and rule mining
      using the small "documents" database.
    """))

    # -- 1.1 -----------------------------------------------------------------
    subsection("1.1  The transaction database")
    rows = [[did, ", ".join(t)] for did, t in zip(DOCS_IDS, DOCS_DATASET)]
    show_table(["TID", "Items"], rows, col_width=50)

    # -- 1.2 -----------------------------------------------------------------
    subsection("1.2  Support of a few itemsets")
    examples = [
        ["rule"],
        ["tree"],
        ["recommendation"],
        ["rule", "tree"],
        ["relation", "join"],
        ["rule", "tree", "recommendation"],
    ]
    rows = []
    for X in examples:
        c = count_support(X, DOCS_DATASET)
        s = support(X, DOCS_DATASET)
        rows.append([
            "{" + ", ".join(X) + "}",
            f"{c}/{len(DOCS_DATASET)}",
            f"{s:.2%}",
            "yes" if s >= 0.5 else "no",
        ])
    show_table(["Itemset", "Count", "Support", "Frequent @ s=50%"],
               rows, col_width=24)

    # -- 1.3 -----------------------------------------------------------------
    subsection("1.3  Rules and their quality metrics")
    rules = [("rule", "tree"), ("tree", "rule"),
             ("recommendation", "rule"), ("rule", "recommendation")]
    rows = []
    for X, Y in rules:
        rows.append([
            f"{X} -> {Y}",
            f"{support([X, Y], DOCS_DATASET):.2%}",
            f"{confidence([X], [Y], DOCS_DATASET):.2%}",
            f"{lift([X], [Y], DOCS_DATASET):.2f}",
            f"{conviction([X], [Y], DOCS_DATASET):.2f}",
        ])
    show_table(["Rule", "Support", "Confidence", "Lift", "Conviction"],
               rows, col_width=14)

    # -- 1.4 -----------------------------------------------------------------
    subsection("1.4  Confidence is asymmetric")
    print("    sup(rule -> tree) = sup(tree -> rule) = 50%")
    print("    BUT confidences differ: 75% vs 100%.")
    print("    Every transaction with `tree` also contains `rule`,")
    print("    yet `rule` appears in transactions without `tree`.")

    # -- 1.5 -----------------------------------------------------------------
    subsection("1.5  Why high confidence is not enough (lift)")
    print("    A rule with conf=90% and lift=1.0 only restates the base rate.")
    print("    A rule with conf=50% and lift=5.0 is much more informative.")


# ════════════════════════════════════════════════════════════════════════════
#  LAB 2 - APRIORI
# ════════════════════════════════════════════════════════════════════════════

def _mlxtend_apriori(dataset, min_support, use_colnames=True):
    """Helper: encode + apriori, returning the frequent-itemsets dataframe."""
    require(["pandas", "mlxtend"])
    import pandas as pd
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori
    te = TransactionEncoder()
    arr = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(arr, columns=te.columns_)
    fi = apriori(df, min_support=min_support, use_colnames=use_colnames)
    return df, fi


def lab2_apriori():
    section("2 - APRIORI (mlxtend)")
    print(textwrap.dedent("""
      Apriori (Agrawal & Srikant, 1994) -- level-wise frequent itemset
      enumeration. We use mlxtend's implementation and reproduce the
      walk-through on the I1..I5 / T1..T9 database.
    """))

    require(["pandas", "mlxtend"])
    import pandas as pd
    from mlxtend.frequent_patterns import association_rules

    # -- 2.1  T1..T9 walk-through --------------------------------------------
    subsection("2.1  T1..T9 walk-through")
    print("    minsup_count = 2 / 9 = 22.2%, minconf = 50%\n")
    df, fi = _mlxtend_apriori(T_DATASET, min_support=2/9)
    fi = fi.sort_values(["support", "itemsets"], ascending=[False, True]).reset_index(drop=True)
    rows = [[i + 1,
             "{" + ", ".join(sorted(it)) + "}",
             f"{s:.2%}",
             round(s * len(T_DATASET))]
            for i, (it, s) in enumerate(zip(fi["itemsets"], fi["support"]))]
    show_table(["#", "Itemset", "Support", "Count"], rows, col_width=22)

    # -- 2.2  rules from {I1,I2,I3} ------------------------------------------
    subsection("2.2  Rules generated from {I1, I2, I3}")
    rules = association_rules(fi, metric="confidence", min_threshold=0.0)
    target = frozenset({"I1", "I2", "I3"})
    mask = rules.apply(
        lambda r: set(r["antecedents"]) | set(r["consequents"]) == target, axis=1
    )
    sub = rules[mask].copy()
    if not sub.empty:
        sub["rule"] = sub.apply(
            lambda r: "{" + ", ".join(sorted(r["antecedents"])) + "} -> {"
                      + ", ".join(sorted(r["consequents"])) + "}",
            axis=1,
        )
        rows = [[r["rule"], f"{r['support']:.2%}", f"{r['confidence']:.0%}",
                 f"{r['lift']:.2f}",
                 "KEEP" if r["confidence"] >= 0.5 else "drop"]
                for _, r in sub.iterrows()]
        show_table(["Rule", "Support", "Confidence", "Lift", "@ conf>=50%"],
                   rows, col_width=22)

    # -- 2.3  ToDo example {A,B,C,D,E} ---------------------------------------
    subsection("2.3  ToDo example")
    print("    dataset = [[A,B], [A,B,D], [B,D], [B,C,D,E], [A,B,C,D]]")
    print("    minsup = 2 / 5 = 40%, minconf = 100%\n")
    df, fi = _mlxtend_apriori(ABCDE_DATASET, min_support=0.4)
    fi = fi.sort_values(["support", "itemsets"], ascending=[False, True]).reset_index(drop=True)
    rows = [[i + 1, "{" + ", ".join(sorted(it)) + "}", f"{s:.0%}",
             round(s * len(ABCDE_DATASET))]
            for i, (it, s) in enumerate(zip(fi["itemsets"], fi["support"]))]
    show_table(["#", "Itemset", "Support", "Count"], rows, col_width=20)

    subsection("2.4  Rules at conf >= 100%")
    rules = association_rules(fi, metric="confidence", min_threshold=1.0)
    if rules.empty:
        print("    no rules survived.")
    else:
        rules = rules.sort_values("support", ascending=False)
        rows = []
        for _, r in rules.iterrows():
            rows.append([
                "{" + ", ".join(sorted(r["antecedents"])) + "} -> {"
                + ", ".join(sorted(r["consequents"])) + "}",
                f"{r['support']:.0%}",
                f"{r['confidence']:.0%}",
                f"{r['lift']:.2f}",
            ])
        show_table(["Rule", "Support", "Confidence", "Lift"],
                   rows, col_width=22)


# ════════════════════════════════════════════════════════════════════════════
#  LAB 3 - FP-GROWTH
# ════════════════════════════════════════════════════════════════════════════

def lab3_fpgrowth():
    section("3 - FP-GROWTH (mlxtend)")
    print(textwrap.dedent("""
      FP-Growth (Han, Pei & Yin, 2000) compresses transactions into an
      FP-Tree and mines patterns by recursive projection - no candidate
      generation, only 2 passes over the data.
    """))

    require(["pandas", "mlxtend"])
    import pandas as pd
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

    # -- 3.1 -----------------------------------------------------------------
    subsection("3.1  FP-Growth on the 10-transaction dataset")
    print("    minsup_count = 2 / 10 = 20%\n")
    te = TransactionEncoder()
    arr = te.fit(FP_DATASET).transform(FP_DATASET)
    df = pd.DataFrame(arr, columns=te.columns_)
    fi = fpgrowth(df, min_support=0.2, use_colnames=True)
    fi = fi.sort_values(["support", "itemsets"], ascending=[False, True]).reset_index(drop=True)
    rows = [[i + 1, "{" + ", ".join(sorted(it)) + "}", f"{s:.0%}",
             round(s * len(FP_DATASET))]
            for i, (it, s) in enumerate(zip(fi["itemsets"], fi["support"]))]
    show_table(["#", "Itemset", "Support", "Count"], rows, col_width=22)

    # -- 3.2  colab dataset ---------------------------------------------------
    subsection("3.2  FP-Growth on the colab dataset")
    arr = te.fit(ABCDE_DATASET).transform(ABCDE_DATASET)
    df = pd.DataFrame(arr, columns=te.columns_)
    fi = fpgrowth(df, min_support=0.4, use_colnames=True)
    fi = fi.sort_values(["support", "itemsets"], ascending=[False, True]).reset_index(drop=True)
    rows = [["{" + ", ".join(sorted(it)) + "}", f"{s:.0%}"]
            for it, s in zip(fi["itemsets"], fi["support"])]
    show_table(["Itemset", "Support"], rows, col_width=24)

    subsection("3.3  Rules at conf = 100% (matches the colab output)")
    rules = association_rules(fi, metric="confidence", min_threshold=1.0)
    if rules.empty:
        print("    no rules survived.")
    else:
        rows = [
            ["{" + ", ".join(sorted(r["antecedents"])) + "} -> {"
             + ", ".join(sorted(r["consequents"])) + "}",
             f"{r['support']:.0%}",
             f"{r['confidence']:.0%}",
             f"{r['lift']:.2f}"]
            for _, r in rules.iterrows()
        ]
        show_table(["Rule", "Support", "Confidence", "Lift"],
                   rows, col_width=22)

    # -- 3.4  Apriori vs FP-Growth runtime -----------------------------------
    subsection("3.4  Apriori vs FP-Growth runtime")
    arr = te.fit(FP_DATASET).transform(FP_DATASET)
    df = pd.DataFrame(arr, columns=te.columns_)
    n = 200
    t0 = time.perf_counter()
    for _ in range(n):
        apriori(df, min_support=0.2, use_colnames=True)
    t_ap = (time.perf_counter() - t0) / n
    t0 = time.perf_counter()
    for _ in range(n):
        fpgrowth(df, min_support=0.2, use_colnames=True)
    t_fp = (time.perf_counter() - t0) / n
    show_table(
        ["Algorithm", "Per-call time", "Speedup"],
        [
            ["Apriori",   f"{t_ap*1000:.2f} ms", "1.0x"],
            ["FP-Growth", f"{t_fp*1000:.2f} ms", f"{t_ap/t_fp:.2f}x"],
        ],
        col_width=20,
    )

    # -- 3.5  Optional Spark FP-Growth ---------------------------------------
    subsection("3.5  Cross-check with mlxtend Apriori")
    print("    Verifying FP-Growth and Apriori produce the same frequent itemsets\n")
    arr = te.fit(ABCDE_DATASET).transform(ABCDE_DATASET)
    df_check = pd.DataFrame(arr, columns=te.columns_)
    fi_ap = apriori(df_check, min_support=0.4, use_colnames=True)
    fi_fp = fpgrowth(df_check, min_support=0.4, use_colnames=True)
    ap_sets = set(frozenset(x) for x in fi_ap["itemsets"])
    fp_sets = set(frozenset(x) for x in fi_fp["itemsets"])
    if ap_sets == fp_sets:
        print(f"    OK - both algorithms found the same {len(ap_sets)} frequent itemsets.")
    else:
        print(f"    MISMATCH - Apriori: {len(ap_sets)}, FP-Growth: {len(fp_sets)}")
        print(f"    Only in Apriori: {ap_sets - fp_sets}")
        print(f"    Only in FP-Growth: {fp_sets - ap_sets}")


# ════════════════════════════════════════════════════════════════════════════
#  LAB 4 - CLASS ASSOCIATION RULES (CARs)
# ════════════════════════════════════════════════════════════════════════════

def mine_cars(transactions, labels, min_support, min_confidence):
    """Naive miner of class association rules X -> y.

    Enumerates every non-empty antecedent X from the union of all items;
    for each (X, y) it computes support and confidence and keeps those
    above the thresholds.  Inefficient for big data.
    """
    items = sorted({i for t in transactions for i in t})
    classes = sorted(set(labels))
    n = len(transactions)
    rules = []
    for k in range(1, len(items) + 1):
        for combo in combinations(items, k):
            mask = [set(combo).issubset(t) for t in transactions]
            sup_X = sum(mask)
            if sup_X == 0:
                continue
            for c in classes:
                sup_Xy = sum(1 for ok, lab in zip(mask, labels) if ok and lab == c)
                sup_rule = sup_Xy / n
                conf_rule = sup_Xy / sup_X
                if sup_rule >= min_support and conf_rule >= min_confidence:
                    rules.append({
                        "antecedent": tuple(combo),
                        "class": c,
                        "support": sup_rule,
                        "confidence": conf_rule,
                        "ant_support": sup_X / n,
                    })
    return rules


def predict_cba(test_transaction, sorted_rules, default_class):
    """First rule whose antecedent matches the test wins (CBA-style)."""
    for r in sorted_rules:
        if set(r["antecedent"]).issubset(test_transaction):
            return r["class"]
    return default_class


def lab4_cars():
    section("4 - CLASS ASSOCIATION RULES (CARs)")
    print(textwrap.dedent("""
      CARs restrict the consequent to a single class label.
      We mine them on the documents dataset with a naive
      enumeration miner, then build a CBA-style classifier on top.
    """))

    # -- 4.1 -----------------------------------------------------------------
    subsection("4.1  Class-labelled transactions")
    rows = [[did, ", ".join(t), lab]
            for did, t, lab in zip(DOCS_IDS, DOCS_DATASET, DOCS_LABELS)]
    show_table(["TID", "Transaction", "Label"], rows, col_width=46)

    # -- 4.2 -----------------------------------------------------------------
    subsection("4.2  A few CARs computed by hand")
    handpicked = [
        (["rule"], "datamining"),
        (["recommendation"], "database"),
        (["rule", "tree"], "datamining"),
        (["relation"], "database"),
    ]
    rows = []
    for ant, c in handpicked:
        mask = [set(ant).issubset(t) for t in DOCS_DATASET]
        sup_X  = sum(mask)
        sup_Xy = sum(1 for ok, lab in zip(mask, DOCS_LABELS) if ok and lab == c)
        n = len(DOCS_DATASET)
        rows.append([
            "{" + ", ".join(ant) + "} -> " + c,
            f"{sup_Xy}/{n} = {sup_Xy/n:.0%}",
            f"{sup_Xy}/{sup_X} = {sup_Xy/sup_X:.0%}" if sup_X else "-",
        ])
    show_table(["CAR", "Support", "Confidence"], rows, col_width=34)

    # -- 4.3 -----------------------------------------------------------------
    subsection("4.3  Mine all CARs at minsup=33%, minconf=60%")
    rules = mine_cars(DOCS_DATASET, DOCS_LABELS,
                      min_support=2/6, min_confidence=0.6)
    rules.sort(key=lambda r: (-r["confidence"], -r["support"],
                              len(r["antecedent"])))
    rows = []
    for r in rules[:15]:
        rows.append([
            "{" + ", ".join(r["antecedent"]) + "} -> " + r["class"],
            f"{r['support']:.0%}",
            f"{r['confidence']:.0%}",
        ])
    show_table(["CAR", "Support", "Confidence"], rows, col_width=42)
    print(f"\n    Total CARs at the thresholds: {len(rules)}")

    # -- 4.4 -----------------------------------------------------------------
    subsection("4.4  CBA classifier on hold-out transactions")
    print("    Sort CARs by (confidence, support, antecedent length).")
    print("    Predict with the first matching antecedent; otherwise default.\n")
    default = Counter(DOCS_LABELS).most_common(1)[0][0]
    held_out = [
        (["rule", "tree"],                    "datamining"),
        (["relation", "join"],                "database"),
        (["variable", "loop", "recommendation"], "programming"),
        (["recommendation"],                  "datamining"),
    ]
    rows = []
    correct = 0
    for tx, gold in held_out:
        pred = predict_cba(set(tx), rules, default)
        ok = "OK" if pred == gold else "MISS"
        correct += pred == gold
        rows.append(["{" + ", ".join(tx) + "}", gold, pred, ok])
    show_table(["Transaction", "Gold", "Predicted", ""], rows, col_width=44)
    print(f"\n    Accuracy: {correct}/{len(held_out)} = {correct/len(held_out):.0%}"
          f"   (default class = '{default}')")


# ════════════════════════════════════════════════════════════════════════════
#  LAB 5 - SEQUENTIAL PATTERNS + VISUALISATIONS
# ════════════════════════════════════════════════════════════════════════════

def _is_subsequence(pat, seq):
    """True if `pat` (list of sets) is a subsequence of `seq`."""
    i = 0
    for ev in seq:
        if set(pat[i]).issubset(ev):
            i += 1
            if i == len(pat):
                return True
    return False


def _all_sub_events(event):
    """All non-empty subsets of an event, returned as sorted tuples."""
    items = sorted(event)
    for k in range(1, len(items) + 1):
        for combo in combinations(items, k):
            yield combo


def _generate_candidates(prev, all_items):
    """Generate length-(k+1) sequence candidates from length-k patterns."""
    new = set()
    for p in prev:
        for it in all_items:
            # extend by a new event = singleton (it,)
            new.add(p + ((it,),))
            # extend the last event of p
            last = p[-1]
            if it > last[-1]:
                extended_last = tuple(sorted(last + (it,)))
                new.add(p[:-1] + (extended_last,))
    return new


def gsp(seq_db, min_support):
    """A tiny GSP-style sequential pattern miner.

    seq_db is a list of sequences; each sequence is a list of events; each
    event is a list of items.  Returns {pattern: support_count}.

    `pattern` is a tuple of tuples of sorted items.
    """
    n = len(seq_db)
    seq_db_sets = [[set(ev) for ev in s] for s in seq_db]
    all_items = sorted({i for s in seq_db for ev in s for i in ev})

    # Level 1 -- singletons
    counts = Counter()
    for it in all_items:
        pat = ((it,),)
        counts[pat] = sum(1 for s in seq_db_sets if _is_subsequence(pat, s))
    L = {p: c for p, c in counts.items() if c / n >= min_support}
    all_freq = dict(L)

    k = 1
    while L and k < 6:  # safety cap on length
        cands = _generate_candidates(list(L.keys()), all_items)
        new_L = {}
        for c in cands:
            cnt = sum(1 for s in seq_db_sets if _is_subsequence(c, s))
            if cnt / n >= min_support:
                new_L[c] = cnt
        all_freq.update(new_L)
        L = new_L
        k += 1
    return all_freq


def _fmt_seq(pat):
    """Pretty-print a sequence pattern as <evt1, evt2, ...>."""
    parts = []
    for ev in pat:
        parts.append("".join(ev) if len(ev) > 1 else ev[0])
    return "<" + ", ".join(parts) + ">"


def lab5_sequential_and_viz():
    section("5 - SEQUENTIAL PATTERNS + VISUALISATIONS")
    print(textwrap.dedent("""
      Mine sequential patterns on the two example databases with a small
      built-in GSP miner, cross-check with the prefixspan library (PrefixSpan),
      then produce the full suite of plots.
    """))

    # -- 5.1 -----------------------------------------------------------------
    subsection("5.1  First sequence database (5 sequences)")
    rows = []
    for i, s in enumerate(SEQ_DATASET, 1):
        rows.append([i, _fmt_seq([tuple(sorted(ev)) for ev in s])])
    show_table(["SeqID", "Sequence"], rows, col_width=42)

    # -- 5.2 -----------------------------------------------------------------
    subsection("5.2  GSP frequent patterns at minsup = 50%")
    freq = gsp(SEQ_DATASET, min_support=0.5)
    by_len = defaultdict(list)
    for p, c in freq.items():
        L = sum(len(ev) for ev in p)
        by_len[L].append((p, c))
    for L in sorted(by_len):
        items = sorted(by_len[L], key=lambda x: (-x[1], x[0]))
        line = ", ".join(f"{_fmt_seq(p)}({c})" for p, c in items)
        print(f"    {L}-sequences: {line}")

    # -- 5.3 -----------------------------------------------------------------
    subsection("5.3  ToDo example")
    rows = []
    for i, s in enumerate(TODO_SEQ_DATASET, 1):
        rows.append([i, _fmt_seq([tuple(sorted(ev)) for ev in s])])
    show_table(["SeqID", "Sequence"], rows, col_width=46)

    freq2 = gsp(TODO_SEQ_DATASET, min_support=2/4)
    print(f"\n    Total frequent sequential patterns (minsup=2/4): {len(freq2)}")
    items = sorted(freq2.items(), key=lambda x: (-x[1], x[0]))[:12]
    rows = [[_fmt_seq(p), c] for p, c in items]
    show_table(["Pattern", "Count"], rows, col_width=24)

    # -- 5.4  Optional: pyspark PrefixSpan -----------------------------------
    subsection("5.4  PrefixSpan cross-check")
    require(["prefixspan"])
    from prefixspan import PrefixSpan
    db = [
        [item for ev in seq for item in ev]
        for seq in SEQ_DATASET
    ]
    ps = PrefixSpan(db)
    ps.minlen = 1
    ps.maxlen = 5
    min_count = int(len(SEQ_DATASET) * 0.5)
    ps_results = ps.frequent(min_count)
    print(f"    PrefixSpan found {len(ps_results)} patterns (minsup count >= {min_count}):\n")
    for count, pat in sorted(ps_results, key=lambda x: (-x[0], x[1])):
        print(f"      <{', '.join(str(p) for p in pat)}>  count={count}")

    # -- 5.5  Visualisations -------------------------------------------------
    subsection("5.5  Saving visualisations to ./outputs/association_rules/")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    require(["pandas", "matplotlib", "mlxtend", "numpy"])
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

    # Use the colab dataset for the plots
    dataset = ABCDE_DATASET
    te = TransactionEncoder()
    arr = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(arr, columns=te.columns_)
    fi = fpgrowth(df, min_support=0.4, use_colnames=True)
    rules = association_rules(fi, metric="confidence", min_threshold=0.6)

    _plot_itemset_support(fi, os.path.join(OUTPUT_DIR, "itemset_support.png"))
    _plot_rules_scatter(rules, os.path.join(OUTPUT_DIR, "rules_scatter.png"))
    _plot_rules_network(rules, os.path.join(OUTPUT_DIR, "rules_network.png"))
    _plot_item_cooccurrence(dataset,
                            os.path.join(OUTPUT_DIR, "item_cooccurrence.png"))
    _plot_algo_runtime(df, os.path.join(OUTPUT_DIR, "algo_runtime.png"))
    _plot_sequence_lengths(freq, freq2,
                           os.path.join(OUTPUT_DIR, "sequence_lengths.png"))

    print(f"\n  All plots saved to {OUTPUT_DIR}/")


# ════════════════════════════════════════════════════════════════════════════
#  VISUALISATIONS
# ════════════════════════════════════════════════════════════════════════════

def _plot_itemset_support(fi, save_path):
    import matplotlib.pyplot as plt
    fi_sorted = fi.sort_values("support")
    labels = ["{" + ", ".join(sorted(it)) + "}" for it in fi_sorted["itemsets"]]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(labels)), fi_sorted["support"], color="steelblue")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Support")
    ax.set_title("Frequent itemset support (colab dataset)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  itemset_support.png      -> {save_path}")


def _plot_rules_scatter(rules, save_path):
    import matplotlib.pyplot as plt
    if rules.empty:
        print("  rules_scatter.png        -> skipped (no rules)")
        return
    from collections import defaultdict as _ddict
    fig, ax = plt.subplots(figsize=(11, 7))

    # Group rules that share the same (support, confidence) coordinate
    groups = _ddict(list)
    for _, r in rules.iterrows():
        label = (", ".join(sorted(r["antecedents"])) + " → "
                 + ", ".join(sorted(r["consequents"])))
        key = (round(r["support"], 4), round(r["confidence"], 4))
        groups[key].append(label)

    x = rules["support"].values
    y = rules["confidence"].values
    sc = ax.scatter(x, y, c=rules["lift"], cmap="viridis", s=80,
                    edgecolors="k", linewidths=0.5, zorder=3)
    plt.colorbar(sc, ax=ax, label="lift")

    # Place one annotation box per unique coordinate, stacking rules vertically
    placed = set()
    offsets = {
        (0.4, 1.0):   (120, -30),
        (0.4, 0.667): (120, 30),
    }
    default_offset = (15, 10)
    for (sx, sy), rule_labels in groups.items():
        if (sx, sy) in placed:
            continue
        placed.add((sx, sy))
        text = "\n".join(rule_labels)
        ofs = offsets.get((round(sx, 1), round(sy, 3)), default_offset)
        ax.annotate(
            text, (sx, sy),
            xytext=ofs, textcoords="offset points",
            fontsize=7, linespacing=1.6,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#aaaaaa",
                      alpha=0.9),
            arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.8),
            zorder=5,
        )

    ax.set_xlabel("Support")
    ax.set_ylabel("Confidence")
    ax.set_title("Support x Confidence (colour = lift)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  rules_scatter.png        -> {save_path}")


def _plot_rules_network(rules, save_path):
    """Tiny rule-network plot without networkx."""
    import matplotlib.pyplot as plt
    import numpy as np
    if rules.empty:
        print("  rules_network.png        -> skipped (no rules)")
        return
    # Collect unique antecedent / consequent labels
    items = set()
    edges = []
    for _, r in rules.iterrows():
        ant = ", ".join(sorted(r["antecedents"]))
        con = ", ".join(sorted(r["consequents"]))
        items.add(ant)
        items.add(con)
        edges.append((ant, con, r["lift"]))
    items = sorted(items)
    angles = np.linspace(0, 2 * np.pi, len(items), endpoint=False)
    pos = {it: (np.cos(a), np.sin(a)) for it, a in zip(items, angles)}

    fig, ax = plt.subplots(figsize=(8, 8))
    # Draw edges
    max_lift = max(e[2] for e in edges) or 1
    for a, b, lift in edges:
        x1, y1 = pos[a]
        x2, y2 = pos[b]
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", lw=0.7 + 2 * lift / max_lift,
                            color="steelblue", alpha=0.6),
        )
    # Draw nodes
    for it, (x, y) in pos.items():
        ax.scatter([x], [y], s=900, c="lightyellow", edgecolors="black", zorder=2)
        ax.text(x, y, it, ha="center", va="center", fontsize=9, zorder=3)
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Association rule network (colab dataset)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  rules_network.png        -> {save_path}")


def _plot_item_cooccurrence(dataset, save_path):
    import matplotlib.pyplot as plt
    import numpy as np
    items = sorted({i for t in dataset for i in t})
    idx = {it: k for k, it in enumerate(items)}
    M = np.zeros((len(items), len(items)), dtype=int)
    for t in dataset:
        s = sorted(set(t))
        for a in s:
            for b in s:
                M[idx[a], idx[b]] += 1
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(M, cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="co-occurrence count")
    ax.set_xticks(range(len(items)))
    ax.set_yticks(range(len(items)))
    ax.set_xticklabels(items)
    ax.set_yticklabels(items)
    for i in range(len(items)):
        for j in range(len(items)):
            ax.text(j, i, str(M[i, j]), ha="center", va="center",
                    color="black" if M[i, j] < M.max() / 2 else "white",
                    fontsize=9)
    ax.set_title("Item co-occurrence matrix (colab dataset)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  item_cooccurrence.png    -> {save_path}")


def _plot_algo_runtime(df, save_path):
    import matplotlib.pyplot as plt
    from mlxtend.frequent_patterns import apriori, fpgrowth
    supports = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    t_ap, t_fp = [], []
    n = 100
    for s in supports:
        t0 = time.perf_counter()
        for _ in range(n):
            apriori(df, min_support=s, use_colnames=True)
        t_ap.append((time.perf_counter() - t0) / n * 1000)
        t0 = time.perf_counter()
        for _ in range(n):
            fpgrowth(df, min_support=s, use_colnames=True)
        t_fp.append((time.perf_counter() - t0) / n * 1000)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(supports, t_ap, "o-", label="Apriori",  color="crimson")
    ax.plot(supports, t_fp, "s-", label="FP-Growth", color="steelblue")
    ax.set_xlabel("min_support")
    ax.set_ylabel("avg time per call (ms)")
    ax.set_title("Apriori vs FP-Growth (colab dataset)")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  algo_runtime.png         -> {save_path}")


def _plot_sequence_lengths(freq_first, freq_todo, save_path):
    import matplotlib.pyplot as plt
    from collections import Counter
    def length(pat):
        return sum(len(ev) for ev in pat)
    c_first = Counter(length(p) for p in freq_first)
    cto = Counter(length(p) for p in freq_todo)
    lens = sorted(set(c_first) | set(cto))
    width = 0.4
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([L - width/2 for L in lens], [c_first.get(L, 0) for L in lens],
           width=width, label="first sequence db (minsup=50%)", color="steelblue")
    ax.bar([L + width/2 for L in lens], [cto.get(L, 0) for L in lens],
           width=width, label="ToDo db (minsup=50%)", color="orange")
    ax.set_xlabel("pattern length (total items)")
    ax.set_ylabel("# frequent patterns")
    ax.set_title("Sequential pattern length distribution")
    ax.set_xticks(lens)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  sequence_lengths.png     -> {save_path}")


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Association Rules & Sequential Patterns Tutorial"
    )
    parser.add_argument(
        "--lab", type=int, choices=[1, 2, 3, 4, 5],
        help=(
            "Run a specific lab "
            "(1=basics, 2=Apriori, 3=FP-Growth, 4=CARs, "
            "5=Sequential patterns (GSP + PrefixSpan) + plots)"
        ),
    )
    args = parser.parse_args()

    print("\n" + "#" * 70)
    print("  ASSOCIATION RULES AND SEQUENTIAL PATTERNS  ")
    print("#" * 70)
    print("""
  Labs:
    1 -> Items, transactions, support, confidence
    2 -> Apriori (mlxtend)
    3 -> FP-Growth (mlxtend)
    4 -> Class Association Rules (CARs) + CBA classifier
    5 -> Sequential Patterns (GSP + PrefixSpan) + visualisations
    """)

    labs = {
        1: lab1_basics,
        2: lab2_apriori,
        3: lab3_fpgrowth,
        4: lab4_cars,
        5: lab5_sequential_and_viz,
    }

    if args.lab is not None:
        labs[args.lab]()
    else:
        for fn in labs.values():
            fn()


if __name__ == "__main__":
    main()
