"""
generate_images.py
-------------------
Generates the diagrams used in ``synthetic.md``.

All figures are simple, dependency-light schematic diagrams drawn with
matplotlib (no external assets). Running this script regenerates every PNG
referenced by the tutorial:

    quality_criteria.png
    pipeline_overview.png
    three_strategies.png
    grounding.png
    filtering_methods.png
    evaluation.png

Usage
-----
    python generate_images.py
"""

import os

import matplotlib
matplotlib.use("Agg")  # headless / server-friendly backend
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# -----------------------------------------------------------------------------
# Shared style
# -----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))

NAVY = "#1f2a44"
BLUE = "#3b6ea5"
TEAL = "#2a9d8f"
ORANGE = "#e76f51"
GOLD = "#d9a441"
PURPLE = "#7d6ba8"
LIGHT = "#f3f5f9"
GRAY = "#5b6473"
INK = "#222831"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "savefig.dpi": 160,
    "figure.dpi": 160,
})


def box(ax, x, y, w, h, text, face, edge=None, fc_text="white",
        fontsize=11, weight="bold", rounding=0.06, lw=1.5, align="center"):
    """Draw a rounded rectangle with centered (multi-line) text."""
    edge = edge or face
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.012,rounding_size={rounding}",
        linewidth=lw, edgecolor=edge, facecolor=face, zorder=2,
    )
    ax.add_patch(patch)
    ha = {"center": "center", "left": "left"}[align]
    tx = x + w / 2 if align == "center" else x + 0.04 * w
    ax.text(tx, y + h / 2, text, ha=ha, va="center",
            color=fc_text, fontsize=fontsize, fontweight=weight, zorder=3,
            linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, color=GRAY, lw=2.2, style="-|>", ms=14):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=ms,
        linewidth=lw, color=color, zorder=1,
        shrinkA=2, shrinkB=2,
    ))


def finalize(fig, ax, path, xlim=(0, 10), ylim=(0, 6)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    fig.tight_layout(pad=0.4)
    out = os.path.join(HERE, path)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}")


# -----------------------------------------------------------------------------
# 1. Quality criteria
# -----------------------------------------------------------------------------
def fig_quality_criteria():
    fig, ax = plt.subplots(figsize=(11, 3.4))
    criteria = [
        ("Diversity", "Samples cover\nmany distinct cases", BLUE),
        ("Realism", "Looks like data\nfrom the real world", TEAL),
        ("Coherence", "Internally\nconsistent & fluent", PURPLE),
        ("Novelty", "Not a copy of the\ntraining examples", ORANGE),
        ("Represent-\nativeness", "Matches the target\ndistribution", GOLD),
    ]
    n = len(criteria)
    w, gap = 1.78, 0.22
    total = n * w + (n - 1) * gap
    x0 = (10 - total) / 2
    for i, (title, desc, color) in enumerate(criteria):
        x = x0 + i * (w + gap)
        box(ax, x, 2.55, w, 1.5, title, color, fontsize=12)
        box(ax, x, 0.85, w, 1.45, desc, LIGHT, edge=color,
            fc_text=INK, fontsize=9.5, weight="normal")
    ax.text(5, 5.55, "Quality criteria for synthetic data",
            ha="center", va="center", fontsize=14, fontweight="bold", color=NAVY)
    finalize(fig, ax, "quality_criteria.png", ylim=(0, 6))


# -----------------------------------------------------------------------------
# 2. Pipeline overview
# -----------------------------------------------------------------------------
def fig_pipeline_overview():
    fig, ax = plt.subplots(figsize=(11, 3.2))
    stages = [
        ("1 · GENERATE", "Prompt an LLM to\nproduce new examples", BLUE),
        ("2 · FILTER", "Discard low-quality\nor mislabeled samples", ORANGE),
        ("3 · EVALUATE", "Measure quality and\ndownstream impact", TEAL),
    ]
    w, h = 2.6, 1.7
    gap = 0.85
    x0 = 0.3
    centers = []
    for i, (title, desc, color) in enumerate(stages):
        x = x0 + i * (w + gap)
        box(ax, x, 2.4, w, h, title, color, fontsize=13)
        box(ax, x, 0.55, w, 1.4, desc, LIGHT, edge=color,
            fc_text=INK, fontsize=10, weight="normal")
        centers.append(x + w)
        if i < len(stages) - 1:
            arrow(ax, x + w + 0.08, 3.25, x + w + gap - 0.08, 3.25, color=NAVY)
    ax.text(5.0, 5.45, "The synthetic-data workflow",
            ha="center", va="center", fontsize=14, fontweight="bold", color=NAVY)
    finalize(fig, ax, "pipeline_overview.png", ylim=(0, 6))


# -----------------------------------------------------------------------------
# 3. Three complementary strategies
# -----------------------------------------------------------------------------
def fig_three_strategies():
    fig, ax = plt.subplots(figsize=(11, 4.2))
    pillars = [
        ("GROUNDING", "Condition generation on\nreal sources so output\nstays anchored to facts\nand realistic context.",
         "documents · personas", BLUE),
        ("TAXONOMY-BASED", "Walk a structured tree of\ntopics/attributes so the\nset systematically covers\nthe space.",
         "categories · attributes", PURPLE),
        ("FILTERING", "Generate broadly, then\nkeep only samples that\npass quality and label\nchecks.",
         "likelihood · checks", ORANGE),
    ]
    w, h = 2.7, 2.4
    gap = 0.6
    x0 = 0.4
    for i, (title, desc, tag, color) in enumerate(pillars):
        x = x0 + i * (w + gap)
        box(ax, x, 1.5, w, h, "", LIGHT, edge=color, lw=2.0)
        ax.text(x + w / 2, 3.55, title, ha="center", va="center",
                fontsize=12.5, fontweight="bold", color=color)
        ax.text(x + w / 2, 2.55, desc, ha="center", va="center",
                fontsize=9.6, color=INK, linespacing=1.4)
        box(ax, x + 0.35, 1.62, w - 0.7, 0.42, tag, color,
            fontsize=8.8, rounding=0.2)
    ax.text(5.0, 5.6, "Three basic, complementary strategies",
            ha="center", va="center", fontsize=14, fontweight="bold", color=NAVY)
    ax.text(5.0, 0.75, "They are not mutually exclusive — real pipelines combine all three.",
            ha="center", va="center", fontsize=10.5, style="italic", color=GRAY)
    finalize(fig, ax, "three_strategies.png", ylim=(0, 6.1))


# -----------------------------------------------------------------------------
# 4. Grounding: documents vs personas
# -----------------------------------------------------------------------------
def fig_grounding():
    fig, ax = plt.subplots(figsize=(11, 3.8))
    # central LLM
    box(ax, 4.2, 2.1, 1.6, 1.0, "LLM", NAVY, fontsize=13)
    arrow(ax, 5.8, 2.6, 7.0, 2.6, color=NAVY)
    box(ax, 7.05, 2.05, 2.6, 1.1, "Grounded\nsynthetic sample", TEAL, fontsize=10.5)

    # document grounding (top)
    box(ax, 0.3, 3.55, 2.9, 1.05, "Document source", BLUE, fontsize=11)
    ax.text(1.75, 3.18, "e.g. articles, manuals, reviews", ha="center",
            va="center", fontsize=8.8, color=GRAY)
    arrow(ax, 3.25, 3.7, 4.4, 3.05, color=BLUE)

    # persona grounding (bottom)
    box(ax, 0.3, 0.55, 2.9, 1.05, "Persona source", ORANGE, fontsize=11)
    ax.text(1.75, 0.2, "e.g. 'a busy parent', 'a film critic'", ha="center",
            va="center", fontsize=8.8, color=GRAY)
    arrow(ax, 3.25, 1.1, 4.4, 2.15, color=ORANGE)

    ax.text(5.0, 5.45, "Grounding: anchor generation to a real source",
            ha="center", va="center", fontsize=14, fontweight="bold", color=NAVY)
    finalize(fig, ax, "grounding.png", ylim=(0, 6))


# -----------------------------------------------------------------------------
# 5. Filtering methods
# -----------------------------------------------------------------------------
def fig_filtering_methods():
    fig, ax = plt.subplots(figsize=(11, 4.0))
    methods = [
        ("Sequence likelihood", BLUE,
         "Read the model's token\nlog-probabilities; keep the\nhigh-probability samples.",
         "uses logits"),
        ("Self-consistency", PURPLE,
         "Generate each item several\ntimes; keep it only if the\nanswers agree.",
         "uses repetition"),
        ("Automatic check", ORANGE,
         "Ask an LLM to judge whether\nthe sample is valid and\ncorrectly labeled.",
         "uses LLM-as-judge"),
    ]
    w, h = 2.7, 2.3
    gap = 0.55
    x0 = 0.4
    for i, (title, color, desc, tag) in enumerate(methods):
        x = x0 + i * (w + gap)
        box(ax, x, 1.4, w, h, "", "white", edge=color, lw=2.2)
        ax.text(x + w / 2, 3.35, title, ha="center", va="center",
                fontsize=12, fontweight="bold", color=color)
        ax.text(x + w / 2, 2.55, desc, ha="center", va="center",
                fontsize=9.4, color=INK, linespacing=1.4)
        box(ax, x + 0.45, 1.52, w - 0.9, 0.4, tag, LIGHT, edge=color,
            fc_text=color, fontsize=8.6, rounding=0.2, weight="bold")
    ax.text(5.0, 5.45, "Three ways to filter generated data",
            ha="center", va="center", fontsize=14, fontweight="bold", color=NAVY)
    ax.text(5.0, 0.7, "Keep the good, drop the rest.",
            ha="center", va="center", fontsize=10.5, style="italic", color=GRAY)
    finalize(fig, ax, "filtering_methods.png", ylim=(0, 6))


# -----------------------------------------------------------------------------
# 6. Evaluation: intrinsic vs downstream
# -----------------------------------------------------------------------------
def fig_evaluation():
    fig, ax = plt.subplots(figsize=(11, 3.7))
    box(ax, 4.05, 4.0, 1.9, 0.95, "Synthetic\ndata", NAVY, fontsize=11)

    # intrinsic branch
    arrow(ax, 4.4, 4.0, 2.7, 2.7, color=BLUE)
    box(ax, 0.5, 1.5, 3.6, 1.25, "Intrinsic properties", BLUE, fontsize=11.5)
    ax.text(2.3, 0.95, "diversity · realism · coherence\n(measure the data itself)",
            ha="center", va="center", fontsize=9.2, color=GRAY, linespacing=1.3)

    # downstream branch
    arrow(ax, 5.6, 4.0, 7.3, 2.7, color=TEAL)
    box(ax, 5.9, 1.5, 3.6, 1.25, "Downstream performance", TEAL, fontsize=11.5)
    ax.text(7.7, 0.95, "train a model on it, then\nmeasure accuracy on a real test set",
            ha="center", va="center", fontsize=9.2, color=GRAY, linespacing=1.3)

    ax.text(5.0, 5.55, "Two complementary ways to evaluate",
            ha="center", va="center", fontsize=14, fontweight="bold", color=NAVY)
    finalize(fig, ax, "evaluation.png", ylim=(0, 6))


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating figures...")
    fig_quality_criteria()
    fig_pipeline_overview()
    fig_three_strategies()
    fig_grounding()
    fig_filtering_methods()
    fig_evaluation()
    print("Done.")
