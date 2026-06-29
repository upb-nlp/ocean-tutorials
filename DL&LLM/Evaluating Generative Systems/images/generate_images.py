"""
generate_images.py
-------------------
Generates the diagrams used in ``evaluating.md``.

All figures are simple, dependency-light schematic diagrams drawn with
matplotlib. Running this script regenerates every PNG referenced by the
tutorial:

    metrics_overview.png
    automatic_vs_judge.png
    judge_workflow.png
    judge_pipeline.png
    eval_designs.png
    benchmark_taxonomy.png
    leaderboard_caution.png
    rag_agentic_eval.png

Usage
-----
    python generate_images.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))

# Core palette (shared with the synthetic-data tutorial for consistency)
NAVY = "#1f2a44"
BLUE = "#3b6ea5"
TEAL = "#2a9d8f"
ORANGE = "#e76f51"
GOLD = "#d9a441"
PURPLE = "#7d6ba8"
RED = "#c1503f"
LIGHT = "#f3f5f9"
GRAY = "#5b6473"
INK = "#222831"

# Pastel panel fills (for the Image-2-style pipeline)
PEACH, PEACH_E = "#fbe7da", "#e0875f"
MINT, MINT_E = "#e6f0e1", "#5f9e63"
CREAM, CREAM_E = "#fdf1cf", "#cda23a"
PERI, PERI_E = "#e7e8f6", "#6f72b8"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "savefig.dpi": 160,
    "figure.dpi": 160,
})


def box(ax, x, y, w, h, text, face, edge=None, fc_text="white",
        fontsize=11, weight="bold", rounding=0.06, lw=1.5, va="center"):
    edge = edge or face
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0.012,rounding_size={rounding}",
        linewidth=lw, edgecolor=edge, facecolor=face, zorder=2))
    yy = y + h / 2 if va == "center" else y + h - 0.18
    ax.text(x + w / 2, yy, text, ha="center", va="center",
            color=fc_text, fontsize=fontsize, fontweight=weight, zorder=3,
            linespacing=1.3)


def arrow(ax, x1, y1, x2, y2, color=GRAY, lw=2.2, style="-|>", ms=14, rad=0.0):
    cs = f"arc3,rad={rad}" if rad else None
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=ms,
        linewidth=lw, color=color, zorder=1, shrinkA=2, shrinkB=2,
        connectionstyle=cs))


def chevron(ax, x, y, color=GRAY):
    ax.text(x, y, "\u279C", ha="center", va="center", fontsize=24,
            color=color, zorder=3, fontweight="bold")


def title(ax, text, y, x=None, fs=15):
    ax.text(x if x is not None else (ax.get_xlim()[1] / 2), y, text,
            ha="center", va="center", fontsize=fs, fontweight="bold", color=NAVY)


def finalize(fig, ax, path, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(HERE, path), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}")


# -----------------------------------------------------------------------------
# 1. Reference-based automated metrics
# -----------------------------------------------------------------------------
def fig_metrics_overview():
    fig, ax = plt.subplots(figsize=(11, 4.4))
    box(ax, 0.4, 4.2, 3.0, 1.0, "System output", BLUE, fontsize=11.5)
    box(ax, 0.4, 2.7, 3.0, 1.0, "Reference\n(gold answer)", TEAL, fontsize=11)
    box(ax, 4.6, 3.35, 2.2, 1.4, "Metric", NAVY, fontsize=12.5)
    box(ax, 7.7, 3.55, 2.0, 1.0, "Score", GOLD, fontsize=12.5)
    arrow(ax, 3.4, 4.55, 4.55, 4.2, color=BLUE)
    arrow(ax, 3.4, 3.05, 4.55, 3.9, color=TEAL)
    arrow(ax, 6.8, 4.05, 7.65, 4.05, color=NAVY)

    # surface <-> semantic axis with the three metrics
    ax.annotate("", xy=(9.7, 1.2), xytext=(0.4, 1.2),
                arrowprops=dict(arrowstyle="-|>", lw=2, color=GRAY))
    ax.text(0.5, 1.55, "surface-form overlap", fontsize=9.5, color=GRAY, ha="left")
    ax.text(9.6, 1.55, "semantic similarity", fontsize=9.5, color=GRAY, ha="right")
    chips = [("BLEU", "n-gram precision", 1.9, BLUE),
             ("ROUGE", "n-gram recall", 4.6, PURPLE),
             ("BERTScore", "embedding match", 7.9, TEAL)]
    for name, desc, cx, c in chips:
        box(ax, cx - 1.05, 0.25, 2.1, 0.7, name, c, fontsize=11)
        ax.text(cx, -0.18, desc, ha="center", va="center", fontsize=8.8, color=GRAY)
        arrow(ax, cx, 0.98, cx, 1.18, color=c, lw=1.6, ms=9)
    title(ax, "Reference-based automated metrics", 5.85)
    finalize(fig, ax, "metrics_overview.png", (0, 10), (-0.6, 6.2))


# -----------------------------------------------------------------------------
# 2. Automated metrics vs LLM-judge / human
# -----------------------------------------------------------------------------
def fig_automatic_vs_judge():
    fig, ax = plt.subplots(figsize=(11, 4.6))
    # left card
    box(ax, 0.4, 1.3, 4.2, 3.4, "", LIGHT, edge=BLUE, lw=2.0)
    ax.text(2.5, 4.35, "Automated metrics", ha="center", fontsize=12.5,
            fontweight="bold", color=BLUE)
    for i, t in enumerate([
        "need a reference answer",
        "measure overlap / similarity only",
        "cheap, fast, reproducible",
        "rigid: miss valid paraphrases",
    ]):
        ax.text(0.75, 3.75 - i * 0.62, "\u2022 " + t, fontsize=10, color=INK, ha="left")

    # right card
    box(ax, 5.4, 1.3, 5.2, 3.4, "", LIGHT, edge=ORANGE, lw=2.0)
    ax.text(8.0, 4.35, "LLM-as-judge  /  Human evaluation", ha="center",
            fontsize=12.5, fontweight="bold", color=ORANGE)
    for i, t in enumerate([
        "work without a reference",
        "flexible criteria (helpful, safe, faithful...)",
        "same workflow \u2014 swap the evaluator",
        "LLM: cheap & fast, but biased",
        "Human: costly & slower, but most real",
    ]):
        ax.text(5.75, 3.75 - i * 0.58, "\u2022 " + t, fontsize=10, color=INK, ha="left")

    arrow(ax, 4.6, 3.0, 5.4, 3.0, color=NAVY)
    ax.text(5.0, 3.35, "solve", fontsize=8.5, color=GRAY, ha="center")
    title(ax, "When overlap is not enough", 5.55)
    finalize(fig, ax, "automatic_vs_judge.png", (0, 11), (1.0, 6.0))


# -----------------------------------------------------------------------------
# 3. Building an LLM-as-a-judge: iterate with examples  (inspired by Image 1)
# -----------------------------------------------------------------------------
def fig_judge_workflow():
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    stages = [
        ("Thinking", BLUE, ["What to evaluate?", "How do humans judge?",
                            "Any reliable examples?"]),
        ("Prompt Design", PURPLE, ["Scoring dimension", "Relative comparison",
                                   "Few-shot example"]),
        ("Model Selection", TEAL, ["Capable model", "Strong reasoning",
                                   "Good at following instructions"]),
        ("Specification", ORANGE, ["The score is: XX", r"\boxed{XX}", "Yes / No"]),
    ]
    w, h = 2.7, 3.1
    gap = 0.45
    x0 = 0.5
    headers_x = []
    for i, (name, color, items) in enumerate(stages):
        x = x0 + i * (w + gap)
        box(ax, x, 0.9, w, h, "", "white", edge=color, lw=2.0)
        box(ax, x + 0.2, 3.45, w - 0.4, 0.55, name, color, fontsize=11.5)
        for j, it in enumerate(items):
            box(ax, x + 0.22, 2.62 - j * 0.78, w - 0.44, 0.6, it, LIGHT,
                edge=color, fc_text=INK, fontsize=8.8, weight="normal")
        headers_x.append(x + w / 2)

    # forward "Test with cases" arrow
    arrow(ax, x0 + 0.2, 0.45, x0 + 4 * w + 3 * gap - 0.2, 0.45, color=BLUE, lw=3, ms=18)
    ax.text((x0 + x0 + 4 * w + 3 * gap) / 2, 0.12, "Test with cases",
            ha="center", fontsize=10.5, color=BLUE, fontweight="bold")

    # curved "Retest" arrow from Specification header back to Thinking header
    arrow(ax, headers_x[-1], 4.15, headers_x[0], 4.15, color=NAVY, lw=2.2, ms=16, rad=0.46)
    ax.text((headers_x[0] + headers_x[-1]) / 2, 6.5, "Retest",
            ha="center", fontsize=11, color=NAVY, fontweight="bold")
    ax.text(headers_x[0] - 1.35, 7.65, "Build a judge by iterating quickly on examples",
            ha="left", fontsize=14, fontweight="bold", color=NAVY)
    finalize(fig, ax, "judge_workflow.png", (0, 13), (-0.1, 8.1))


# -----------------------------------------------------------------------------
# 4. LLM-as-a-Judge evaluation pipeline  (inspired by Image 2)
# -----------------------------------------------------------------------------
def fig_judge_pipeline():
    fig, ax = plt.subplots(figsize=(14.2, 5.4))
    # inputs
    box(ax, 0.2, 2.2, 1.7, 1.5, "Text\nImage\nVideo", "white", edge=GRAY,
        fc_text=INK, fontsize=10, weight="normal", lw=1.6)
    ax.text(1.05, 1.85, "Inputs", ha="center", fontsize=10.5, fontweight="bold", color=INK)
    chevron(ax, 2.25, 2.95, GRAY)

    panels = [
        ("In-Context Learning", PEACH, PEACH_E,
         ["Scores  (rate 1\u201310)", "Yes / No  (is it supported?)",
          "Pairs  (which is better?)", "Multiple-choice  (pick valid ones)"]),
        ("Model Selection", MINT, MINT_E,
         ["General LLM", "  \u2013 closed-source", "  \u2013 open-source",
          "Fine-tuned judge LLM"]),
        ("Post-Processing", CREAM, CREAM_E,
         ["Special tokens  ('The score is 4')",
          "Logits  (P over 'Yes' tokens)", "Selected sentences"]),
        ("Evaluation", PERI, PERI_E,
         ["Numbers   \u2192  4", "Options   \u2192  Response 1",
          "Probability  \u2192  0.328", "Choices   \u2192  A, C, D"]),
    ]
    pw, ph = 2.55, 3.4
    gap = 0.55
    x0 = 2.7
    for i, (name, fill, edge, items) in enumerate(panels):
        x = x0 + i * (pw + gap)
        box(ax, x, 1.4, pw, ph, "", fill, edge=edge, lw=1.8)
        ax.text(x + pw / 2, 4.5, name, ha="center", fontsize=11,
                fontweight="bold", color=edge)
        for j, it in enumerate(items):
            ax.text(x + 0.18, 4.0 - j * 0.62, it, fontsize=8.4, color=INK, ha="left")
        if i < len(panels) - 1:
            chevron(ax, x + pw + gap / 2, 2.95, GRAY)

    title(ax, "LLM-as-a-Judge evaluation pipeline", 5.45, x=7.45)
    ax.text(7.45, 4.95, "to grade / verify / rank data, models, or agents",
            ha="center", fontsize=10, style="italic", color=BLUE)
    finalize(fig, ax, "judge_pipeline.png", (0, 14.7), (0.6, 5.9))


# -----------------------------------------------------------------------------
# 5. Common evaluation designs
# -----------------------------------------------------------------------------
def fig_eval_designs():
    fig, ax = plt.subplots(figsize=(11, 3.8))
    designs = [
        ("Likert", BLUE, "rate on a scale", "1  2  3  4  5"),
        ("Binary", TEAL, "single yes/no judgment", "Yes   /   No"),
        ("Multi-choice", PURPLE, "pick the valid option(s)", "A   B   C   D"),
        ("Pairwise", ORANGE, "which output is better", "A   vs   B"),
    ]
    w, h = 2.25, 2.2
    gap = 0.3
    x0 = 0.35
    for i, (name, color, desc, sample) in enumerate(designs):
        x = x0 + i * (w + gap)
        box(ax, x, 1.2, w, h, "", "white", edge=color, lw=2.0)
        box(ax, x + 0.18, 2.75, w - 0.36, 0.5, name, color, fontsize=11)
        ax.text(x + w / 2, 2.25, desc, ha="center", fontsize=8.8, color=GRAY)
        box(ax, x + 0.35, 1.42, w - 0.7, 0.6, sample, LIGHT, edge=color,
            fc_text=color, fontsize=11, weight="bold")
    title(ax, "Common LLM / human evaluation designs", 4.35)
    finalize(fig, ax, "eval_designs.png", (0, 10.5), (0.9, 4.9))


# -----------------------------------------------------------------------------
# 6. Benchmark taxonomy
# -----------------------------------------------------------------------------
def fig_benchmark_taxonomy():
    fig, ax = plt.subplots(figsize=(12, 4.6))
    box(ax, 4.1, 4.0, 2.3, 0.95, "Benchmarks", NAVY, fontsize=12.5)
    cats = [
        ("General", BLUE, ["Linguistics", "Knowledge", "Reasoning"]),
        ("Domain-specific", TEAL, ["Natural Sciences", "Humanities & Soc. Sci.",
                                   "Engineering & Tech"]),
        ("Task-specific", PURPLE, ["Risk & Reliability", "Safety", "..."]),
        ("Multimodal", ORANGE, ["Text + Image", "Text + Video", "Audio ..."]),
    ]
    w = 2.5
    gap = 0.3
    x0 = 0.35
    for i, (name, color, items) in enumerate(cats):
        x = x0 + i * (w + gap)
        cx = x + w / 2
        box(ax, x, 2.55, w, 0.7, name, color, fontsize=11)
        arrow(ax, 5.25, 3.95, cx, 3.3, color=color, lw=1.6, ms=11)
        for j, it in enumerate(items):
            box(ax, x + 0.12, 1.7 - j * 0.62, w - 0.24, 0.5, it, LIGHT, edge=color,
                fc_text=INK, fontsize=8.8, weight="normal")
    title(ax, "A rough classification of benchmarks", 5.55)
    finalize(fig, ax, "benchmark_taxonomy.png", (0, 11.5), (-0.2, 6.1))


# -----------------------------------------------------------------------------
# 7. Leaderboards: read with caution
# -----------------------------------------------------------------------------
def fig_leaderboard_caution():
    fig, ax = plt.subplots(figsize=(11, 4.4))
    # mock leaderboard
    box(ax, 0.4, 1.0, 4.4, 3.6, "", "white", edge=BLUE, lw=2.0)
    ax.text(2.6, 4.25, "Leaderboard", ha="center", fontsize=12, fontweight="bold", color=BLUE)
    rows = [("1", "Model A", "87.4"), ("2", "Model B", "86.9"),
            ("3", "Model C", "85.1"), ("4", "Model D", "82.7")]
    box(ax, 0.7, 3.45, 3.8, 0.5, "rank      model           score", LIGHT,
        edge=BLUE, fc_text=INK, fontsize=9.5, weight="bold")
    for j, (r, m, s) in enumerate(rows):
        c = GOLD if j == 0 else LIGHT
        tc = "white" if j == 0 else INK
        box(ax, 0.7, 2.8 - j * 0.55, 3.8, 0.46,
            f"{r}        {m}          {s}", c, edge=BLUE if j else GOLD,
            fc_text=tc, fontsize=9.4, weight="normal")

    arrow(ax, 4.9, 2.8, 5.8, 2.8, color=RED, lw=2.4)
    ax.text(5.35, 3.15, "but...", fontsize=9.5, color=RED, ha="center", style="italic")

    # caution panel
    box(ax, 5.9, 1.0, 4.7, 3.6, "", "#fdf0ee", edge=RED, lw=2.0)
    ax.text(8.25, 4.25, "Treat rankings with caution", ha="center",
            fontsize=12, fontweight="bold", color=RED)
    cautions = [
        ("Data contamination", "test items may have leaked into training"),
        ("Evaluation heterogeneity", "design choices change the results"),
        ("Sensitivity to biases", "order, length, formatting effects"),
    ]
    for j, (hd, sub) in enumerate(cautions):
        ax.text(6.2, 3.6 - j * 0.95, "\u26A0  " + hd, fontsize=10.3,
                fontweight="bold", color=INK, ha="left")
        ax.text(6.5, 3.25 - j * 0.95, sub, fontsize=8.8, color=GRAY, ha="left")
    title(ax, "Benchmarks publish leaderboards \u2014 read them critically", 5.55, x=5.5)
    finalize(fig, ax, "leaderboard_caution.png", (0, 11), (0.6, 6.1))


# -----------------------------------------------------------------------------
# 8. RAG and agentic evaluation
# -----------------------------------------------------------------------------
def fig_rag_agentic():
    fig, ax = plt.subplots(figsize=(11, 4.6))
    # RAG panel
    box(ax, 0.4, 1.0, 4.5, 3.5, "", LIGHT, edge=TEAL, lw=2.0)
    ax.text(2.65, 4.15, "RAG evaluation", ha="center", fontsize=12.5,
            fontweight="bold", color=TEAL)
    ax.text(2.65, 3.6, "retrieve, then generate", ha="center", fontsize=9,
            style="italic", color=GRAY)
    box(ax, 0.75, 2.45, 1.85, 0.85, "Retrieval", TEAL, fontsize=10)
    box(ax, 2.95, 2.45, 1.6, 0.85, "Generation", BLUE, fontsize=10)
    for j, t in enumerate(["context precision / recall", "faithfulness to sources",
                           "answer relevance"]):
        ax.text(0.75, 2.0 - j * 0.45, "\u2022 " + t, fontsize=9, color=INK, ha="left")

    # Agentic panel
    box(ax, 5.4, 1.0, 5.2, 3.5, "", LIGHT, edge=ORANGE, lw=2.0)
    ax.text(8.0, 4.15, "Agentic evaluation", ha="center", fontsize=12.5,
            fontweight="bold", color=ORANGE)
    ax.text(8.0, 3.6, "multi-step, tool-using systems", ha="center", fontsize=9,
            style="italic", color=GRAY)
    for j, t in enumerate(["task success rate (did it finish the goal?)",
                           "tool-use correctness",
                           "trajectory / step efficiency",
                           "cost & safety along the way"]):
        ax.text(5.75, 3.05 - j * 0.5, "\u2022 " + t, fontsize=9.3, color=INK, ha="left")
    title(ax, "Evaluating RAG and agentic systems", 5.55, x=5.5)
    finalize(fig, ax, "rag_agentic_eval.png", (0, 11), (0.6, 6.1))


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating figures...")
    fig_metrics_overview()
    fig_automatic_vs_judge()
    fig_judge_workflow()
    fig_judge_pipeline()
    fig_eval_designs()
    fig_benchmark_taxonomy()
    fig_leaderboard_caution()
    fig_rag_agentic()
    print("Done.")
