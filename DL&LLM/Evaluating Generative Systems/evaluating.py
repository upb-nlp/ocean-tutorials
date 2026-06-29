"""
evaluating.py
=============
A hands-on tutorial for evaluating a generative system with two complementary
approaches, on a summarization task:

    Stage 1  Generate           summarize documents from a Hugging Face benchmark
    Stage 2  Automated metrics  BLEU, ROUGE-1/2/L, (BERTScore) vs the references
    Stage 3  LLM-as-judge       rate relevance & coherence (Likert) + faithfulness
                                (binary), using the SAME model as the judge
    Stage 4  Compare            aggregate both views and correlate them

A single instruction-tuned model, served with vLLM, does both the generating
and the judging. The reference-based metrics use the lightweight ``rouge-score``
and ``sacrebleu`` libraries; BERTScore is optional.

Run:
    python evaluating.py
"""

import os
import re
import json
import math
from collections import Counter

# Shared Hugging Face cache (matches the lab server). Set before HF imports.
os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", "/export/projects/nlp/.cache"))

import numpy as np
from vllm import LLM, SamplingParams


# =============================================================================
# CONFIG
# =============================================================================
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-8B")
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "4096"))
TENSOR_PARALLEL = int(os.environ.get("TENSOR_PARALLEL", "1"))

DATASET_NAME = os.environ.get("DATASET_NAME", "abisee/cnn_dailymail")
DATASET_CONFIG = os.environ.get("DATASET_CONFIG", "3.0.0")
DATASET_SPLIT = os.environ.get("DATASET_SPLIT", "validation")
INPUT_FIELD = os.environ.get("INPUT_FIELD", "article")
TARGET_FIELD = os.environ.get("TARGET_FIELD", "highlights")

N_SAMPLES = int(os.environ.get("N_SAMPLES", "20"))
MAX_SOURCE_CHARS = int(os.environ.get("MAX_SOURCE_CHARS", "2500"))
SEED = int(os.environ.get("SEED", "0"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs")


# A tiny built-in fallback so the pipeline still runs if the benchmark cannot
# be reached (no network / dataset unavailable).
FALLBACK_DATA = [
    {"article": "The city council approved a new bus network on Tuesday. The plan "
                "adds three express routes and extends service until midnight. "
                "Officials say it will cut average commute times by 15 percent.",
     "highlights": "City council approved an expanded bus network with three new "
                   "express routes and later service."},
    {"article": "Researchers reported a battery that charges to 80 percent in ten "
                "minutes. The prototype uses a silicon anode and survived 1,000 "
                "cycles with little capacity loss in lab tests.",
     "highlights": "A new fast-charging battery prototype reached 80 percent in ten "
                   "minutes and held up over 1,000 cycles."},
    {"article": "The national team won the final 2-1 after a late goal. The captain "
                "scored in the 88th minute, securing the country's first title in "
                "two decades amid celebrations downtown.",
     "highlights": "A late goal gave the national team a 2-1 win and their first "
                   "title in twenty years."},
    {"article": "A wildfire near the coast forced the evacuation of 4,000 residents. "
                "Firefighters contained 30 percent of the blaze by evening, and no "
                "injuries were reported as cooler weather moved in.",
     "highlights": "A coastal wildfire forced 4,000 to evacuate; crews contained 30 "
                   "percent with no injuries reported."},
]


# =============================================================================
# MODEL HELPERS  
# =============================================================================
def build_model():
    print(f"[setup] loading {MODEL_NAME} with vLLM ...")
    return LLM(
        model=MODEL_NAME,
        max_model_len=MAX_MODEL_LEN,
        tensor_parallel_size=TENSOR_PARALLEL,
        trust_remote_code=True,
    )


def render_chat(tokenizer, user_msg, system_msg=None):
    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": user_msg})
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def parse_json_object(text):
    """Best-effort extraction of a JSON object from model output."""
    text = strip_think(text)
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fenced:
        text = fenced.group(1).strip()
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}
    try:
        return json.loads(match.group(0), strict=False)
    except json.JSONDecodeError:
        return {}


# =============================================================================
# DATA — load a slice of an online HF benchmark (Stage 1 input)
# =============================================================================
def load_benchmark():
    """Stream N_SAMPLES examples from a Hugging Face benchmark. Falls back to a
    small built-in sample if the dataset cannot be loaded."""
    try:
        from datasets import load_dataset
        print(f"[data] streaming {DATASET_NAME}/{DATASET_CONFIG} "
              f"[{DATASET_SPLIT}] from Hugging Face ...")
        ds = load_dataset(DATASET_NAME, DATASET_CONFIG,
                          split=DATASET_SPLIT, streaming=True)
        rows = []
        for ex in ds:
            rows.append({"article": ex[INPUT_FIELD], "highlights": ex[TARGET_FIELD]})
            if len(rows) >= N_SAMPLES:
                break
        if rows:
            return rows, DATASET_NAME
    except Exception as e:  # network / dataset issues -> graceful fallback
        print(f"[data] could not load benchmark ({type(e).__name__}: {e}).")
    print("[data] using the built-in fallback sample instead.")
    reps = math.ceil(N_SAMPLES / len(FALLBACK_DATA))
    return (FALLBACK_DATA * reps)[:N_SAMPLES], "builtin-fallback"


# =============================================================================
# STAGE 1 — GENERATION
# =============================================================================
def generate_summaries(llm, tokenizer, rows):
    sampling = SamplingParams(temperature=0.3, top_p=0.9, max_tokens=160, seed=SEED)
    prompts = []
    for r in rows:
        article = r["article"][:MAX_SOURCE_CHARS]
        user_msg = (
            "Summarize the following article in 1-3 concise sentences. "
            "Capture only the key facts; do not add information.\n\n"
            f"Article:\n{article}\n\nSummary:"
        )
        prompts.append(render_chat(tokenizer, user_msg))

    outputs = llm.generate(prompts, sampling, use_tqdm=False)
    return [strip_think(o.outputs[0].text) for o in outputs]


# =============================================================================
# METRIC PRIMITIVES  (standard library only)
# =============================================================================
def _tokenize(text):
    """Lowercase, word-boundary tokenizer."""
    return re.findall(r'\b\w+\b', text.lower())


def _ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _rouge_n_f1(pred_tokens, ref_tokens, n):
    pred_counts = Counter(_ngrams(pred_tokens, n))
    ref_counts = Counter(_ngrams(ref_tokens, n))
    overlap = sum((pred_counts & ref_counts).values())
    pred_total = sum(pred_counts.values())
    ref_total = sum(ref_counts.values())
    if pred_total == 0 or ref_total == 0:
        return 0.0
    p = overlap / pred_total
    r = overlap / ref_total
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def _lcs_length(a, b):
    """LCS length via dynamic programming (two-row, O(n) memory)."""
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def _rouge_l_f1(pred_tokens, ref_tokens):
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    p = lcs / len(pred_tokens)
    r = lcs / len(ref_tokens)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def _corpus_bleu(predictions, references, max_n=4):
    """Corpus BLEU with brevity penalty (no smoothing)."""
    matches = [0] * max_n
    totals = [0] * max_n
    hyp_len = 0
    ref_len = 0
    for pred, ref in zip(predictions, references):
        p_tok = _tokenize(pred)
        r_tok = _tokenize(ref)
        hyp_len += len(p_tok)
        ref_len += len(r_tok)
        for n in range(1, max_n + 1):
            p_counts = Counter(_ngrams(p_tok, n))
            r_counts = Counter(_ngrams(r_tok, n))
            matches[n - 1] += sum(min(c, r_counts[g]) for g, c in p_counts.items())
            totals[n - 1] += sum(p_counts.values())
    if hyp_len == 0 or any(t == 0 for t in totals):
        return 0.0
    log_avg = sum(
        math.log(matches[n] / totals[n]) if matches[n] > 0 else float('-inf')
        for n in range(max_n)
    ) / max_n
    if log_avg == float('-inf'):
        return 0.0
    bp = 1.0 if hyp_len >= ref_len else math.exp(1 - ref_len / hyp_len)
    return bp * math.exp(log_avg) * 100


# =============================================================================
# STAGE 2 — AUTOMATED, REFERENCE-BASED METRICS
# =============================================================================
def compute_automated_metrics(predictions, references):
    """ROUGE-1/2/L and BLEU (standard library). BERTScore if bert-score is installed."""
    r1, r2, rl_scores, rouge_l_per_item = [], [], [], []
    for pred, ref in zip(predictions, references):
        p_tok = _tokenize(pred)
        r_tok = _tokenize(ref)
        r1.append(_rouge_n_f1(p_tok, r_tok, 1))
        r2.append(_rouge_n_f1(p_tok, r_tok, 2))
        score = _rouge_l_f1(p_tok, r_tok)
        rl_scores.append(score)
        rouge_l_per_item.append(score)

    bleu = _corpus_bleu(predictions, references)

    metrics = {
        "rouge1_f": round(float(np.mean(r1)), 4),
        "rouge2_f": round(float(np.mean(r2)), 4),
        "rougeL_f": round(float(np.mean(rl_scores)), 4),
        "bleu": round(float(bleu), 4),
    }

    # BERTScore is optional (heavy: downloads a contextual encoder).
    try:
        from bert_score import score as bertscore
        _, _, f1 = bertscore(predictions, references, lang="en", verbose=False)
        metrics["bertscore_f1"] = round(float(f1.mean().item()), 4)
    except ImportError:
        print("[metrics] BERTScore skipped (bert-score not installed).")
        metrics["bertscore_f1"] = None

    return metrics, rouge_l_per_item


# =============================================================================
# STAGE 3 — LLM-AS-JUDGE  (Likert relevance/coherence + binary faithfulness)
# =============================================================================
JUDGE_SYSTEM = (
    "You are a careful, impartial evaluation judge. You assess a candidate "
    "summary against the source article and report scores in strict JSON."
)


def judge_summaries(llm, tokenizer, rows, predictions):
    sampling = SamplingParams(temperature=0.0, max_tokens=200, seed=SEED)
    prompts = []
    for r, pred in zip(rows, predictions):
        article = r["article"][:MAX_SOURCE_CHARS]
        user_msg = (
            "Evaluate the SUMMARY of the ARTICLE below.\n\n"
            f"ARTICLE:\n{article}\n\n"
            f"SUMMARY:\n{pred}\n\n"
            "Rate the summary on:\n"
            "- relevance: 1-5, does it capture the article's key points?\n"
            "- coherence: 1-5, is it fluent and well-structured?\n"
            "- faithful: \"yes\" or \"no\", is every claim supported by the article?\n\n"
            "Return ONLY strict JSON with exactly these keys: "
            '{"relevance": <1-5>, "coherence": <1-5>, "faithful": "yes|no"}.'
        )
        prompts.append(render_chat(tokenizer, user_msg, system_msg=JUDGE_SYSTEM))

    outputs = llm.generate(prompts, sampling, use_tqdm=False)

    verdicts, overall_per_item = [], []
    for o in outputs:
        obj = parse_json_object(o.outputs[0].text)
        rel = _clamp_int(obj.get("relevance"), 1, 5)
        coh = _clamp_int(obj.get("coherence"), 1, 5)
        faith = str(obj.get("faithful", "")).strip().lower().startswith("y")
        verdicts.append({"relevance": rel, "coherence": coh, "faithful": faith})
        overall_per_item.append((rel + coh) / 2.0 if rel and coh else None)
    return verdicts, overall_per_item


def _clamp_int(v, lo, hi):
    try:
        return max(lo, min(hi, int(round(float(v)))))
    except (TypeError, ValueError):
        return None


def aggregate_judge(verdicts):
    rels = [v["relevance"] for v in verdicts if v["relevance"] is not None]
    cohs = [v["coherence"] for v in verdicts if v["coherence"] is not None]
    faiths = [v["faithful"] for v in verdicts]
    return {
        "mean_relevance": round(float(np.mean(rels)), 3) if rels else None,
        "mean_coherence": round(float(np.mean(cohs)), 3) if cohs else None,
        "faithfulness_rate": round(float(np.mean(faiths)), 3) if faiths else None,
        "n_parsed": len(rels),
    }


# =============================================================================
# STAGE 4 — COMPARE THE TWO VIEWS
# =============================================================================
def correlate(a, b):
    """Pearson correlation over items where both views produced a number."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    xs, ys = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    if xs.std() == 0 or ys.std() == 0:
        return None
    return round(float(np.corrcoef(xs, ys)[0, 1]), 3)


# =============================================================================
# MAIN
# =============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    llm = build_model()
    tokenizer = llm.get_tokenizer()

    rows, source = load_benchmark()
    references = [r["highlights"] for r in rows]
    print(f"[data] {len(rows)} examples from '{source}'")

    # ---- Stage 1: generate -------------------------------------------------
    print("\n[Stage 1] generating summaries ...")
    predictions = generate_summaries(llm, tokenizer, rows)

    # ---- Stage 2: automated metrics ---------------------------------------
    print("[Stage 2] computing automated metrics (BLEU / ROUGE / BERTScore) ...")
    auto_metrics, rouge_l_per_item = compute_automated_metrics(predictions, references)

    # ---- Stage 3: LLM-as-judge --------------------------------------------
    print("[Stage 3] running LLM-as-judge (relevance, coherence, faithfulness) ...")
    verdicts, overall_per_item = judge_summaries(llm, tokenizer, rows, predictions)
    judge_metrics = aggregate_judge(verdicts)

    # ---- Stage 4: compare --------------------------------------------------
    rho = correlate(rouge_l_per_item, overall_per_item)

    # ---- persist ----------------------------------------------------------
    per_item = []
    for r, pred, rl, v, ov in zip(rows, predictions, rouge_l_per_item,
                                  verdicts, overall_per_item):
        per_item.append({
            "reference": r["highlights"],
            "prediction": pred,
            "rougeL_f": round(rl, 4),
            "judge": v,
            "judge_overall": ov,
        })
    with open(os.path.join(OUTPUT_DIR, "generations.json"), "w", encoding="utf-8") as f:
        json.dump(per_item, f, indent=2, ensure_ascii=False)

    metrics = {
        "model": MODEL_NAME,
        "benchmark": source,
        "n_samples": len(rows),
        "automated_metrics": auto_metrics,
        "llm_judge": judge_metrics,
        "corr_rougeL_vs_judge": rho,
    }
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # ---- report -----------------------------------------------------------
    print("\n===== AUTOMATED METRICS (reference-based) =====")
    for k, v in auto_metrics.items():
        print(f"  {k:<14} {v}")
    print("\n===== LLM-AS-JUDGE (reference-free) =====")
    for k, v in judge_metrics.items():
        print(f"  {k:<18} {v}")
    print("\n===== AGREEMENT =====")
    print(f"  Pearson corr (ROUGE-L vs judge overall): {rho}")
    if rho is not None and rho < 0.5:
        print("  -> the two views diverge: overlap metrics and judgments "
              "capture different things.")
    print("\nSaved: outputs/generations.json, outputs/metrics.json")


if __name__ == "__main__":
    main()
