"""
synthetic_sentiment_transformer_downstream.py
============================================
Synthetic-data generation for a binary sentiment-classification task.

The script compares three training sets:

    Stage A  Naive generation
             Plain prompts produce labeled positive/negative reviews.

    Stage B  Persona-grounded generation
             Prompts add personas and concrete review domains.

    Stage C  Filtering and selection
             The grounded pool is scored with:
               1. a label-confidence check based on first-token logprobs
               2. an LLM-as-judge rating: GOOD, BORDERLINE, or BAD

             The filtered dataset is selected from the scored pool while
             preserving class balance and domain coverage.

    Stage D  Downstream evaluation
             A transformer sequence classifier is fine-tuned separately on each
             dataset and evaluated on a fixed held-out sentiment test set.

Run:
    python synthetic_sentiment_transformer_downstream.py
"""

import os
import re
import gc
import json
import math
import random
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from vllm import LLM, SamplingParams


# =============================================================================
# CONFIG
# =============================================================================
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-8B")
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "4096"))
TENSOR_PARALLEL = int(os.environ.get("TENSOR_PARALLEL", "1"))

N_PER_CLASS = int(os.environ.get("N_PER_CLASS", "500"))
REVIEWS_PER_PROMPT = int(os.environ.get("REVIEWS_PER_PROMPT", "5"))
POOL_MULTIPLIER = int(os.environ.get("POOL_MULTIPLIER", "3"))

LABEL_CONF_THRESHOLD = float(os.environ.get("LABEL_CONF_THRESHOLD", "0.55"))
MAX_PER_DOMAIN_PER_LABEL = int(os.environ.get("MAX_PER_DOMAIN_PER_LABEL", "0"))

DOWNSTREAM_MODEL = os.environ.get("DOWNSTREAM_MODEL", "distilbert-base-uncased")
DOWNSTREAM_EPOCHS = int(os.environ.get("DOWNSTREAM_EPOCHS", "3"))
DOWNSTREAM_BATCH_SIZE = int(os.environ.get("DOWNSTREAM_BATCH_SIZE", "16"))
DOWNSTREAM_LR = float(os.environ.get("DOWNSTREAM_LR", "2e-5"))
DOWNSTREAM_MAX_LENGTH = int(os.environ.get("DOWNSTREAM_MAX_LENGTH", "160"))
DOWNSTREAM_WARMUP_RATIO = float(os.environ.get("DOWNSTREAM_WARMUP_RATIO", "0.06"))

SEED = int(os.environ.get("SEED", "0"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs")

LABELS = ["positive", "negative"]
LABEL_TO_ID = {"negative": 0, "positive": 1}
ID_TO_LABEL = {0: "negative", 1: "positive"}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =============================================================================
# GENERATION MATERIAL
# =============================================================================
PERSONAS = [
    "a budget-conscious student who compares every purchase carefully",
    "a busy parent who values convenience and reliability",
    "a retired engineer who notices build quality and durability",
    "a frequent business traveller who depends on things working smoothly",
    "a skeptical reviewer who is not easily impressed",
    "a cheerful optimist who still mentions practical details",
    "a first-time buyer who explains what was confusing or pleasant",
    "a detail-oriented professional who writes precise reviews",
    "an impatient customer who dislikes wasted time",
    "a careful shopper who reads reviews before buying",
    "a hobbyist who pays attention to small features",
    "a practical user who cares more about everyday use than hype",
]

DOMAINS = [
    "a pair of wireless headphones",
    "a budget smartphone",
    "a kitchen blender",
    "a local Italian restaurant",
    "a meal delivery order",
    "a pair of running shoes",
    "a productivity app",
    "a robot vacuum",
    "an espresso machine",
    "a paperback novel",
    "a hotel stay",
    "a streaming TV series",
    "a grocery delivery service",
    "a children's toy",
    "a laptop backpack",
    "a coffee shop",
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


def release_generation_model(llm):
    """Release generation-model memory before downstream training."""
    try:
        del llm
    except Exception:
        pass

    try:
        from vllm.distributed.parallel_state import (
            destroy_model_parallel,
            destroy_distributed_environment,
        )
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        pass

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def render_chat(tokenizer, user_msg, system_msg=None):
    """Apply the model chat template."""
    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": user_msg})

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def strip_think(text):
    """Remove model reasoning blocks when present."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def parse_json_array(text):
    """Extract a JSON array of strings from model output."""
    text = strip_think(text)

    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fenced:
        text = fenced.group(1).strip()

    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        return []

    try:
        data = json.loads(match.group(0), strict=False)
    except json.JSONDecodeError:
        return []

    out = []
    for x in data:
        if isinstance(x, (str, int, float)):
            s = re.sub(r"\s+", " ", str(x)).strip()
            if s:
                out.append(s)
    return out


# =============================================================================
# STAGE A & B: GENERATION
# =============================================================================
def make_generation_prompt(label, index, grounded):
    """Build a generation prompt for one batch of reviews."""
    if not grounded:
        return (
            f"Write {REVIEWS_PER_PROMPT} short {label} customer reviews. "
            "Each review should be 1-2 sentences.\n"
            "Return ONLY a JSON array of strings."
        )

    persona = PERSONAS[index % len(PERSONAS)]
    domain = DOMAINS[index % len(DOMAINS)]

    if label == "positive":
        polarity_note = (
            "The overall sentiment must be positive. Minor drawbacks are allowed "
            "when the review remains clearly positive overall."
        )
    else:
        polarity_note = (
            "The overall sentiment must be negative. Minor positives are allowed "
            "when the review remains clearly negative overall."
        )

    return (
        f"You are {persona}.\n"
        f"Write {REVIEWS_PER_PROMPT} short {label} customer reviews about {domain}. "
        f"{polarity_note} Each review should sound like a real user review and include "
        "at least one concrete detail.\n"
        "Return ONLY a JSON array of strings."
    )


def generate_dataset(llm, tokenizer, grounded, pool_multiplier=1):
    """Generate a balanced labeled review dataset."""
    sampling = SamplingParams(temperature=0.9, top_p=0.95, max_tokens=700)

    target = N_PER_CLASS * pool_multiplier
    records = []
    max_rounds = 8
    prompt_offset = 0

    for label in LABELS:
        seen = set()
        unique = []
        rounds = 0

        while len(unique) < target and rounds < max_rounds:
            remaining = target - len(unique)
            n_prompts = max(1, math.ceil(remaining / REVIEWS_PER_PROMPT))

            prompts = []
            metadata = []
            for j in range(n_prompts):
                i = prompt_offset + j
                domain = DOMAINS[i % len(DOMAINS)] if grounded else "general customer review"
                persona = PERSONAS[i % len(PERSONAS)] if grounded else "none"
                prompts.append(render_chat(tokenizer, make_generation_prompt(label, i, grounded)))
                metadata.append({"domain": domain, "persona": persona, "prompt_index": i})

            outputs = llm.generate(prompts, sampling, use_tqdm=False)

            for meta, out in zip(metadata, outputs):
                for review in parse_json_array(out.outputs[0].text):
                    text = re.sub(r"\s+", " ", review).strip()
                    key = text.lower()
                    if key not in seen and len(text) >= 20:
                        seen.add(key)
                        unique.append((text, meta))
                        if len(unique) >= target:
                            break
                if len(unique) >= target:
                    break

            prompt_offset += n_prompts
            rounds += 1

        if len(unique) < target:
            print(
                f"          warning: generated {len(unique)} usable unique "
                f"{label} examples; requested {target}."
            )

        for text, meta in unique[:target]:
            records.append({
                "text": text,
                "label": label,
                "domain": meta["domain"],
                "persona": meta["persona"],
                "prompt_index": meta["prompt_index"],
            })

    random.shuffle(records)
    return records


# =============================================================================
# STAGE C1: SEQUENCE-LIKELIHOOD / LABEL-CONFIDENCE CHECK
# =============================================================================
def label_confidence_scores(llm, tokenizer, records):
    """Estimate the model probability assigned to each sample's claimed label."""
    sampling = SamplingParams(temperature=0.0, max_tokens=2, logprobs=20)

    prompts = []
    for rec in records:
        user_msg = (
            "Classify the sentiment of the following review using exactly one "
            'word: "positive" or "negative". Answer with only that word.\n\n'
            f'Review: """{rec["text"]}"""\n\n'
            "Sentiment:"
        )
        prompts.append(render_chat(tokenizer, user_msg))

    outputs = llm.generate(prompts, sampling, use_tqdm=False)

    scores = []
    predicted = []
    raw_answers = []

    for rec, out in zip(records, outputs):
        text_answer = strip_think(out.outputs[0].text).strip().lower()
        raw_answers.append(text_answer)

        first = out.outputs[0].logprobs[0] if out.outputs[0].logprobs else {}

        p = {"positive": 0.0, "negative": 0.0}
        for tok_id, lp in first.items():
            tok = getattr(lp, "decoded_token", None)
            if tok is None:
                tok = tokenizer.decode([tok_id])

            t = tok.strip().lower()
            prob = math.exp(lp.logprob)

            if t.startswith("pos"):
                p["positive"] += prob
            elif t.startswith("neg"):
                p["negative"] += prob

        total = p["positive"] + p["negative"]
        if total > 0:
            conf = p[rec["label"]] / total
            pred = "positive" if p["positive"] >= p["negative"] else "negative"
        else:
            conf = 0.0
            pred = "positive" if text_answer.startswith("pos") else (
                "negative" if text_answer.startswith("neg") else "unknown"
            )

        scores.append(conf)
        predicted.append(pred)

    return scores, predicted, raw_answers


# =============================================================================
# STAGE C2: LLM-AS-JUDGE RATING
# =============================================================================
def judge_training_examples(llm, tokenizer, records):
    """Rate each review as GOOD, BORDERLINE, or BAD for sentiment training."""
    sampling = SamplingParams(temperature=0.0, max_tokens=6)

    prompts = []
    for rec in records:
        label_upper = rec["label"].upper()
        user_msg = (
            "A synthetic customer review is labeled POSITIVE or NEGATIVE.\n\n"
            f"Claimed label: {label_upper}\n"
            f'Review: """{rec["text"]}"""\n\n'
            "Rate this review as a training example for sentiment classification.\n\n"
            "GOOD:\n"
            "- the overall sentiment matches the claimed label;\n"
            "- the review is specific enough to be useful;\n"
            "- the text reads like a customer review.\n\n"
            "BORDERLINE:\n"
            "- the overall sentiment probably matches the claimed label, but the review is mixed, subtle, short, or only moderately specific;\n"
            "- the example may still be useful because real reviews often contain concessions or mixed details.\n\n"
            "BAD:\n"
            "- the overall sentiment does not match the claimed label;\n"
            "- the sentiment is neutral or too ambiguous;\n"
            "- the text is generic filler, malformed, repetitive, or not a review.\n\n"
            "Do not mark a review BAD merely because it contains a minor drawback, concession, "
            "or mixed detail, as long as the overall sentiment is clear.\n\n"
            "Answer only GOOD, BORDERLINE, or BAD."
        )
        prompts.append(render_chat(tokenizer, user_msg))

    outputs = llm.generate(prompts, sampling, use_tqdm=False)

    ratings = []
    raw_answers = []
    for out in outputs:
        ans = strip_think(out.outputs[0].text).strip().lower()
        raw_answers.append(ans)

        if ans.startswith("good"):
            ratings.append("GOOD")
        elif ans.startswith("border"):
            ratings.append("BORDERLINE")
        elif ans.startswith("bad"):
            ratings.append("BAD")
        else:
            ratings.append("BAD")

    return ratings, raw_answers


def score_grounded_pool(llm, tokenizer, records):
    """Add confidence and judge scores to grounded examples."""
    conf, pred, raw_cls = label_confidence_scores(llm, tokenizer, records)

    likelihood_pass = [
        (c >= LABEL_CONF_THRESHOLD) and (p == rec["label"])
        for rec, c, p in zip(records, conf, pred)
    ]

    candidates = []
    candidate_indices = []
    for i, (rec, ok) in enumerate(zip(records, likelihood_pass)):
        if ok:
            candidates.append(rec)
            candidate_indices.append(i)

    judge_ratings, judge_answers = judge_training_examples(llm, tokenizer, candidates) if candidates else ([], [])

    rating_by_index = {}
    answer_by_index = {}
    for idx, rating, answer in zip(candidate_indices, judge_ratings, judge_answers):
        rating_by_index[idx] = rating
        answer_by_index[idx] = answer

    scored = []
    for i, rec in enumerate(records):
        rating = rating_by_index.get(i, "BAD")
        item = {
            **rec,
            "label_confidence": round(float(conf[i]), 3),
            "llm_predicted_label": pred[i],
            "classification_answer": raw_cls[i],
            "passed_likelihood": bool(likelihood_pass[i]),
            "judge_rating": rating,
            "judge_answer": answer_by_index.get(i, None),
            "eligible": bool(likelihood_pass[i] and rating in {"GOOD", "BORDERLINE"}),
        }
        scored.append(item)

    conf_arr = np.array(conf) if conf else np.array([0.0])
    stats = {
        "input": len(records),
        "passed_likelihood": int(sum(likelihood_pass)),
        "judge_good": sum(1 for r in scored if r["passed_likelihood"] and r["judge_rating"] == "GOOD"),
        "judge_borderline": sum(1 for r in scored if r["passed_likelihood"] and r["judge_rating"] == "BORDERLINE"),
        "judge_bad": sum(1 for r in scored if r["judge_rating"] == "BAD"),
        "eligible": sum(1 for r in scored if r["eligible"]),
        "conf_min": round(float(conf_arr.min()), 3),
        "conf_mean": round(float(conf_arr.mean()), 3),
        "conf_max": round(float(conf_arr.max()), 3),
    }

    return scored, stats


# =============================================================================
# STAGE C3: DIVERSITY-AWARE SELECTION
# =============================================================================
def selection_priority(record):
    """Sort key for selecting examples after scoring."""
    rating_rank = {"GOOD": 0, "BORDERLINE": 1, "BAD": 2}
    return (
        rating_rank.get(record.get("judge_rating"), 2),
        -float(record.get("label_confidence", 0.0)),
        len(record.get("text", "")),
    )


def round_robin_by_domain(candidates, target_n, max_per_domain):
    """Select candidates across domains before filling remaining slots."""
    by_domain = defaultdict(list)
    for r in candidates:
        by_domain[r["domain"]].append(r)

    for domain in by_domain:
        by_domain[domain].sort(key=selection_priority)

    domains = sorted(by_domain.keys())
    selected = []
    selected_ids = set()
    per_domain_counts = Counter()

    progress = True
    while len(selected) < target_n and progress:
        progress = False
        for domain in domains:
            if len(selected) >= target_n:
                break
            if max_per_domain and per_domain_counts[domain] >= max_per_domain:
                continue

            pool = by_domain[domain]
            while pool and id(pool[0]) in selected_ids:
                pool.pop(0)

            if not pool:
                continue

            item = pool.pop(0)
            selected.append(item)
            selected_ids.add(id(item))
            per_domain_counts[domain] += 1
            progress = True

    return selected, selected_ids


def select_filtered_dataset(scored_records, target_per_class):
    """Build a balanced filtered dataset while preserving domain coverage."""
    selected_all = []
    selection_stats = {}

    if MAX_PER_DOMAIN_PER_LABEL > 0:
        max_per_domain = MAX_PER_DOMAIN_PER_LABEL
    else:
        max_per_domain = max(1, math.ceil(target_per_class / len(DOMAINS)))

    for label in LABELS:
        label_records = [r for r in scored_records if r["label"] == label]

        good = [r for r in label_records if r["passed_likelihood"] and r["judge_rating"] == "GOOD"]
        borderline = [r for r in label_records if r["passed_likelihood"] and r["judge_rating"] == "BORDERLINE"]

        selected, selected_ids = round_robin_by_domain(good, target_per_class, max_per_domain)

        if len(selected) < target_per_class:
            more, more_ids = round_robin_by_domain(
                borderline,
                target_per_class - len(selected),
                max_per_domain,
            )
            selected.extend(more)
            selected_ids.update(more_ids)

        if len(selected) < target_per_class:
            remaining = [
                r for r in good + borderline
                if id(r) not in selected_ids
            ]
            remaining.sort(key=selection_priority)
            needed = target_per_class - len(selected)
            selected.extend(remaining[:needed])

        if len(selected) < target_per_class:
            raise ValueError(
                f"Not enough eligible {label} examples after filtering: "
                f"have {len(selected)}, need {target_per_class}. "
                "Try increasing POOL_MULTIPLIER or lowering LABEL_CONF_THRESHOLD."
            )

        selected = selected[:target_per_class]
        for r in selected:
            selected_all.append({**r, "keep": True})

        domain_counts = Counter(r["domain"] for r in selected)
        rating_counts = Counter(r["judge_rating"] for r in selected)

        selection_stats[label] = {
            "selected": len(selected),
            "domain_counts": dict(domain_counts),
            "rating_counts": dict(rating_counts),
            "max_per_domain_first_pass": max_per_domain,
        }

    random.shuffle(selected_all)
    return selected_all, selection_stats


def filter_dataset(llm, tokenizer, records, target_per_class):
    """Score the grounded pool and select a filtered dataset."""
    scored, scoring_stats = score_grounded_pool(llm, tokenizer, records)
    selected, selection_stats = select_filtered_dataset(scored, target_per_class)

    selected_keys = {(r["label"], r["text"]) for r in selected}
    enriched = []
    for r in scored:
        keep = (r["label"], r["text"]) in selected_keys
        enriched.append({**r, "keep": keep})

    stats = {
        **scoring_stats,
        "selected": len(selected),
        "selected_per_class": target_per_class,
        "selection": selection_stats,
        "acceptance_rate": round(len(selected) / max(1, len(records)), 3),
    }

    return selected, enriched, stats


# =============================================================================
# STAGE D: DOWNSTREAM DATA
# =============================================================================
GOLD_TEST = [
    # Positive
    ("Battery easily lasts two full days and the sound is crisp.", "positive"),
    ("Honestly the best purchase I've made all year, no regrets.", "positive"),
    ("Setup took five minutes and it just worked out of the box.", "positive"),
    ("The staff remembered my name on the second visit, lovely place.", "positive"),
    ("Lightweight, comfortable, and they held up after a muddy trail run.", "positive"),
    ("It's quiet, efficient, and my floors have never been cleaner.", "positive"),
    ("Rich espresso every morning without the cafe price tag.", "positive"),
    ("Couldn't put it down, finished the whole thing in a weekend.", "positive"),
    ("Customer support picked up quickly and fixed my issue fast.", "positive"),
    ("Great value for the money, exceeded what I expected at this price.", "positive"),
    ("The screen is gorgeous and it never stutters during games.", "positive"),
    ("Pours a smooth shot and the milk frother is a useful bonus.", "positive"),
    ("My kids haven't stopped playing with it since it arrived.", "positive"),
    ("Clean interface, syncs across devices, genuinely saves me time.", "positive"),
    ("Tender, flavourful, and the portions were generous for the price.", "positive"),
    ("Solid build, feels premium in the hand, very happy overall.", "positive"),
    ("Noise cancelling is fantastic on the train commute.", "positive"),
    ("Arrived a day early and works exactly as advertised.", "positive"),
    ("A charming, well-paced story with characters I cared about.", "positive"),
    ("Cleans the whole apartment on one charge, set and forget.", "positive"),
    ("The hotel room was spotless and the bed was surprisingly comfortable.", "positive"),
    ("The shoes needed no break-in time and felt great on the first run.", "positive"),
    ("The app's reminders are simple but exactly what I needed.", "positive"),
    ("Even though it was expensive, the build quality makes it feel worth it.", "positive"),
    ("The restaurant was crowded, but the service stayed friendly and quick.", "positive"),
    ("The backpack looks small but fits my laptop, charger, and lunch easily.", "positive"),
    ("The toy survived a full week of rough play without a scratch.", "positive"),
    ("The series starts slowly, but the characters become genuinely compelling.", "positive"),
    ("The delivery arrived warm, complete, and earlier than promised.", "positive"),
    ("The coffee was strong, smooth, and not bitter at all.", "positive"),
    ("The camera struggles at night, but daytime photos look excellent.", "positive"),
    ("The blender is loud, yet it makes perfectly smooth soups.", "positive"),
    ("The room faced the street, but the windows blocked almost all the noise.", "positive"),
    ("The keyboard feels comfortable and the battery barely moves all day.", "positive"),
    ("The pasta was simple, fresh, and much better than expected.", "positive"),
    ("I expected a basic bag, but the stitching and zippers feel sturdy.", "positive"),
    ("The vacuum missed one corner, but the rest of the floor looked spotless.", "positive"),
    ("The book is short, thoughtful, and stayed with me after I finished it.", "positive"),
    ("The shoes are pricey, but the cushioning is excellent.", "positive"),
    ("The app looks plain, but it has made my mornings much more organized.", "positive"),
    ("The headphones clamp a little tightly, but the audio quality is superb.", "positive"),
    ("The hotel breakfast was limited, but everything available tasted fresh.", "positive"),
    ("The espresso machine takes practice, but the results are worth it.", "positive"),
    ("The restaurant portions were not huge, but every bite was delicious.", "positive"),
    ("The phone is not flashy, but it is reliable and fast.", "positive"),
    ("The toy is simple, but my child keeps choosing it over newer ones.", "positive"),
    ("The delivery packaging was plain, but the meal was hot and flavorful.", "positive"),
    ("The laptop runs warm, but performance has been excellent.", "positive"),
    ("The novel has a quiet plot, but the writing is beautiful.", "positive"),
    ("The shoes look better in person and feel supportive on long walks.", "positive"),

    # Negative
    ("Died after three weeks and the warranty process was a nightmare.", "negative"),
    ("Overpriced and underwhelming, I returned it the next day.", "negative"),
    ("The app crashes every time I try to save my work.", "negative"),
    ("Cold food, slow service, and they got the order wrong.", "negative"),
    ("Fell apart at the seams after a couple of short runs.", "negative"),
    ("Loud, clumsy, and it kept getting stuck under the couch.", "negative"),
    ("Leaks all over the counter and the coffee tastes burnt.", "negative"),
    ("Dull, predictable, and I gave up halfway through.", "negative"),
    ("Support never replied and the phone keeps rebooting itself.", "negative"),
    ("Cheap plastic that scratched within a day of normal use.", "negative"),
    ("The screen froze constantly and lost my progress twice.", "negative"),
    ("Watery shots and the machine is impossible to clean.", "negative"),
    ("Broke on the first day, my kid was in tears.", "negative"),
    ("Constant sync errors wiped out an afternoon of notes.", "negative"),
    ("Bland, greasy, and far too expensive for what you get.", "negative"),
    ("Feels flimsy and the buttons already stopped responding.", "negative"),
    ("The ear cushions started peeling within a month.", "negative"),
    ("Shipped late, arrived damaged, and the box was crushed.", "negative"),
    ("A tedious slog with a plot that goes nowhere.", "negative"),
    ("Misses entire rooms and the dustbin barely holds anything.", "negative"),
    ("The hotel lobby looked nice, but the room smelled damp.", "negative"),
    ("The shoes felt fine at first, then gave me blisters by mile two.", "negative"),
    ("The app has a clean design, but it loses data too often to trust.", "negative"),
    ("The restaurant staff were polite, but the food was cold and bland.", "negative"),
    ("The backpack has many pockets, but the zipper broke in a week.", "negative"),
    ("The toy looked cute, but the batteries died almost immediately.", "negative"),
    ("The first episode was promising, but the season became boring fast.", "negative"),
    ("The delivery was on time, but half the order was missing.", "negative"),
    ("The coffee smelled good but tasted sour and stale.", "negative"),
    ("The camera is decent outside, but indoor photos are blurry and noisy.", "negative"),
    ("The blender handles fruit but leaves chunks of ice every time.", "negative"),
    ("The room was large, but the air conditioner rattled all night.", "negative"),
    ("The keyboard is attractive, but several keys double-type randomly.", "negative"),
    ("The pasta looked good, but the sauce was watery and dull.", "negative"),
    ("The bag feels light, but the straps dig into my shoulders.", "negative"),
    ("The vacuum starts well, then gets trapped under the same chair every run.", "negative"),
    ("The book has a strong opening, but the ending feels rushed and empty.", "negative"),
    ("The shoes have nice colors, but the soles wore down in two weeks.", "negative"),
    ("The app promises focus, but notifications are buggy and distracting.", "negative"),
    ("The headphones sound acceptable, but the connection drops constantly.", "negative"),
    ("The hotel view was good, but the sheets were stained.", "negative"),
    ("The espresso machine looks premium, but it leaks after every use.", "negative"),
    ("The restaurant menu is creative, but the dishes arrived lukewarm.", "negative"),
    ("The phone feels sturdy, but the battery drains before dinner.", "negative"),
    ("The toy packaging was nice, but the main piece snapped right away.", "negative"),
    ("The meal tasted okay, but it arrived cold and soggy.", "negative"),
    ("The laptop is fast when it works, but it shuts down without warning.", "negative"),
    ("The novel has a clever premise, but the characters are flat.", "negative"),
    ("The shoes are comfortable indoors, but they slip badly on wet pavement.", "negative"),
    ("The backpack looks professional, but the fabric stains immediately.", "negative"),
]


class ReviewDataset(Dataset):
    """Tokenized review dataset for sequence classification."""
    def __init__(self, records, tokenizer, max_length):
        self.texts = [r["text"] for r in records]
        self.labels = [LABEL_TO_ID[r["label"]] for r in records]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def make_gold_records():
    """Convert held-out examples to record dictionaries."""
    return [{"text": text, "label": label} for text, label in GOLD_TEST]


def evaluate_model(model, data_loader, device):
    """Evaluate a sequence classifier."""
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in data_loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            pred = torch.argmax(outputs.logits, dim=-1)

            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(pred.detach().cpu().tolist())

    y_true_labels = [ID_TO_LABEL[i] for i in y_true]
    y_pred_labels = [ID_TO_LABEL[i] for i in y_pred]

    wrong = []
    gold_records = make_gold_records()
    for rec, true_id, pred_id in zip(gold_records, y_true, y_pred):
        if true_id != pred_id:
            wrong.append((rec["text"], ID_TO_LABEL[true_id], ID_TO_LABEL[pred_id]))

    return {
        "accuracy": round(float(accuracy_score(y_true_labels, y_pred_labels)), 4),
        "macro_f1": round(float(f1_score(y_true_labels, y_pred_labels, average="macro", labels=LABELS)), 4),
        "n_wrong": len(wrong),
        "wrong_examples": wrong[:5],
        "classification_report": classification_report(
            y_true_labels,
            y_pred_labels,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        ),
    }


def train_transformer_classifier(train_records, run_name):
    """Fine-tune a transformer classifier and evaluate it."""
    seed_value = SEED
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(DOWNSTREAM_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        DOWNSTREAM_MODEL,
        num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
    )
    model.to(device)

    train_dataset = ReviewDataset(train_records, tokenizer, DOWNSTREAM_MAX_LENGTH)
    gold_dataset = ReviewDataset(make_gold_records(), tokenizer, DOWNSTREAM_MAX_LENGTH)

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=DOWNSTREAM_BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )
    gold_loader = DataLoader(
        gold_dataset,
        batch_size=DOWNSTREAM_BATCH_SIZE,
        shuffle=False,
    )

    optimizer = AdamW(model.parameters(), lr=DOWNSTREAM_LR)

    total_steps = max(1, len(train_loader) * DOWNSTREAM_EPOCHS)
    warmup_steps = int(total_steps * DOWNSTREAM_WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"          fine-tuning {run_name}: {len(train_records)} examples on {device}")

    model.train()
    for epoch in range(DOWNSTREAM_EPOCHS):
        running_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch, labels=labels)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += float(loss.detach().cpu())
            n_batches += 1

        avg_loss = running_loss / max(1, n_batches)
        print(f"            epoch {epoch + 1}/{DOWNSTREAM_EPOCHS} loss={avg_loss:.4f}")

    metrics = evaluate_model(model, gold_loader, device)
    metrics["n_train"] = len(train_records)
    metrics["downstream_model"] = DOWNSTREAM_MODEL

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


# =============================================================================
# HELPERS
# =============================================================================
def label_balance(records):
    """Return class counts."""
    c = Counter(r["label"] for r in records)
    return {label: c.get(label, 0) for label in LABELS}


def judge_balance(records):
    """Return judge-rating counts."""
    return dict(Counter(r.get("judge_rating", "NA") for r in records))


def print_samples(records, n=2):
    """Print a few examples from each label."""
    by_label = {}
    for r in records:
        by_label.setdefault(r["label"], []).append(r)

    for label in LABELS:
        pool = by_label.get(label, [])
        for r in random.sample(pool, min(n, len(pool))):
            preview = r["text"][:120].replace("\n", " ")
            suffix = "..." if len(r["text"]) > 120 else ""
            extra = ""
            if "label_confidence" in r:
                extra = (
                    f" [conf={r['label_confidence']:.2f}, "
                    f"judge={r.get('judge_rating')}, domain={r.get('domain')}]"
                )
            print(f"    [{label:<8}] {preview}{suffix}{extra}")


def print_rejections(scored_records, n=3):
    """Print examples not selected into the filtered dataset."""
    not_selected = [r for r in scored_records if not r.get("keep")]
    if not not_selected:
        print("    (none)")
        return

    groups = {
        "failed likelihood": [
            r for r in not_selected
            if not r.get("passed_likelihood")
        ],
        "judge BAD": [
            r for r in not_selected
            if r.get("passed_likelihood") and r.get("judge_rating") == "BAD"
        ],
        "eligible but not selected": [
            r for r in not_selected
            if r.get("eligible") and not r.get("keep")
        ],
    }

    for reason, pool in groups.items():
        if not pool:
            continue
        print(f"    {reason}:")
        for r in random.sample(pool, min(n, len(pool))):
            preview = r["text"][:120].replace("\n", " ")
            suffix = "..." if len(r["text"]) > 120 else ""
            print(
                f"      [{r['label']:<8}] {preview}{suffix} "
                f"[conf={r['label_confidence']:.2f}, pred={r['llm_predicted_label']}, "
                f"judge={r['judge_rating']}, domain={r['domain']}]"
            )


def save_json(obj, name):
    """Save an object under OUTPUT_DIR."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return path


def take_balanced(records, n_per_class):
    """Sample the same number of examples from each class."""
    out = []
    for label in LABELS:
        pool = [r for r in records if r["label"] == label]
        if len(pool) < n_per_class:
            raise ValueError(
                f"Not enough {label} records: have {len(pool)}, need {n_per_class}. "
                "Lower N_PER_CLASS, lower LABEL_CONF_THRESHOLD, or increase POOL_MULTIPLIER."
            )
        out.extend(random.sample(pool, n_per_class))
    random.shuffle(out)
    return out


def print_results_table(results):
    """Print downstream metrics."""
    print("\n===== DOWNSTREAM RESULTS: TRANSFORMER SENTIMENT CLASSIFICATION =====")
    header = f"{'dataset':<26}{'n_train':>9}{'accuracy':>11}{'macro_f1':>11}{'n_wrong':>9}"
    print(header)
    print("-" * len(header))

    for name, m in results.items():
        if m is None:
            print(f"{name:<26}{'-- skipped --':>40}")
        else:
            print(
                f"{name:<26}"
                f"{m['n_train']:>9}"
                f"{m['accuracy']:>11.4f}"
                f"{m['macro_f1']:>11.4f}"
                f"{m['n_wrong']:>9}"
            )


def check_expected_order(results):
    """Check whether the three accuracies follow the expected ordering."""
    names = ["naive", "grounded unfiltered", "grounded filtered"]
    if any(results[n] is None for n in names):
        return False

    a = results["naive"]["accuracy"]
    b = results["grounded unfiltered"]["accuracy"]
    c = results["grounded filtered"]["accuracy"]

    return a <= b <= c and a < c


# =============================================================================
# MAIN
# =============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    llm = build_model()
    tokenizer = llm.get_tokenizer()

    # ---- Stage A: naive generation ----------------------------------------
    print("\n[Stage A] naive sentiment generation ...")
    naive = generate_dataset(llm, tokenizer, grounded=False, pool_multiplier=1)
    print(f"          {len(naive)} samples  balance={label_balance(naive)}")
    print("          Samples:")
    print_samples(naive)
    save_json(naive, "naive.json")

    # ---- Stage B: persona-grounded generation -----------------------------
    print("\n[Stage B] persona-grounded sentiment generation ...")
    grounded_pool = generate_dataset(
        llm,
        tokenizer,
        grounded=True,
        pool_multiplier=POOL_MULTIPLIER,
    )
    print(f"          {len(grounded_pool)} samples  balance={label_balance(grounded_pool)}")
    print("          Samples:")
    print_samples(grounded_pool)
    save_json(grounded_pool, "grounded_pool.json")

    # ---- Stage C: filtering and selection ---------------------------------
    print("\n[Stage C] scoring and selecting from grounded pool ...")
    grounded_filtered, grounded_scored, fstats = filter_dataset(
        llm,
        tokenizer,
        grounded_pool,
        target_per_class=N_PER_CLASS,
    )

    print(
        f"          {fstats['input']} in -> "
        f"{fstats['passed_likelihood']} pass likelihood -> "
        f"{fstats['eligible']} eligible -> "
        f"{fstats['selected']} selected "
        f"({100 * fstats['acceptance_rate']:.1f}% selected)"
    )
    print(
        "          judge ratings after likelihood: "
        f"GOOD={fstats['judge_good']}, "
        f"BORDERLINE={fstats['judge_borderline']}, "
        f"BAD={fstats['judge_bad']}"
    )
    print(
        "          confidence: "
        f"min={fstats['conf_min']:.3f}, "
        f"mean={fstats['conf_mean']:.3f}, "
        f"max={fstats['conf_max']:.3f}"
    )
    print(f"          filtered balance={label_balance(grounded_filtered)}")
    print(f"          filtered judge balance={judge_balance(grounded_filtered)}")
    print("          Filtered samples:")
    print_samples(grounded_filtered)
    print("          Not-selected samples:")
    print_rejections(grounded_scored)

    save_json(grounded_scored, "grounded_scored.json")
    save_json(grounded_filtered, "grounded_filtered.json")

    release_generation_model(llm)

    # ---- Stage D: downstream evaluation -----------------------------------
    print("\n[Stage D] downstream evaluation on fixed held-out reviews ...")

    train_per_class = min(
        N_PER_CLASS,
        min(label_balance(naive).values()),
        min(label_balance(grounded_pool).values()),
        min(label_balance(grounded_filtered).values()),
    )

    if train_per_class < N_PER_CLASS:
        print(
            f"          warning: using {train_per_class} samples per class because "
            "one dataset has fewer usable examples than requested."
        )

    naive_eq = take_balanced(naive, train_per_class)
    grounded_unfiltered_eq = take_balanced(grounded_pool, train_per_class)
    grounded_filtered_eq = take_balanced(grounded_filtered, train_per_class)

    print(f"          Each training set uses {train_per_class} samples per class ({2 * train_per_class} total).")

    results = {
        "naive": train_transformer_classifier(naive_eq, "naive"),
        "grounded unfiltered": train_transformer_classifier(grounded_unfiltered_eq, "grounded unfiltered"),
        "grounded filtered": train_transformer_classifier(grounded_filtered_eq, "grounded filtered"),
    }

    metrics = {
        "model": MODEL_NAME,
        "task": "binary sentiment classification",
        "downstream_model": DOWNSTREAM_MODEL,
        "downstream_epochs": DOWNSTREAM_EPOCHS,
        "downstream_batch_size": DOWNSTREAM_BATCH_SIZE,
        "downstream_learning_rate": DOWNSTREAM_LR,
        "n_per_class_requested": N_PER_CLASS,
        "n_per_class_used": train_per_class,
        "pool_multiplier": POOL_MULTIPLIER,
        "label_conf_threshold": LABEL_CONF_THRESHOLD,
        "max_per_domain_per_label": MAX_PER_DOMAIN_PER_LABEL,
        "gold_test_size": len(GOLD_TEST),
        "filtering_stats": fstats,
        "downstream": results,
        "expected_order_observed": check_expected_order(results),
    }
    save_json(metrics, "metrics.json")

    print_results_table(results)

    print("\nExpected ordering:")
    print("    naive <= grounded unfiltered <= grounded filtered")
    print(f"Observed in this run: {check_expected_order(results)}")

    print("\nMisclassified examples for grounded filtered:")
    m = results["grounded filtered"]
    if m and m["wrong_examples"]:
        for text, true_label, pred_label in m["wrong_examples"]:
            preview = text[:110].replace("\n", " ")
            suffix = "..." if len(text) > 110 else ""
            print(f"    true={true_label:<8} pred={pred_label:<8} | {preview}{suffix}")
    else:
        print("    (none)")

    print(
        "\nSaved: outputs/{naive,grounded_pool,grounded_scored,"
        "grounded_filtered,metrics}.json"
    )


if __name__ == "__main__":
    main()
