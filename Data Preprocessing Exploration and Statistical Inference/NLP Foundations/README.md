# 📝 Text Preprocessing and NLP Foundations

Before transformers, before word2vec, every NLP system starts with the same question: **how do we turn raw, messy text into something a machine can compute with?** This tutorial covers the classical foundations that still power (and precede) every modern pipeline.

```
Raw Text → [Cleaning / Regex] → [Tokenization] → [Normalization] → [Linguistic Analysis] → [Vectorization]
              remove noise         split into        stemming /        POS, NER,             BoW, TF-IDF,
              lowercase, strip     units             lemmatization     dependency parse      embeddings
```

## 1. Domain Evolution and Challenges

### 1.1 A Brief History of NLP

```
1950s ─────── 1980s ─────── 1990s ─────── 2013 ──────── 2017 ──────── 2018+
  │             │             │             │              │              │
Rule-based    Expert       Statistical    Neural        Transformers   Pretrained
systems       systems      NLP            embeddings    (Attention)    LLM era
(ELIZA,       (hand-       (n-grams,     (word2vec,    (BERT, GPT)   (GPT-3/4,
 SHRDLU)       written      HMMs, CRFs,    GloVe,                      LLaMA,
               grammars)    naive Bayes)   LSTMs)                      Claude)
```

| Era | Approach | Limitation |
|-----|----------|------------|
| **Rule-based** | Hand-crafted grammars and pattern matching | Brittle, doesn't scale, no coverage of real language variety |
| **Statistical** | Probabilities learned from corpora (n-grams, HMMs) | Sparse counts, limited context window, feature engineering |
| **Neural (static)** | Dense word vectors (word2vec, GloVe) + RNNs/LSTMs | One vector per word — no context sensitivity ("bank" is always the same) |
| **Transformers** | Contextual representations via self-attention | Compute-hungry; still built on top of tokenization + preprocessing! |

> Even the most modern LLM pipeline begins with the topics in this tutorial: cleaning, tokenization, and vectorization never went away — they just changed shape.

### 1.2 Why Text Is Hard

Text is *not* naturally numeric, and human language resists simple rules:

| Challenge | Example |
|-----------|---------|
| **Lexical ambiguity** | "bank" → river bank or financial bank? |
| **Syntactic ambiguity** | "I saw the man with the telescope" — who has the telescope? |
| **Morphology** | run, runs, ran, running — one concept, many surface forms |
| **Sparsity (Zipf's law)** | A few words are extremely frequent; most words are rare |
| **Context dependence** | "That's just great." — sincere or sarcastic? |
| **Noise** | Typos, slang, HTML tags, emojis, inconsistent casing |
| **World knowledge** | "The trophy didn't fit in the suitcase because *it* was too big" — what is "it"? |

Preprocessing exists to tame the first layers of this complexity so downstream models see cleaner, more regular input.

## 2. Regex Recap

Regular expressions are the workhorse of text cleaning — the first tool you reach for before any tokenizer runs.

### 2.1 Core Syntax

| Pattern | Matches | Example |
|---------|---------|---------|
| `.` | Any character (except newline) | `c.t` → cat, cut, c9t |
| `\d` / `\D` | Digit / non-digit | `\d{4}` → 2024 |
| `\w` / `\W` | Word char [a-zA-Z0-9_] / non-word | `\w+` → hello_42 |
| `\s` / `\S` | Whitespace / non-whitespace | `\s+` → spaces, tabs, newlines |
| `*` `+` `?` | 0+, 1+, 0-or-1 repetitions | `colou?r` → color, colour |
| `{m,n}` | Between m and n repetitions | `\d{2,4}` → 42, 2024 |
| `[...]` / `[^...]` | Character class / negated class | `[aeiou]`, `[^0-9]` |
| `^` / `$` | Start / end of string (or line with `re.M`) | `^Subject:` |
| `(...)` | Capturing group | `(\d+)-(\d+)` |
| `(?:...)` | Non-capturing group | `(?:Mr|Ms)\. \w+` |
| `(?P<name>...)` | Named group | `(?P<year>\d{4})` |
| `\b` | Word boundary | `\bcat\b` matches "cat" not "category" |
| `A\|B` | Alternation | `cat\|dog` |

### 2.2 Common NLP Cleaning Patterns

| Task | Pattern (simplified) |
|------|----------------------|
| Email addresses | `[\w.+-]+@[\w-]+\.[\w.-]+` |
| URLs | `https?://\S+` |
| Dates (ISO) | `\d{4}-\d{2}-\d{2}` |
| HTML tags | `<[^>]+>` |
| Hashtags / mentions | `[#@]\w+` |
| Repeated whitespace | `\s+` → replace with single space |
| Non-alphabetic chars | `[^a-zA-Z\s]` → remove or replace |

### 2.3 Python `re` Essentials

| Function | Purpose |
|----------|---------|
| `re.search(p, s)` | First match anywhere in string (or `None`) |
| `re.match(p, s)` | Match only at the *beginning* of string |
| `re.findall(p, s)` | List of all non-overlapping matches |
| `re.finditer(p, s)` | Iterator of match objects (with positions) |
| `re.sub(p, repl, s)` | Replace matches — the cleaning workhorse |
| `re.split(p, s)` | Split string by pattern |
| `re.compile(p)` | Precompile for repeated use (faster in loops) |

Useful flags: `re.IGNORECASE` (`re.I`), `re.MULTILINE` (`re.M`), `re.DOTALL` (`re.S`), `re.VERBOSE` (`re.X`).

## 3. The NLP Pipeline: Tokenization, Stemming, Lemmatization

```
"The cats were running quickly!"
        │
        ▼  Tokenization
["The", "cats", "were", "running", "quickly", "!"]
        │
        ▼  Lowercasing + stop-word / punctuation removal
["cats", "running", "quickly"]
        │
        ▼  Stemming            OR        Lemmatization
["cat", "run", "quickli"]              ["cat", "run", "quickly"]
```

### 3.1 Tokenization

Splitting text into units (tokens). Sounds trivial — it isn't:

- `"don't"` → `don't`? `do` + `n't`? `don` + `'` + `t`?
- `"state-of-the-art"` → one token or four?
- `"Dr. Smith paid $4.50."` — which periods end sentences?
- Chinese/Japanese have no spaces at all.

**Common strategies:**

| Tokenizer | Behavior |
|-----------|----------|
| Whitespace split | Fast, naive — punctuation sticks to words ("dog." ≠ "dog") |
| Punkt (NLTK) | Unsupervised sentence splitter — handles abbreviations |
| Treebank (NLTK) | Penn Treebank conventions — splits contractions (`do` + `n't`) |
| spaCy | Rule-based + exceptions list, fast and production-grade |
| Subword (BPE etc.) | Used by transformers |

### 3.2 Stemming

**Stemming** chops word endings using crude heuristic rules. Fast, but the output is often not a real word.

| Word | Porter stem |
|------|-------------|
| running | run |
| studies | studi |
| easily | easili |
| university | univers |
| universal | univers ← same stem as "university"! |

**Failure modes:**
- **Over-stemming:** unrelated words collapse together (`university` / `universal` → `univers`)
- **Under-stemming:** related words stay apart (`alumnus` / `alumni`)

**Popular stemmers:** Porter (classic, gentle), Snowball (Porter2, improved + multilingual), Lancaster (very aggressive).

### 3.3 Lemmatization

**Lemmatization** maps a word to its dictionary form (*lemma*) using vocabulary + morphological analysis. Slower but linguistically correct — and it needs to know the part of speech:

```
"better"  + POS=adjective → "good"
"was"     + POS=verb      → "be"
"running" + POS=verb      → "run"
"running" + POS=noun      → "running"   (as in "running is fun")
```

| | Stemming | Lemmatization |
|---|----------|---------------|
| **Method** | Rule-based suffix chopping | Dictionary + morphology |
| **Output** | May not be a real word | Always a valid lemma |
| **Speed** | Very fast | Slower (lookup + POS needed) |
| **Use when** | Search/IR, quick baselines | Anything user-facing or linguistically sensitive |

### 3.4 Stop Words

Extremely frequent function words (*the, is, at, on, and…*) that often carry little topical content. Removing them shrinks the feature space for BoW/TF-IDF models — but beware: for sentiment ("not good") or phrase queries, stop words matter. Modern transformer models keep them.

## 4. POS Tagging, Named Entity Recognition, and Dependency Parsing

These three tasks add *linguistic structure* on top of tokens.

### 4.1 Part-of-Speech (POS) Tagging

Assign a grammatical category to each token.

```
"The   quick  brown  fox   jumps  over  the   lazy  dog"
 DET   ADJ    ADJ    NOUN  VERB   ADP   DET   ADJ   NOUN
```

Two common tagsets:

| Universal (coarse) | Penn Treebank (fine) | Example |
|--------------------|----------------------|---------|
| NOUN | NN, NNS, NNP, NNPS | dog, dogs, London |
| VERB | VB, VBD, VBG, VBN, VBP, VBZ | run, ran, running |
| ADJ | JJ, JJR, JJS | quick, quicker, quickest |
| ADV | RB, RBR, RBS | quickly |
| ADP | IN | over, in, of |
| DET | DT | the, a |
| PRON | PRP, PRP$ | she, her |

POS tags disambiguate ("book a flight" VERB vs "read a book" NOUN) and feed lemmatizers and parsers.

### 4.2 Named Entity Recognition (NER)

Locate and classify real-world entities in text:

```
"[Apple]ORG  is looking at buying a [U.K.]GPE startup for [$1 billion]MONEY in [2025]DATE"
```

| Entity type | Meaning | Examples |
|-------------|---------|----------|
| PERSON | People | Ada Lovelace |
| ORG | Companies, institutions | Google, MIT |
| GPE | Countries, cities, states | France, Tokyo |
| LOC | Non-GPE locations | the Alps, Pacific Ocean |
| DATE / TIME | Temporal expressions | July 2026, 3pm |
| MONEY / PERCENT | Monetary / percentage values | $4.5M, 30% |
| PRODUCT / EVENT | Objects, named events | iPhone, World Cup |

**The BIO scheme** marks entity boundaries at the token level — needed because entities span multiple tokens:

```
Token:   Barack   Obama    visited   New      York     City
BIO:     B-PER    I-PER    O         B-GPE    I-GPE    I-GPE

B- = Beginning of entity    I- = Inside entity    O = Outside
```

### 4.3 Dependency Parsing

Build a tree of grammatical relations: every word has exactly one **head**, and the relation is labeled.

```
"The quick brown fox jumps over the lazy dog."

jumps (ROOT)
├── fox (nsubj)          ← nominal subject
│   ├── The (det)
│   ├── quick (amod)     ← adjectival modifier
│   └── brown (amod)
├── over (prep)          ← prepositional modifier
│   └── dog (pobj)       ← object of preposition
│       ├── the (det)
│       └── lazy (amod)
└── . (punct)          ← punctuation
```

| Relation | Meaning | Example |
|----------|---------|---------|
| nsubj | Nominal subject | *fox* jumps |
| dobj / obj | Direct object | eats *pizza* |
| amod | Adjectival modifier | *lazy* dog |
| det | Determiner | *the* dog |
| prep / pobj | Preposition + its object | over → dog |
| advmod | Adverbial modifier | runs *quickly* |
| conj / cc | Conjunct / coordinator | cats *and* dogs |

Dependency trees power relation extraction ("who did what to whom"), question answering, and grammar checking.

## 5. Text Representation

Models need numbers, not strings. Three generations of turning text into vectors:

### 5.1 Bag-of-Words (BoW)

Represent each document as a **vector of word counts**, ignoring order.

```
d1: "the cat sat on the mat"
d2: "the dog sat on the log"

Vocabulary: [cat, dog, log, mat, on, sat, the]

        cat  dog  log  mat  on  sat  the
d1  →  [ 1,   0,   0,   1,   1,   1,   2 ]
d2  →  [ 0,   1,   1,   0,   1,   1,   2 ]
```

**Pros:** simple, fast, surprisingly strong baseline with linear models.
**Cons:** no word order ("dog bites man" = "man bites dog"), huge sparse vectors, all words weighted equally — "the" counts as much as "cat".

**N-grams** partially recover order by counting token *sequences* (`"new york"` as one feature), at the cost of an exploding vocabulary.

### 5.2 TF-IDF

Fix BoW's equal weighting: a term matters if it's **frequent in this document** but **rare across the corpus**.

```
tf-idf(t, d) = tf(t, d) × idf(t)

tf(t, d)  = count of t in d / total terms in d
idf(t)    = log( N / df(t) )        N = number of docs, df = docs containing t
```

Worked example — corpus of 3 documents, scoring terms in d1 = "the cat sat on the mat" (6 terms):

| Term | tf(t, d1) | df(t) | idf = log(3/df) | tf-idf |
|------|-----------|-------|------------------|--------|
| the | 2/6 = 0.333 | 3 | log(1.0) = 0.000 | **0.000** |
| cat | 1/6 = 0.167 | 1 | log(3.0) = 1.099 | **0.183** |
| sat | 1/6 = 0.167 | 2 | log(1.5) = 0.405 | **0.068** |

"the" appears everywhere → zero weight. "cat" is distinctive → highest weight. Exactly the behavior we wanted.

> **Note:** scikit-learn uses a *smoothed* variant, `idf = ln((1+N)/(1+df)) + 1`, plus L2 normalization of each row — so its numbers differ slightly from the textbook formula. Lab 4 verifies this by hand.

### 5.3 Word Embeddings Overview

BoW/TF-IDF vectors are sparse, high-dimensional, and treat "cat" and "feline" as totally unrelated. **Embeddings** learn dense vectors (~100–300 dims) where similar words are *close*:

> **Distributional hypothesis:** "You shall know a word by the company it keeps." — J.R. Firth, 1957

**word2vec (2013)** — a shallow network trained on a fake task, whose weights become the embeddings:

```
CBOW: predict center from context        Skip-gram: predict context from center

[the] [cat] [___] [on] [mat]             [___] [___] [sat] [___] [___]
   \     \    ▲    /    /                              │
    ────── "sat" ──────                     "the","cat","on","mat"
```

**Famous property — vector arithmetic captures analogies:**

```
vec(king) − vec(man) + vec(woman) ≈ vec(queen)
vec(paris) − vec(france) + vec(germany) ≈ vec(berlin)
```

| Model | Idea | Strength |
|-------|------|----------|
| **word2vec** (Google, 2013) | Predictive: CBOW / skip-gram | Fast, quality analogies |
| **GloVe** (Stanford, 2014) | Count-based: factorize global co-occurrence matrix | Global statistics |
| **fastText** (Facebook, 2016) | word2vec + character n-grams | Handles typos and unseen words |

**The limitation that led to transformers:** these are *static* embeddings — one vector per word, so "bank" (river) and "bank" (money) collide. Contextual embeddings (BERT, GPT) solve this — see the Transformer Foundations tutorial.

## Tutorial

### Installation


```bash
# Create a virtual environment
python -m venv nlp-course
nlp-course\Scripts\activate        # Linux/macOS: source nlp-course/bin/activate

# Install dependencies
pip install nltk spacy scikit-learn gensim numpy pandas matplotlib

# Download the spaCy English model
python -m spacy download en_core_web_sm
```

NLTK data (tokenizers, WordNet, stop words) is downloaded automatically on first run.

### Quick Start

```bash
# Run everything (inside the virtual environment)
python nlp_foundations.py

# Or run individual labs
python nlp_foundations.py --lab 1   # Regex text cleaning
python nlp_foundations.py --lab 2   # Tokenization, stemming, lemmatization
python nlp_foundations.py --lab 3   # POS tagging, NER, dependency parsing
python nlp_foundations.py --lab 4   # BoW, TF-IDF, word2vec embeddings
python nlp_foundations.py --lab 5   # Full preprocessing pipeline on 20 Newsgroups
```

> **Note:** Labs 4 and 5 download the 20 Newsgroups corpus (~14 MB) on first run, and Lab 5 runs spaCy over a few hundred documents — expect a couple of minutes on CPU.
