import argparse
import re
import sys
import textwrap

# Windows consoles default to cp1252, which cannot print the box-drawing
# characters used below.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Check that packages are importable

def require(packages: list[str]):
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


def ensure_nltk_data(resources: dict):
    """Download NLTK data packages if not already present.
    resources maps download-name -> path used by nltk.data.find()."""
    import nltk
    for name, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"  Downloading NLTK resource: {name} …")
            nltk.download(name, quiet=True)


def load_spacy_model(name: str = "en_core_web_sm"):
    import spacy
    try:
        return spacy.load(name)
    except OSError:
        print(f"\n  spaCy model '{name}' not found.")
        print(f"     Install with:  python -m spacy download {name}\n")
        sys.exit(1)


# ════════════════════════════════════════════════════════════════════════════
#  SHARED UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def section(title: str):
    width = 70
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def subsection(title: str):
    print(f"\n  ── {title} {'─' * max(1, 60 - len(title))}")


# Simple ASCII table printer.
def show_table(headers: list, rows: list, col_width: int = 20):
    fmt = "  " + "".join(f"{{:<{col_width}}}" for _ in headers)
    print(fmt.format(*headers))
    print("  " + "-" * (col_width * len(headers)))
    for row in rows:
        print(fmt.format(*[str(c)[:col_width - 1] for c in row]))


# ════════════════════════════════════════════════════════════════════════════
#  LAB 1 — REGEX FOR TEXT CLEANING
# ════════════════════════════════════════════════════════════════════════════

RAW_SAMPLE = """
From: jane.doe42@research-lab.org (Jane Doe)
Subject: Re: Preprocessing question   <URGENT>
Date: 2026-07-10

Hi all,

Check out https://nlp-course.example.com/lesson-3 and http://arxiv.org/abs/1706.03762
for details!!! My colleague (email: bob_smith@uni.edu) said the deadline
moved from 2026-06-30 to 2026-07-15.

Call me at +1-555-867-5309 or 555.123.4567.

<b>IMPORTANT:</b> the corpus has 1,204,567 documents &amp; ~98.5% are English.
#NLP @students
"""


def lab1_regex():
    section("1 — REGEX FOR TEXT CLEANING")
    print(textwrap.dedent("""
      Raw text from the wild contains emails, URLs, markup, phone numbers,
      and inconsistent formatting. Regex is the first tool in the pipeline:
      extract what you need, strip what you don't.
    """))

    print("  Raw sample text:")
    print(textwrap.indent(RAW_SAMPLE, "  │ "))

    # ── 1.1  Extraction with findall ─────────────────────────────────────────
    subsection("1.1  Extracting structured items with re.findall")

    patterns = {
        "Emails":     r"[\w.+-]+@[\w-]+\.[\w.-]+",
        "URLs":       r"https?://\S+",
        "ISO dates":  r"\d{4}-\d{2}-\d{2}",
        "Phones":     r"\+?\d[\d\-.]{7,}\d",
        "Hashtags":   r"[#@]\w+",
        "HTML tags":  r"<[^>]+>",
    }

    for name, pat in patterns.items():
        matches = re.findall(pat, RAW_SAMPLE)
        print(f"  {name:<12} {pat:<28} → {matches}")

    print(textwrap.dedent("""
      ⚠ Look closely: the phone pattern also matched the dates and the arXiv
      ID, and '@research'/'@uni' inside the emails matched the mention
      pattern. Regex precision is hard — replacement ORDER fixes this below.
    """))

    # ── 1.2  Named groups: parsing dates into components ─────────────────────
    subsection("1.2  Named groups — parsing dates")

    date_pat = re.compile(r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})")
    print("  Pattern: (?P<year>\\d{4})-(?P<month>\\d{2})-(?P<day>\\d{2})\n")
    for m in date_pat.finditer(RAW_SAMPLE):
        print(f"    Match {m.group(0)!r}  →  year={m['year']}  month={m['month']}  day={m['day']}")

    # ── 1.3  A cleaning pipeline with re.sub ─────────────────────────────────
    subsection("1.3  Cleaning pipeline with re.sub")

    steps = [
        ("Remove email headers",   r"(?m)^(From|Subject|Date):.*$", " "),
        ("Remove HTML tags",       r"<[^>]+>",                      " "),
        ("Remove HTML entities",   r"&\w+;",                        " "),
        ("Replace URLs",           r"https?://\S+",                 " <URL> "),
        ("Replace emails",         r"[\w.+-]+@[\w-]+\.[\w.-]+",     " <EMAIL> "),
        ("Replace ISO dates",      r"\d{4}-\d{2}-\d{2}",            " <DATE> "),
        ("Replace phone numbers",  r"\+?\d[\d\-.]{7,}\d",           " <PHONE> "),
        ("Drop hashtags/mentions", r"[#@]\w+",                      " "),
        ("Collapse punctuation",   r"([!?.])\1+",                   r"\1"),
        ("Collapse whitespace",    r"\s+",                          " "),
    ]

    text = RAW_SAMPLE
    for name, pat, repl in steps:
        before_len = len(text)
        text = re.sub(pat, repl, text)
        print(f"  {name:<26} {before_len:>4} → {len(text):>4} chars")

    text = text.strip().lower()
    print(f"\n  Cleaned text:\n{textwrap.indent(textwrap.fill(text, 64), '  │ ')}")

    # ── 1.4  Naive tokenization with re.split — and its limits ───────────────
    subsection("1.4  Naive regex tokenization (and why it isn't enough)")

    tricky = "Don't split state-of-the-art wrongly... it costs $4.50, Dr. Smith!"
    print(f"  Input: {tricky!r}\n")

    naive_ws   = tricky.split()
    naive_word = re.findall(r"\w+", tricky)
    smarter    = re.findall(r"\w+(?:[-']\w+)*|\$?\d+(?:\.\d+)?|[^\w\s]", tricky)

    print(f"  str.split()          → {naive_ws}")
    print(f"  re.findall(r'\\w+')   → {naive_word}")
    print(f"  smarter pattern      → {smarter}")
    print(textwrap.dedent("""
      Notice the trade-offs: whitespace splitting glues punctuation to words,
      \\w+ destroys "don't" and "$4.50". Real tokenizers (Lab 2, Lab 3) encode
      hundreds of such rules — don't reinvent them with regex.
    """))


# ════════════════════════════════════════════════════════════════════════════
#  LAB 2 — TOKENIZATION, STEMMING, LEMMATIZATION
# ════════════════════════════════════════════════════════════════════════════

def lab2_tokenize_stem_lemma():
    require(["nltk"])
    ensure_nltk_data({
        "punkt":                       "tokenizers/punkt",
        "punkt_tab":                   "tokenizers/punkt_tab",
        "wordnet":                     "corpora/wordnet",
        "omw-1.4":                     "corpora/omw-1.4",
        "stopwords":                   "corpora/stopwords",
        "averaged_perceptron_tagger":  "taggers/averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
    })

    from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize
    from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer, WordNetLemmatizer
    from nltk.corpus import stopwords, wordnet
    from nltk import pos_tag

    section("2 — TOKENIZATION, STEMMING, LEMMATIZATION")

    # ── 2.1  Sentence tokenization ───────────────────────────────────────────
    subsection("2.1  Sentence tokenization — the abbreviation problem")

    para = ("Dr. Smith paid $4.50 for the U.S.A. edition. He liked it! "
            "Prof. Jones, however, was skeptical... She bought vol. 2 instead.")
    print(f"  Input: {para!r}\n")

    naive = re.split(r"(?<=[.!?])\s+", para)
    print("  Naive split on [.!?]:")
    for s in naive:
        print(f"    • {s!r}")

    print("\n  NLTK Punkt sent_tokenize:")
    for s in sent_tokenize(para):
        print(f"    • {s!r}")
    print("\n  Punkt learned that 'Dr.', 'U.S.A.' and 'Prof.' do not end sentences —")
    print("  but it still splits after 'vol.' Sentence segmentation is never 100%;")
    print("  rare abbreviations before digits remain a classic failure case.")

    # ── 2.2  Word tokenizers compared ────────────────────────────────────────
    subsection("2.2  Word tokenizers on the same sentence")

    tricky = "Don't judge state-of-the-art models by $4.50 benchmarks, Dr. Smith."
    print(f"  Input: {tricky!r}\n")

    print(f"  str.split()         → {tricky.split()}")
    print(f"  word_tokenize       → {word_tokenize(tricky)}")
    print(f"  wordpunct_tokenize  → {wordpunct_tokenize(tricky)}")
    print("\n  word_tokenize (Treebank rules) splits \"Don't\" into 'Do' + \"n't\" —")
    print("  a deliberate convention so \"n't\" can be normalized to 'not'.")

    # ── 2.3  Stemmers compared ───────────────────────────────────────────────
    subsection("2.3  Porter vs Snowball vs Lancaster stemmers")

    porter, snowball, lancaster = PorterStemmer(), SnowballStemmer("english"), LancasterStemmer()
    words = ["running", "ran", "runs", "easily", "fairly", "studies", "studying",
             "university", "universal", "caresses", "ponies", "meeting", "generously"]

    show_table(
        ["Word", "Porter", "Snowball", "Lancaster"],
        [[w, porter.stem(w), snowball.stem(w), lancaster.stem(w)] for w in words],
        col_width=16,
    )
    print("\n  Lancaster is the most aggressive (shortest stems, most collisions).")
    print("  Over-stemming: 'university' and 'universal' both → 'univers'.")

    # ── 2.4  Lemmatization needs POS ─────────────────────────────────────────
    subsection("2.4  Lemmatization — WordNet + POS tags")

    lemmatizer = WordNetLemmatizer()

    print("  Without POS, the lemmatizer assumes NOUN and misses verbs:\n")
    for w in ["running", "was", "better", "meeting"]:
        print(f"    {w:<10} noun→ {lemmatizer.lemmatize(w, 'n'):<10}"
              f" verb→ {lemmatizer.lemmatize(w, 'v'):<10}"
              f" adj→ {lemmatizer.lemmatize(w, 'a')}")

    def to_wordnet_pos(treebank_tag: str):
        """Map Penn Treebank tags to WordNet POS categories."""
        if treebank_tag.startswith("J"):
            return wordnet.ADJ
        if treebank_tag.startswith("V"):
            return wordnet.VERB
        if treebank_tag.startswith("R"):
            return wordnet.ADV
        return wordnet.NOUN

    sent = "The striped bats were hanging on their feet and ate better meals"
    tokens = word_tokenize(sent)
    tagged = pos_tag(tokens)

    print(f"\n  Full pipeline on: {sent!r}\n")
    show_table(
        ["Token", "PennPOS", "Porter stem", "Lemma (POS-aware)"],
        [[tok, tag, porter.stem(tok), lemmatizer.lemmatize(tok.lower(), to_wordnet_pos(tag))]
         for tok, tag in tagged],
        col_width=17,
    )
    print("\n  Lemmatization wins: were→be, ate→eat, better→good, feet→foot.")

    # ── 2.5  Stop words ──────────────────────────────────────────────────────
    subsection("2.5  Stop-word removal")

    sw = set(stopwords.words("english"))
    print(f"  NLTK English stop-word list: {len(sw)} words")
    print(f"  Sample: {sorted(sw)[:12]} …\n")

    sent = "This movie was not good and I would not recommend it to anyone"
    kept = [w for w in word_tokenize(sent.lower()) if w not in sw]
    print(f"  Input:    {sent!r}")
    print(f"  Filtered: {kept}")
    print("\n  ⚠ 'not' is a stop word — sentiment flipped from negative to 'good movie'!")
    print("  Always inspect the stop-word list against your task.")


# ════════════════════════════════════════════════════════════════════════════
#  LAB 3 — POS TAGGING, NER, DEPENDENCY PARSING (spaCy)
# ════════════════════════════════════════════════════════════════════════════

def lab3_linguistic_analysis():
    require(["spacy"])
    nlp = load_spacy_model()

    section("3 — POS TAGGING, NER, DEPENDENCY PARSING")
    print(f"\n  spaCy pipeline components: {nlp.pipe_names}")

    # ── 3.1  POS tagging ─────────────────────────────────────────────────────
    subsection("3.1  Part-of-speech tagging")

    doc = nlp("The quick brown fox jumps over the lazy dog.")
    show_table(
        ["Token", "Universal", "PennTag", "Lemma", "Explanation"],
        [[t.text, t.pos_, t.tag_, t.lemma_, __import__("spacy").explain(t.tag_)] for t in doc],
        col_width=15,
    )

    # POS disambiguation
    print("\n  POS resolves ambiguity — 'book' in two contexts:")
    for s in ["I want to book a flight.", "I read a good book."]:
        d = nlp(s)
        tok = next(t for t in d if t.text == "book")
        print(f"    {s:<28} → book/{tok.pos_}")

    # ── 3.2  Named entity recognition ────────────────────────────────────────
    subsection("3.2  Named entity recognition")

    text = ("Apple is looking at buying a U.K. startup for $1 billion, "
            "Tim Cook told Reuters in San Francisco on Thursday.")
    doc = nlp(text)
    print(f"  Input: {text!r}\n")

    show_table(
        ["Entity", "Label", "Explanation"],
        [[ent.text, ent.label_, __import__("spacy").explain(ent.label_)] for ent in doc.ents],
        col_width=22,
    )

    print("\n  Token-level BIO tags (B=begin, I=inside, O=outside):\n")
    row_tok = [t.text for t in doc[:10]]
    row_bio = [f"{t.ent_iob_}-{t.ent_type_}" if t.ent_type_ else "O" for t in doc[:10]]
    print("    " + "  ".join(f"{w:<9}" for w in row_tok))
    print("    " + "  ".join(f"{b:<9}" for b in row_bio))

    # ── 3.3  Dependency parsing ──────────────────────────────────────────────
    subsection("3.3  Dependency parsing")

    doc = nlp("The quick brown fox jumps over the lazy dog.")
    show_table(
        ["Token", "DepRel", "Head", "Explanation"],
        [[t.text, t.dep_, t.head.text, __import__("spacy").explain(t.dep_)] for t in doc],
        col_width=16,
    )

    def print_tree(token, indent="  ", last=True):
        branch = "└── " if last else "├── "
        label = "ROOT" if token.dep_ == "ROOT" else token.dep_
        print(f"{indent}{branch}{token.text} ({label})")
        children = list(token.children)
        for i, child in enumerate(children):
            ext = "    " if last else "│   "
            print_tree(child, indent + ext, i == len(children) - 1)

    print("\n  Dependency tree:\n")
    root = next(t for t in doc if t.dep_ == "ROOT")
    print_tree(root)

    # ── 3.4  Noun chunks — cheap phrase extraction ───────────────────────────
    subsection("3.4  Noun chunks (base noun phrases)")

    text = "Autonomous cars from large manufacturers shift insurance liability toward software vendors."
    doc = nlp(text)
    print(f"  Input: {text!r}\n")
    show_table(
        ["Chunk", "Root", "Root dep", "Root head"],
        [[c.text, c.root.text, c.root.dep_, c.root.head.text] for c in doc.noun_chunks],
        col_width=22,
    )
    print("\n  Noun chunks + dependency relations = lightweight relation extraction:")
    print("  (subject chunk) ── verb ── (object chunk)")


# ════════════════════════════════════════════════════════════════════════════
#  LAB 4 — TEXT REPRESENTATION: BoW, TF-IDF, WORD EMBEDDINGS
# ════════════════════════════════════════════════════════════════════════════

TOY_CORPUS = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the dog chased the cat",
    "cats and dogs are popular pets",
]


def lab4_representation():
    require(["sklearn", "numpy", "gensim"])

    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    section("4 — TEXT REPRESENTATION: BoW, TF-IDF, EMBEDDINGS")

    print("\n  Toy corpus:")
    for i, d in enumerate(TOY_CORPUS):
        print(f"    d{i}: {d!r}")

    # ── 4.1  Bag-of-Words ────────────────────────────────────────────────────
    subsection("4.1  Bag-of-Words with CountVectorizer")

    cv = CountVectorizer()
    bow = cv.fit_transform(TOY_CORPUS)
    vocab = cv.get_feature_names_out()

    print(f"\n  Vocabulary ({len(vocab)} terms): {list(vocab)}\n")
    show_table(
        ["doc"] + list(vocab),
        [[f"d{i}"] + list(row) for i, row in enumerate(bow.toarray())],
        col_width=9,
    )
    print("\n  Note: 'cat' and 'cats' are different columns — no lemmatization")
    print("  happened. Vectorizers count strings, not concepts (fixed in Lab 5).")

    # ── 4.2  N-grams ─────────────────────────────────────────────────────────
    subsection("4.2  Adding bigrams — ngram_range=(1, 2)")

    cv2 = CountVectorizer(ngram_range=(1, 2))
    cv2.fit(TOY_CORPUS)
    bigrams = [t for t in cv2.get_feature_names_out() if " " in t]
    print(f"\n  Unigram vocab: {len(vocab)} terms → with bigrams: {len(cv2.get_feature_names_out())} terms")
    print(f"  Sample bigram features: {bigrams[:8]}")
    print("\n  Word order partially recovered ('dog chased' ≠ 'chased dog'),")
    print("  but the feature space grows combinatorially.")

    # ── 4.3  TF-IDF ──────────────────────────────────────────────────────────
    subsection("4.3  TF-IDF with TfidfVectorizer")

    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(TOY_CORPUS)
    vocab_t = tfidf.get_feature_names_out()

    show_table(
        ["doc"] + list(vocab_t),
        [[f"d{i}"] + [f"{v:.2f}" for v in row] for i, row in enumerate(X.toarray())],
        col_width=9,
    )

    the_idx, chased_idx = list(vocab_t).index("the"), list(vocab_t).index("chased")
    print(f"\n  'the'    appears in 3/4 docs → low weight   (d0: {X[0, the_idx]:.3f})")
    print(f"  'chased' appears in 1/4 docs → high weight  (d2: {X[2, chased_idx]:.3f})")

    # Manual verification of sklearn's smoothed formula
    print("\n  Verifying sklearn's formula by hand for 'chased' in d2:")
    N, df = 4, 1
    idf = np.log((1 + N) / (1 + df)) + 1
    print(f"    idf = ln((1+4)/(1+1)) + 1 = {idf:.4f}")
    print(f"    raw tf-idf = count(1) × idf = {idf:.4f}, then the row is L2-normalized")
    row = np.zeros(len(vocab_t))
    for term in TOY_CORPUS[2].split():
        idx = list(vocab_t).index(term)
        df_t = sum(1 for d in TOY_CORPUS if term in d.split())
        row[idx] += np.log((1 + N) / (1 + df_t)) + 1
    row = row / np.linalg.norm(row)
    print(f"    hand-computed value: {row[chased_idx]:.4f}   sklearn: {X[2, chased_idx]:.4f}  "
          f"{'✓' if abs(row[chased_idx] - X[2, chased_idx]) < 1e-6 else '✗'}")

    # ── 4.4  Document similarity ─────────────────────────────────────────────
    subsection("4.4  Document similarity in TF-IDF space")

    sim = cosine_similarity(X)
    show_table(
        [""] + [f"d{i}" for i in range(len(TOY_CORPUS))],
        [[f"d{i}"] + [f"{v:.3f}" for v in row] for i, row in enumerate(sim)],
        col_width=8,
    )
    print("\n  d0 (cat/mat) and d1 (dog/log) share 'sat on the' → moderately similar.")
    print("  d3 uses 'cats'/'dogs' (plural) → near-zero similarity. Sparse vectors")
    print("  cannot see that cats≈cat. Embeddings fix this ↓")

    # ── 4.5  word2vec on real data ───────────────────────────────────────────
    subsection("4.5  Training word2vec on 20 Newsgroups (gensim)")

    from sklearn.datasets import fetch_20newsgroups
    from gensim.models import Word2Vec
    from gensim.utils import simple_preprocess

    print("\n  Fetching 20 Newsgroups (downloads ~14 MB on first run)…")
    news = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
    print(f"  {len(news.data)} documents across {len(news.target_names)} categories")

    print("  Tokenizing with gensim.simple_preprocess…")
    sentences = [simple_preprocess(doc) for doc in news.data]

    print("  Training Word2Vec (skip-gram, 100 dims, 5 epochs)…")
    w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=5,
                   sg=1, epochs=5, workers=4, seed=42)
    print(f"  ✓ Vocabulary: {len(w2v.wv):,} words, vector size: {w2v.wv.vector_size}")

    print("\n  Nearest neighbors (cosine similarity in embedding space):\n")
    for query in ["computer", "car", "space", "religion", "hockey"]:
        if query in w2v.wv:
            nbrs = [(w, round(s, 3)) for w, s in w2v.wv.most_similar(query, topn=4)]
            print(f"    {query:<10} → {nbrs}")

    print("\n  Vector arithmetic (small corpus, so results are approximate):\n")
    for pos, neg in [((["windows", "apple"]), ["microsoft"]),
                     ((["car", "pilot"]), ["driver"])]:
        try:
            result = w2v.wv.most_similar(positive=pos, negative=neg, topn=3)
            print(f"    {' + '.join(pos)} − {neg[0]:<10} ≈ {[(w, round(s, 3)) for w, s in result]}")
        except KeyError as e:
            print(f"    Skipped (word not in vocab: {e})")

    print(textwrap.dedent("""
      Trained on ~11k noisy posts, these vectors already cluster topics.
      word2vec trained on billions of tokens (Google News) produces the
      famous king−man+woman≈queen result. Still: ONE vector per word —
      no context. That limitation is what transformers solved.
    """))


# ════════════════════════════════════════════════════════════════════════════
#  LAB 5 — FULL PREPROCESSING PIPELINE ON A TEXT CORPUS
# ════════════════════════════════════════════════════════════════════════════

def lab5_pipeline():
    require(["sklearn", "spacy", "numpy"])

    import numpy as np
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    nlp = load_spacy_model()

    section("5 — FULL NLP PREPROCESSING PIPELINE (20 NEWSGROUPS)")
    print(textwrap.dedent("""
      Everything from Labs 1–4 combined into one pipeline:

        raw corpus → regex clean → spaCy tokenize + lemmatize
                   → stop-word/POS filter → TF-IDF → features → classifier
    """))

    # ── 5.1  Load corpus ─────────────────────────────────────────────────────
    subsection("5.1  Load the corpus")

    categories = ["rec.sport.hockey", "sci.space", "comp.graphics", "talk.religion.misc"]
    news = fetch_20newsgroups(subset="train", categories=categories,
                              remove=("headers", "footers", "quotes"),
                              shuffle=True, random_state=42)

    MAX_DOCS = 800  # keep spaCy runtime reasonable on CPU
    docs   = news.data[:MAX_DOCS]
    labels = news.target[:MAX_DOCS]

    print(f"  Categories: {news.target_names}")
    print(f"  Using {len(docs)} documents (of {len(news.data)} available)")
    lengths = [len(d) for d in docs]
    print(f"  Document length: min={min(lengths)}  median={sorted(lengths)[len(lengths)//2]}  max={max(lengths)} chars")
    print(f"\n  Raw sample:\n{textwrap.indent(docs[0][:300].strip(), '  │ ')}")

    # ── 5.2  Stage 1: regex cleaning ─────────────────────────────────────────
    subsection("5.2  Stage 1 — regex cleaning (Lab 1)")

    def clean(text: str) -> str:
        text = re.sub(r"\S+@\S+", " ", text)          # emails
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # URLs
        text = re.sub(r"<[^>]+>", " ", text)          # markup
        text = re.sub(r"[^a-zA-Z\s]", " ", text)      # non-letters
        text = re.sub(r"\s+", " ", text)              # whitespace
        return text.strip().lower()

    cleaned = [clean(d) for d in docs]
    before = sum(len(d) for d in docs)
    after  = sum(len(d) for d in cleaned)
    print(f"  Total characters: {before:,} → {after:,}  ({100 * after / before:.1f}% kept)")
    print(f"\n  Cleaned sample:\n{textwrap.indent(textwrap.fill(cleaned[0][:280], 64), '  │ ')}")

    # ── 5.3  Stage 2: tokenize + lemmatize + filter ──────────────────────────
    subsection("5.3  Stage 2 — spaCy tokenize, lemmatize, filter (Labs 2–3)")

    KEEP_POS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
    print(f"  Keeping only content words: {sorted(KEEP_POS)}")
    print("  Dropping stop words and tokens shorter than 3 characters.")
    print(f"  Processing {len(cleaned)} docs with spaCy (parser/NER disabled for speed)…")

    processed = []
    for doc in nlp.pipe(cleaned, disable=["parser", "ner"], batch_size=64):
        lemmas = [t.lemma_ for t in doc
                  if t.pos_ in KEEP_POS and not t.is_stop and len(t.lemma_) >= 3]
        processed.append(" ".join(lemmas))

    n_tokens_before = sum(len(d.split()) for d in cleaned)
    n_tokens_after  = sum(len(d.split()) for d in processed)
    print(f"  ✓ Done. Tokens: {n_tokens_before:,} → {n_tokens_after:,} "
          f"({100 * n_tokens_after / n_tokens_before:.1f}% kept)")
    print(f"\n  Processed sample:\n{textwrap.indent(textwrap.fill(processed[0][:280], 64), '  │ ')}")

    # ── 5.4  Stage 3: TF-IDF vectorization ───────────────────────────────────
    subsection("5.4  Stage 3 — TF-IDF vectorization (Lab 4)")

    vectorizer = TfidfVectorizer(max_features=5000, min_df=3, max_df=0.9,
                                 ngram_range=(1, 2))
    X = vectorizer.fit_transform(processed)
    print(f"  Feature matrix: {X.shape[0]} docs × {X.shape[1]} features "
          f"(sparsity: {100 * (1 - X.nnz / (X.shape[0] * X.shape[1])):.1f}% zeros)")
    print("  min_df=3 drops hapax terms, max_df=0.9 drops corpus-wide boilerplate.")

    # ── 5.5  Inspect: top terms per category ─────────────────────────────────
    subsection("5.5  Top TF-IDF terms per category")

    feature_names = vectorizer.get_feature_names_out()
    labels_arr = np.asarray(labels)
    print()
    for cat_idx, cat_name in enumerate(news.target_names):
        mask = labels_arr == cat_idx
        mean_tfidf = np.asarray(X[mask].mean(axis=0)).ravel()
        top = mean_tfidf.argsort()[::-1][:8]
        print(f"  {cat_name:<22} → {[feature_names[i] for i in top]}")

    print("\n  The pipeline recovered each newsgroup's topic vocabulary —")
    print("  hockey teams, orbital terms, graphics formats, theology.")

    # ── 5.6  Sanity check: do the features work? ─────────────────────────────
    subsection("5.6  Sanity check — logistic regression on the features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels_arr, test_size=0.25, random_state=42, stratify=labels_arr)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"\n  Train: {X_train.shape[0]} docs   Test: {X_test.shape[0]} docs")
    print(f"  Accuracy: {acc:.1%}  (chance level: {1 / len(categories):.1%})\n")
    print(textwrap.indent(
        classification_report(y_test, preds, target_names=news.target_names), "  "))

    print(textwrap.dedent("""
      A plain linear model on carefully preprocessed TF-IDF features is a
      strong baseline — often within a few points of neural models on topic
      classification. Preprocessing quality IS model quality at this level.

      Next step: swap TF-IDF for contextual embeddings — see the
      Transformer Foundations tutorial.
    """))


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="NLP Foundations Tutorial")
    parser.add_argument(
        "--lab", type=int, choices=[1, 2, 3, 4, 5],
        help="Run a specific lab (1=Regex, 2=Tokenize/Stem/Lemma, "
             "3=POS/NER/Parsing, 4=BoW/TF-IDF/Embeddings, 5=Full pipeline)"
    )
    args = parser.parse_args()

    print("\n" + "█" * 70)
    print("  NLP FOUNDATIONS — TEXT PREPROCESSING  ")
    print("█" * 70)
    print("""
  Labs:
    1 → Regex for Text Cleaning        (extraction, substitution)
    2 → Tokenization / Stemming / Lemmatization   (NLTK)
    3 → POS, NER, Dependency Parsing   (spaCy)
    4 → BoW, TF-IDF, word2vec          (scikit-learn, gensim)
    5 → Full Preprocessing Pipeline    (20 Newsgroups, end-to-end)
    """)

    lab_map = {
        1: lab1_regex,
        2: lab2_tokenize_stem_lemma,
        3: lab3_linguistic_analysis,
        4: lab4_representation,
        5: lab5_pipeline,
    }

    if args.lab is not None:
        lab_map[args.lab]()
    else:
        for lab in lab_map.values():
            lab()


if __name__ == "__main__":
    main()
