# Code written by Gemini 4 Hydrogen on 11/20/2026

import re
import time
import pickle
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import plotly.express as px
import streamlit as st
from fastopic import FASTopic
from topmost.preprocess import Preprocess

DATA_PATH = Path("data/filtered_submissions.parquet")
EMBED_DIR = Path("embeddings")
MODEL_DIR = Path("models")
EMBED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(layout="wide", page_title="K-Drama Topic Workbench v3")
st.title("K-Drama Submissions: Topic Modeling Workbench (v3)")

# ==============================================================================
# PERSISTENT SESSION STATE DEFAULTS
# ==============================================================================
CURATED_STOPS = (
    "drama, dramas, kdrama, kdramas, watch, watching, watched, episode, episodes, "
    "show, shows, series, really, just, like, good, great, best, time, times, know, "
    "think, thought, looking, look, want, wanted, dont, didnt, ive, youre, thats, "
    "theres, havent, isnt, wasnt, ill, way, new, loved, seen, see, seeing, feel, "
    "feeling, felt, remember, people, person, help, need, lot, bit, little, things, "
    "thing, guys, thank, thanks, kind, ones, started, start, going, got, right, "
    "actually, maybe, probably, definitely, honestly, basically, literally, wiki, "
    "spoiler, spoilers, spoilertags, tags, tag, rules, policies, faq, faqs, roster, "
    "markdown, redacted, guidance, guide, glossary, index, definitions, navigating, "
    "homepage, wheretowatchkdramas, kdramarecommends, flair, sub, post, posts, "
    "thread, comments, discussion, discussions, discussing, content, recommendations, "
    "recommendation, recommend, recs, suggest, suggestions, similar, korean, korea"
)

defaults = {
    "active_tab": "1. Length Distribution",
    "min_doc_words": 10,
    "max_doc_words": 400,
    "vocab_size": 10000,
    "min_doc_count": 5,
    "strip_urls": True,
    "custom_stops": CURATED_STOPS,
    "num_topics": 35,
    "benchmark_history": [],
    "last_run_topics": None,
    "last_run_meta": None,
    "last_doc_topic_dist": None,
    "last_kept_texts": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ==============================================================================
# TEXT CLEANING & DATA LOADING
# ==============================================================================
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((?:https?://|www\.)[^)]+\)")
URL_RE = re.compile(r"(?:https?://|www\.)\S+")
HTML_ENT_RE = re.compile(r"&(?:amp|lt|gt|quot|#x200B);")


def clean_reddit_text(text: str, remove_urls: bool = True) -> str:
    text = HTML_ENT_RE.sub(" ", text)
    if remove_urls:
        text = MD_LINK_RE.sub(r"\1", text)
        text = URL_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


@st.cache_data
def load_submissions(path: Path) -> pd.DataFrame:
    df = pq.read_table(path).to_pandas()
    if "type" in df.columns:
        df = df[df["type"] == "submission"].copy()
    df["raw_text"] = (
        df["submission_title"].fillna("") + " " + df["body"].fillna("")
    ).str.strip()
    df["word_count"] = df["raw_text"].str.split().str.len()
    return df.reset_index(drop=True)


if not DATA_PATH.exists():
    st.error(f"Dataset not found at `{DATA_PATH}`. Place your Parquet file inside `data/`.")
    st.stop()

df = load_submissions(DATA_PATH)

# Sidebar Status
available_npy = sorted([f.name for f in EMBED_DIR.glob("*.npy")])
st.sidebar.header("Workspace Status")
st.sidebar.metric("Total Submissions", f"{len(df):,}")
st.sidebar.metric("Pre-Computed `.npy` Files", len(available_npy))
if available_npy:
    for fname in available_npy:
        size_mb = (EMBED_DIR / fname).stat().st_size / (1024 * 1024)
        st.sidebar.caption(f"• `{fname}` ({size_mb:.1f} MB)")
else:
    st.sidebar.warning("No `.npy` files found in `embeddings/`. Run `compute_embeddings.py` first.")


# ==============================================================================
# PURE-NUMPY NPMI COHERENCE (No gensim / NumPy 2.x conflicts)
# ==============================================================================
def compute_npmi_coherence(topics: list[list[str]], tokenized_docs: list[list[str]]) -> float:
    n_docs = len(tokenized_docs)
    if n_docs == 0 or not topics:
        return 0.0

    target_words = set(w for t in topics for w in t)
    word_doc_sets = {w: set() for w in target_words}
    for doc_idx, tokens in enumerate(tokenized_docs):
        for tok in set(tokens):
            if tok in word_doc_sets:
                word_doc_sets[tok].add(doc_idx)

    eps = 1e-12
    topic_scores = []
    for words in topics:
        pair_scores = []
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                docs_i = word_doc_sets.get(words[i], set())
                docs_j = word_doc_sets.get(words[j], set())
                p_i = len(docs_i) / n_docs
                p_j = len(docs_j) / n_docs
                p_ij = len(docs_i & docs_j) / n_docs
                if p_ij > 0 and p_i > 0 and p_j > 0:
                    pmi = np.log((p_ij + eps) / (p_i * p_j + eps))
                    npmi = pmi / (-np.log(p_ij + eps))
                    pair_scores.append(npmi)
                else:
                    pair_scores.append(-1.0)
        if pair_scores:
            topic_scores.append(float(np.mean(pair_scores)))
    return round(float(np.mean(topic_scores)), 4)


# ==============================================================================
# CACHED PREPROCESSING & ZIPF COMPUTATION
# ==============================================================================
@st.cache_data
def run_preprocessing(texts, min_words, max_words, max_vocab, min_docs, stop_str, remove_urls):
    extra_stops = set(w.strip().lower() for w in stop_str.split(",") if w.strip())

    kept_texts = []
    kept_indices = []
    dropped_short = 0
    for idx, t in enumerate(texts):
        cleaned = clean_reddit_text(t, remove_urls=remove_urls)
        words = cleaned.split()
        if len(words) < min_words:
            dropped_short += 1
        else:
            kept_texts.append(" ".join(words[:max_words]))
            kept_indices.append(idx)

    base_prep = Preprocess(
        vocab_size=max_vocab,
        min_doc_count=min_docs,
        stopwords="English",
    )
    base_rst = base_prep.preprocess(kept_texts)

    term_counts = Counter()
    doc_counts = Counter()
    for doc in base_rst["train_texts"]:
        toks = doc.split()
        term_counts.update(toks)
        doc_counts.update(set(toks))

    n_docs = max(len(base_rst["train_texts"]), 1)
    freq_rows = []
    for rank, (word, count) in enumerate(term_counts.most_common(), start=1):
        freq_rows.append(
            {
                "Rank": rank,
                "Word": word,
                "Total Count": count,
                "Doc Count": doc_counts[word],
                "Doc Share (%)": round(doc_counts[word] / n_docs * 100, 2),
                "In Custom Stopwords": word in extra_stops,
            }
        )
    freq_df = pd.DataFrame(freq_rows)

    filtered_for_model = [
        " ".join(
            [w for w in t.split() if w.lower().strip(".,!?\"'()[]{}:;") not in extra_stops]
        )
        for t in kept_texts
    ]
    final_prep = Preprocess(
        vocab_size=max_vocab,
        min_doc_count=min_docs,
        stopwords="English",
    )
    final_rst = final_prep.preprocess(filtered_for_model)

    raw_tokens = set(
        w.lower().strip(".,!?\"'()[]{}:;")
        for t in kept_texts
        for w in t.split()
        if w.isalpha()
    )
    kept_vocab = set(final_rst["vocab"])
    dropped_vocab = sorted(list(raw_tokens - kept_vocab))

    return kept_texts, kept_indices, dropped_short, final_rst, dropped_vocab, freq_df


# ==============================================================================
# PERSISTENT TOP NAVIGATION BAR
# ==============================================================================
TAB_OPTIONS = [
    "1. Length Distribution",
    "2. Preprocessing & Zipf Audit",
    "3. Topic Modeling & Inspection",
]
active_tab = st.radio(
    "Navigation",
    TAB_OPTIONS,
    key="active_tab",
    horizontal=True,
    label_visibility="collapsed",
)
st.divider()


# ==============================================================================
# TAB 1: SUBMISSION LENGTH DISTRIBUTION
# ==============================================================================
if active_tab == "1. Length Distribution":
    st.header("Submission Length Distribution (Title + Body)")

    pcts = np.percentile(df["word_count"], [10, 25, 50, 75, 90, 95, 99])
    p_cols = st.columns(7)
    for col, label, val in zip(
        p_cols,
        ["10th %", "25th %", "Median (50%)", "75th %", "90th %", "95th %", "99th %"],
        pcts,
    ):
        col.metric(label, f"{int(val)} w")

    bins = [0, 15, 40, 150, 400, 1_000_000]
    bucket_labels = [
        "Tiny (1–15w)",
        "Short (16–40w)",
        "Medium (41–150w)",
        "Long (151–400w)",
        "Very Long (>400w)",
    ]
    df["length_bucket"] = pd.cut(df["word_count"], bins=bins, labels=bucket_labels)

    c1, c2 = st.columns([2, 1])
    with c1:
        fig = px.histogram(
            df[df["word_count"] <= 600],
            x="word_count",
            nbins=60,
            title="Word Count Distribution (Posts ≤ 600 words)",
            labels={"word_count": "Words (Title + Body)"},
        )
        fig.add_vline(
            x=st.session_state.max_doc_words,
            line_dash="dash",
            line_color="red",
            annotation_text=f"{st.session_state.max_doc_words}w Cap",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        bucket_df = df["length_bucket"].value_counts().reindex(bucket_labels).reset_index()
        bucket_df.columns = ["Length Bucket", "Posts"]
        bucket_df["Share (%)"] = (bucket_df["Posts"] / len(df) * 100).round(1)
        st.dataframe(bucket_df, use_container_width=True, hide_index=True)

        inspect_bucket = st.selectbox("Sample 3 random posts from bucket:", bucket_labels, index=1)
        subset = df[df["length_bucket"] == inspect_bucket]["raw_text"]
        sample_posts = subset.sample(min(3, len(subset)), random_state=42)
        for i, txt in enumerate(sample_posts, 1):
            st.caption(
                f"**Sample {i} ({len(txt.split())}w):** {txt[:280]}{'...' if len(txt) > 280 else ''}"
            )


# ==============================================================================
# TAB 2: PREPROCESSING, URL STRIPPING & ZIPF WORD EXPLORER
# ==============================================================================
elif active_tab == "2. Preprocessing & Zipf Audit":
    st.header("Preprocessing, URL Cleaning & Zipf Word Explorer")
    st.info(
        "Keep **Min words = 10**, **Max words = 400**, and **Strip URLs = Checked** so row counts "
        "match your pre-computed `.npy` embeddings. You can freely edit **Max vocabulary size**, "
        "**Min document frequency**, and **Extra domain stopwords** without re-computing embeddings."
    )

    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 0.9])
    col1.slider("Min words per post", 1, 50, key="min_doc_words")
    col2.slider("Max words per post (cap)", 100, 1000, step=50, key="max_doc_words")
    col3.slider("Max vocabulary size", 2000, 30000, step=1000, key="vocab_size")
    col4.slider("Min document frequency", 2, 50, key="min_doc_count")
    col5.checkbox("Strip URLs & Markdown links", key="strip_urls")

    st.text_input(
        "Extra domain stopwords to drop (comma-separated)",
        key="custom_stops",
    )

    kept_texts, _, dropped_short, prep_rst, dropped_vocab, freq_df = run_preprocessing(
        df["raw_text"].tolist(),
        st.session_state.min_doc_words,
        st.session_state.max_doc_words,
        st.session_state.vocab_size,
        st.session_state.min_doc_count,
        st.session_state.custom_stops,
        st.session_state.strip_urls,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Kept Submissions", f"{len(kept_texts):,}", f"-{dropped_short:,} short")
    m2.metric("Final Vocab Size", f"{len(prep_rst['vocab']):,}")
    m3.metric("Dropped Unique Words", f"{len(dropped_vocab):,}")
    m4.metric(
        "Avg Tokens / Post",
        f"{np.mean([len(d.split()) for d in prep_rst['train_texts']]):.1f}",
    )

    st.subheader("Top Words After English Stopwords (Zipf Curve & Sortable Table)")
    z_col, t_col = st.columns([1.15, 1])
    with z_col:
        top_n_zipf = st.slider("Words shown on Zipf plot", 50, 2000, 300, step=50)
        zipf_sub = freq_df.head(top_n_zipf)
        fig_zipf = px.scatter(
            zipf_sub,
            x="Rank",
            y="Total Count",
            hover_data=["Word", "Doc Share (%)", "In Custom Stopwords"],
            color="In Custom Stopwords",
            color_discrete_map={True: "#EF553B", False: "#636EFA"},
            log_x=True,
            log_y=True,
            title=f"Zipf Rank-Frequency Plot (Top {top_n_zipf} Non-English-Stopword Terms)",
        )
        fig_zipf.update_traces(marker=dict(size=7, opacity=0.85))
        st.plotly_chart(fig_zipf, use_container_width=True)

    with t_col:
        hide_already_dropped = st.checkbox("Hide words already in custom stopwords", value=False)
        display_freq = freq_df[~freq_df["In Custom Stopwords"]] if hide_already_dropped else freq_df
        st.dataframe(
            display_freq.head(500),
            use_container_width=True,
            hide_index=True,
            height=430,
        )

    with st.expander("Inspect Before/After Cleaning on Individual Posts", expanded=False):
        sample_idx = st.number_input("Inspect document index", 0, len(kept_texts) - 1, 0)
        st.write("**Original Raw Post:**", df.loc[sample_idx, "raw_text"][:1500])
        st.write("**After URL Cleaning & Word Cap:**", kept_texts[sample_idx])
        st.write("**After Stopwords & Vocab Filter (FASTopic BoW Input):**", prep_rst["train_texts"][sample_idx])


# ==============================================================================
# TAB 3: TOPIC MODELING FROM PRE-COMPUTED EMBEDDINGS
# ==============================================================================
elif active_tab == "3. Topic Modeling & Inspection":
    st.header("Topic Modeling & Model Benchmark (Pre-Computed Embeddings)")

    kept_texts, kept_indices, dropped_short, prep_rst, _, _ = run_preprocessing(
        df["raw_text"].tolist(),
        st.session_state.min_doc_words,
        st.session_state.max_doc_words,
        st.session_state.vocab_size,
        st.session_state.min_doc_count,
        st.session_state.custom_stops,
        st.session_state.strip_urls,
    )

    c_left, c_mid, c_right = st.columns([1.2, 1.2, 1])
    selected_file = c_left.selectbox(
        "Select `.npy` file from `embeddings/`",
        available_npy if available_npy else ["No .npy files found — run compute_embeddings.py"],
    )
    uploaded_npy = c_mid.file_uploader("Or upload a `.npy` embedding matrix", type=["npy"])
    c_right.slider("Number of Topics (K)", 10, 80, step=5, key="num_topics")

    save_model_chk = st.checkbox(
        "Save trained FASTopic model to `models/` (for projecting comments later)",
        value=True,
    )

    if st.button("Run FASTopic & Log Benchmark", type="primary"):
        if uploaded_npy is not None:
            doc_embeddings = np.load(uploaded_npy)
            model_label = uploaded_npy.name
        elif available_npy:
            doc_embeddings = np.load(EMBED_DIR / selected_file)
            model_label = selected_file
        else:
            st.error("Please generate or upload an `.npy` embedding file first.")
            st.stop()

        if doc_embeddings.shape[0] != len(kept_texts):
            st.error(
                f"Row mismatch! Embedding file has **{doc_embeddings.shape[0]:,}** rows, "
                f"but current Min/Max word filters kept **{len(kept_texts):,}** posts. "
                f"Set Min words=10, Max words=400, Strip URLs=True in Tab 2 to match."
            )
            st.stop()

        class PrecomputedEmbedder:
            def encode(self, docs, **kwargs):
                return doc_embeddings

        num_topics = st.session_state.num_topics
        t0 = time.time()
        with st.spinner(
            f"Fitting FASTopic (K={num_topics}) on {doc_embeddings.shape[0]:,} embeddings ({doc_embeddings.shape[1]}d)..."
        ):
            model = FASTopic(
                num_topics=num_topics,
                doc_embed_model=PrecomputedEmbedder(),
                verbose=False,
            )
            top_words, doc_topic_dist = model.fit_transform(prep_rst["train_texts"])
        fit_sec = round(time.time() - t0, 1)

        if save_model_chk:
            stem = Path(model_label).stem
            model_out = MODEL_DIR / f"fastopic_{stem}_k{num_topics}.pkl"
            with open(model_out, "wb") as f:
                pickle.dump(model, f)
            st.toast(f"Saved model checkpoint to {model_out}")

        topic_lists = [[w for w, _ in model.get_topic(i)[:10]] for i in range(num_topics)]
        unique_words = len(set(w for t in topic_lists for w in t))
        topic_diversity = round(unique_words / (num_topics * 10), 4)

        tokenized_corpus = [t.split() for t in prep_rst["train_texts"]]
        npmi_score = compute_npmi_coherence(topic_lists, tokenized_corpus)

        probs_clipped = np.clip(doc_topic_dist, 1e-12, 1.0)
        mean_entropy = round(
            float(-np.sum(probs_clipped * np.log(probs_clipped), axis=1).mean()), 4
        )

        st.session_state.benchmark_history.append(
            {
                "Embedding File": model_label,
                "Topics (K)": num_topics,
                "Vocab Size": len(prep_rst["vocab"]),
                "Coherence (NPMI) ↑": npmi_score,
                "Topic Diversity ↑": topic_diversity,
                "Doc Entropy ↓": mean_entropy,
                "Fit Time (s)": fit_sec,
            }
        )

        topic_df = pd.DataFrame(
            {
                "Topic ID": range(num_topics),
                "Avg Weight": doc_topic_dist.mean(axis=0).round(4),
                "Top 10 Words": [", ".join(words) for words in topic_lists],
            }
        ).sort_values("Avg Weight", ascending=False)

        st.session_state.last_run_topics = topic_df
        st.session_state.last_run_meta = f"{model_label} | K={num_topics} | Fit in {fit_sec}s"
        st.session_state.last_doc_topic_dist = doc_topic_dist
        st.session_state.last_kept_texts = kept_texts

    if st.session_state.last_run_topics is not None:
        st.subheader(f"Discovered Topics — {st.session_state.last_run_meta}")
        st.dataframe(st.session_state.last_run_topics, use_container_width=True, hide_index=True)

        # Interactive Topic Drill-Down: inspect top posts for any discovered topic
        with st.expander("Inspect Top Posts for a Specific Topic", expanded=True):
            topic_options = [
                f"Topic {row['Topic ID']} ({row['Avg Weight']:.3f}): {row['Top 10 Words'][:70]}..."
                for _, row in st.session_state.last_run_topics.iterrows()
            ]
            chosen_label = st.selectbox("Select topic to inspect top representative posts:", topic_options)
            chosen_tid = int(chosen_label.split()[1])

            probs = st.session_state.last_doc_topic_dist[:, chosen_tid]
            top_doc_indices = np.argsort(probs)[::-1][:10]
            for rank, doc_i in enumerate(top_doc_indices, 1):
                st.markdown(
                    f"**#{rank} (Topic Weight: {probs[doc_i]:.3f}):** "
                    f"{st.session_state.last_kept_texts[doc_i][:350]}..."
                )

    if st.session_state.benchmark_history:
        st.subheader("Side-by-Side Model Comparison Log")
        hist_df = pd.DataFrame(st.session_state.benchmark_history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        if st.button("Clear Comparison Log"):
            st.session_state.benchmark_history = []
            st.rerun()