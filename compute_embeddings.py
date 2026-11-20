import os
import re
import time
from pathlib import Path

import torch
import numpy as np
import pyarrow.parquet as pq
from dotenv import load_dotenv
from huggingface_hub import login
from sentence_transformers import SentenceTransformer

# 1. Load HF_TOKEN from .env
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN and not HF_TOKEN.startswith("hf_paste_your"):
    login(token=HF_TOKEN, add_to_git_credential=False)

DATA_PATH = Path("data/filtered_submissions.parquet")
EMBED_DIR = Path("embeddings")
EMBED_DIR.mkdir(parents=True, exist_ok=True)

# Match your Streamlit preprocessing settings
MIN_WORDS = 10
MAX_WORDS = 400
STRIP_URLS = True

MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((?:https?://|www\.)[^)]+\)")
URL_RE = re.compile(r"(?:https?://|www\.)\S+")
HTML_ENT_RE = re.compile(r"&(?:amp|lt|gt|quot|#x200B);")


def clean_reddit_text(text: str) -> str:
    text = HTML_ENT_RE.sub(" ", text)
    if STRIP_URLS:
        text = MD_LINK_RE.sub(r"\1", text)
        text = URL_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_kept_texts():
    df = pq.read_table(DATA_PATH).to_pandas()
    df = df[df["type"] == "submission"].copy()
    raw_texts = (df["submission_title"].fillna("") + " " + df["body"].fillna("")).str.strip().tolist()

    kept = []
    for t in raw_texts:
        words = clean_reddit_text(t).split()
        if len(words) >= MIN_WORDS:
            kept.append(" ".join(words[:MAX_WORDS]))
    return kept


def encode_model(texts, model_name: str, out_filename: str):
    out_path = EMBED_DIR / out_filename
    if out_path.exists():
        print(f"[SKIP] {out_path} already exists.")
        return

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Encoding {len(texts):,} posts with {model_name} on {device.upper()} ---")
    start_t = time.time()

    if model_name == "BAAI/bge-m3":
        kwargs = {"torch_dtype": torch.float16} if device in ("cuda", "mps") else {}
        model = SentenceTransformer(model_name, device=device, model_kwargs=kwargs)
        inputs = texts
    else:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        model = SentenceTransformer(model_name, device=device, token=HF_TOKEN, model_kwargs={"torch_dtype": dtype})
        inputs = [f"task: clustering | query: {t}" for t in texts]

    model.max_seq_length = 512

    # Sort by length so short posts aren't padded to 512 tokens
    order = np.argsort([len(t) for t in inputs])
    sorted_inputs = [inputs[i] for i in order]

    chunk_size = 5000
    chunks = []
    for i in range(0, len(sorted_inputs), chunk_size):
        batch_slice = sorted_inputs[i : i + chunk_size]
        emb = model.encode(batch_slice, batch_size=16, show_progress_bar=True, normalize_embeddings=True)
        chunks.append(emb)
        if device == "mps":
            torch.mps.empty_cache()
        print(f"Progress: {min(i + len(batch_slice), len(sorted_inputs)):,} / {len(sorted_inputs):,}")

    all_sorted = np.vstack(chunks)
    embs = np.empty_like(all_sorted)
    embs[order] = all_sorted

    np.save(out_path, embs)
    print(f"[DONE] Saved {embs.shape} to {out_path} in {(time.time() - start_t)/60:.1f} mins.")


if __name__ == "__main__":
    kept_texts = load_kept_texts()
    encode_model(kept_texts, "google/embeddinggemma-300m", "gemma_300m_min10_max400.npy")
    encode_model(kept_texts, "BAAI/bge-m3", "bge_m3_min10_max400.npy")