"""
embedder.py — Sentence-transformer encoding + cosine similarity (improved)

Fixes:
  - #4  Section-level embedding instead of truncating full text
  - #9  Session state awareness (model cached via st.cache_resource)
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st


MODEL_NAME = "all-MiniLM-L6-v2"
# Safe per-chunk limit for this model (512 tokens ≈ 1800 chars)
CHUNK_CHAR_LIMIT = 1800


@st.cache_resource(show_spinner=False)
def load_model() -> SentenceTransformer:
    """Load and cache the sentence-transformer model (persists across reruns)."""
    return SentenceTransformer(MODEL_NAME)


def encode(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    """Encode a list of strings into embeddings."""
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def _chunk_text(text: str, max_chars: int = CHUNK_CHAR_LIMIT) -> list[str]:
    """
    Split long text into overlapping chunks that respect sentence boundaries.
    Prevents information loss from hard truncation.
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    current = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent) + 2  # +2 for ". "
        if current_len + sent_len > max_chars and current:
            chunks.append(". ".join(current) + ".")
            # Overlap: keep last 2 sentences for context continuity
            current = current[-2:]
            current_len = sum(len(s) + 2 for s in current)
        current.append(sent)
        current_len += sent_len

    if current:
        chunks.append(". ".join(current))

    return chunks if chunks else [text[:max_chars]]


def _embed_long_text(text: str, model: SentenceTransformer) -> np.ndarray:
    """
    Embed a potentially long text by chunking and mean-pooling chunk embeddings.
    Fixes #4: no more lossy hard truncation.
    """
    chunks = _chunk_text(text)
    embeddings = encode(chunks, model)
    # Mean pool across chunks (weighted equally)
    return np.mean(embeddings, axis=0)


def compute_similarity(text_a: str, text_b: str, model: SentenceTransformer) -> float:
    """
    Compute cosine similarity between two (potentially long) texts.
    Uses chunked mean-pooling instead of truncation.
    Returns float in [0, 1].
    """
    emb_a = _embed_long_text(text_a, model)
    emb_b = _embed_long_text(text_b, model)
    score = cosine_similarity([emb_a], [emb_b])[0][0]
    return float(np.clip(score, 0.0, 1.0))


def compute_section_scores(
    sections: dict,
    jd_text: str,
    model: SentenceTransformer,
) -> dict:
    """
    Score each resume section against the full JD.
    Returns dict: { section_name: score_0_to_100 }
    """
    jd_emb = _embed_long_text(jd_text, model)
    scores = {}

    for section_name, section_text in sections.items():
        if section_text and len(section_text.strip()) > 20:
            sec_emb = _embed_long_text(section_text, model)
            raw = cosine_similarity([sec_emb], [jd_emb])[0][0]
            scores[section_name] = round(float(np.clip(raw, 0, 1)) * 100, 1)

    return scores