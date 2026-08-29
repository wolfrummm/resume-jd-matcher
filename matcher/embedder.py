"""
embedder.py — Sentence-transformer encoding + cosine similarity (improved)

Fixes:
  - #4  Section-level embedding instead of truncating full text
  - #9  Session state awareness (model cached via st.cache_resource)
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache


MODEL_NAME = "all-MiniLM-L6-v2"
# Safe per-chunk limit for this model (512 tokens ≈ 1800 chars)
CHUNK_CHAR_LIMIT = 1800


@lru_cache(maxsize=1)
def load_model():
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


def embed_long_text(text: str, model: SentenceTransformer) -> np.ndarray:
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
    emb_a = embed_long_text(text_a, model)
    emb_b = embed_long_text(text_b, model)
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
    jd_emb = embed_long_text(jd_text, model)
    scores = {}

    for section_name, section_text in sections.items():
        if section_name == "header":
            continue
        if section_text and len(section_text.strip()) > 20:
            sec_emb = embed_long_text(section_text, model)
            raw = cosine_similarity([sec_emb], [jd_emb])[0][0]
            scores[section_name] = round(float(np.clip(raw, 0, 1)) * 100, 1)

    return scores

def find_semantic_skill_matches(
    missing_skills: list[str],
    resume_text: str,
    model: SentenceTransformer,
    threshold: float = 0.62,
) -> dict[str, float]:
    """
    For JD skills that didn't exact-match, check if a semantically similar
    phrase exists in the resume (e.g. 'ReactJS' in resume vs 'react' in JD).
    Returns {skill: similarity_score} for skills above threshold.
    """
    if not missing_skills:
        return {}

    lines = [l.strip() for l in resume_text.split("\n") if 3 < len(l.strip()) < 120]
    if not lines:
        return {}

    skill_embs = encode(missing_skills, model)
    line_embs = encode(lines, model)
    sims = cosine_similarity(skill_embs, line_embs)

    matches = {}
    for i, skill in enumerate(missing_skills):
        best = float(np.max(sims[i]))
        if best >= threshold:
            matches[skill] = round(best, 2)
    return matches

_SECTION_WEIGHTS = {"skills": 0.35, "experience": 0.35, "projects": 0.20, "education": 0.05, "summary": 0.05}


def compute_weighted_overall_score(section_scores: dict, fallback_score: float) -> float:
    """
    Blend section scores using weights favoring skills/experience/projects,
    instead of scoring the raw full resume text (which dilutes with
    contact info, dates, etc). Falls back to raw score if sections are sparse.
    """
    if len(section_scores) <= 1:
        return fallback_score

    weighted_sum, weight_total = 0.0, 0.0
    for section, score in section_scores.items():
        w = _SECTION_WEIGHTS.get(section, 0.05)
        weighted_sum += score * w
        weight_total += w

    return round(weighted_sum / weight_total, 1) if weight_total else fallback_score

def compute_similarity_with_embedding(text_a_emb: np.ndarray, text_b: str, model: SentenceTransformer) -> float:
    """Like compute_similarity, but takes a precomputed embedding for one side."""
    emb_b = embed_long_text(text_b, model)
    score = cosine_similarity([text_a_emb], [emb_b])[0][0]
    return float(np.clip(score, 0.0, 1.0))


def batch_compute_similarity(texts: list[str], reference_text: str, model: SentenceTransformer) -> list[float]:
    """
    Compute similarity of many texts against one reference in a single batched
    encode() call — faster than encoding each resume one-by-one in a loop.
    """
    ref_emb = embed_long_text(reference_text, model)

    all_chunks, owner_index = [], []
    for i, text in enumerate(texts):
        chunks = _chunk_text(text)
        all_chunks.extend(chunks)
        owner_index.extend([i] * len(chunks))

    chunk_embs = encode(all_chunks, model)

    pooled = [[] for _ in texts]
    for idx, emb in zip(owner_index, chunk_embs):
        pooled[idx].append(emb)

    scores = []
    for chunk_embs_for_resume in pooled:
        resume_emb = np.mean(chunk_embs_for_resume, axis=0)
        sim = cosine_similarity([resume_emb], [ref_emb])[0][0]
        scores.append(float(np.clip(sim, 0.0, 1.0)))
    return scores