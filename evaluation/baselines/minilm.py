"""
MiniLM semantic retrieval baseline.

Uses the same sentence-transformer model as the
production Resume-JD Matcher.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "all-MiniLM-L6-v2"


def load_model() -> SentenceTransformer:
    """Load the SentenceTransformer model."""
    return SentenceTransformer(MODEL_NAME)


def rank_candidates(
    job_description: str,
    resumes: list[str],
    model: SentenceTransformer,
) -> np.ndarray:
    """
    Rank resumes against a job description using
    sentence-transformer embeddings.

    Returns scores in the same order as `resumes`.
    """

    if not resumes:
        return np.array([], dtype=float)

    texts = [job_description, *resumes]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    jd_embedding = embeddings[0]
    resume_embeddings = embeddings[1:]

    scores = cosine_similarity(
        jd_embedding.reshape(1, -1),
        resume_embeddings,
    ).ravel()

    return scores.astype(float)