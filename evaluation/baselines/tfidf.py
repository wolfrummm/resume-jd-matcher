"""
TF-IDF lexical baseline for resume-JD ranking.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def rank_candidates(
    job_description: str,
    resumes: list[str],
) -> np.ndarray:
    """
    Rank resumes against a job description using TF-IDF cosine similarity.

    Returns:
        Similarity score for each resume in the original input order.
    """
    if not resumes:
        return np.array([], dtype=float)

    corpus = [job_description, *resumes]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )

    matrix = vectorizer.fit_transform(corpus)

    jd_vector = matrix[0]
    resume_vectors = matrix[1:]

    scores = cosine_similarity(
        jd_vector,
        resume_vectors,
    ).ravel()

    return scores.astype(float)