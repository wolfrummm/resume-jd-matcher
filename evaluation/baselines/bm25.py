"""
BM25 lexical retrieval baseline.
"""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Simple word tokenizer for BM25."""
    return re.findall(
        r"\b\w+\b",
        text.lower(),
    )


def rank_candidates(
    job_description: str,
    resumes: list[str],
) -> np.ndarray:
    """
    Rank resumes against a JD using BM25.

    Returns scores in the same order as the input resumes.
    """
    if not resumes:
        return np.array([], dtype=float)

    tokenized_resumes = [
        tokenize(resume)
        for resume in resumes
    ]

    query = tokenize(job_description)

    bm25 = BM25Okapi(tokenized_resumes)

    scores = bm25.get_scores(query)

    return np.asarray(scores, dtype=float)