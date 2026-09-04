"""
Current Resume-JD Matcher ranking implementation.

This adapter exposes the scoring logic currently used by
the application's Compare Resumes mode so it can be
evaluated against standard retrieval baselines.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from matcher.analyzer import analyze_skill_gap, compute_impact_score
from matcher.embedder import batch_compute_similarity
from matcher.extractor import split_into_sections


def rank_candidates(
    job_description: str,
    resumes: list[str],
    model: SentenceTransformer,
) -> np.ndarray:
    """
    Rank resumes using the current application's scoring logic.

    Current production recommendation score:

        40% semantic similarity
        40% required-skill coverage
        20% impact score

    Returns one score per resume in the same order as input.
    """
    if not resumes:
        return np.array([], dtype=float)

    semantic_scores = batch_compute_similarity(
        resumes,
        job_description,
        model,
    )

    scores = []

    for resume_text, semantic_score in zip(resumes, semantic_scores):
        sections = split_into_sections(resume_text)

        # Semantic skill fallback is intentionally disabled here because
        # possible_matches currently does not contribute to the score.
        gap = analyze_skill_gap(
            resume_text,
            job_description,
            model=None,
        )

        impact = compute_impact_score(
            resume_text,
            sections,
        )

        semantic_score_100 = semantic_score * 100
        required_skill_score = gap["required_match_rate"]
        impact_score = impact["score"]

        combined_score = (
            semantic_score_100 * 0.40
            + required_skill_score * 0.40
            + impact_score * 0.20
        )

        scores.append(combined_score)

    return np.asarray(scores, dtype=float)