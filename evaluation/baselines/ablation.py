"""
Ablation rankers for the Resume-JD Matcher.

Each variant starts with the same MiniLM semantic score and
optionally adds the existing skill-match and impact signals.

Variants:
    - minilm
    - minilm_skills
    - minilm_impact
    - full_hybrid
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from matcher.analyzer import (
    analyze_skill_gap,
    compute_impact_score,
)
from matcher.embedder import batch_compute_similarity
from matcher.extractor import split_into_sections


def rank_candidates(
    job_description: str,
    resumes: list[str],
    model: SentenceTransformer,
    variant: str,
) -> np.ndarray:
    """
    Rank candidates using one ablation variant.

    Scores are normalized to the same 0-100 scale:

        semantic score       -> 0-100
        required skill score -> 0-100
        impact score         -> 0-100
    """

    if not resumes:
        return np.array([], dtype=float)

    if variant not in {
        "minilm",
        "minilm_skills",
        "minilm_impact",
        "full_hybrid",
    }:
        raise ValueError(f"Unknown ablation variant: {variant}")

    # Same semantic signal used by the current matcher.
    semantic_scores = batch_compute_similarity(
        resumes,
        job_description,
        model,
    )

    scores = []

    for resume_text, semantic_score in zip(
        resumes,
        semantic_scores,
    ):
        semantic_score_100 = float(semantic_score) * 100

        # MiniLM-only baseline.
        if variant == "minilm":
            scores.append(semantic_score_100)
            continue

        gap = analyze_skill_gap(
            resume_text,
            job_description,
            model=None,
        )

        required_skill_score = float(
            gap["required_match_rate"]
        )

        if variant == "minilm_skills":
            combined_score = (
                semantic_score_100 * 0.5
                + required_skill_score * 0.5
            )
            scores.append(combined_score)
            continue

        sections = split_into_sections(resume_text)

        impact = compute_impact_score(
            resume_text,
            sections,
        )

        impact_score = float(impact["score"])

        if variant == "minilm_impact":
            combined_score = (
                semantic_score_100 * 0.5
                + impact_score * 0.5
            )
            scores.append(combined_score)
            continue

        # Existing production hybrid:
        # 40% semantic + 40% required skills + 20% impact.
        combined_score = (
            semantic_score_100 * 0.40
            + required_skill_score * 0.40
            + impact_score * 0.20
        )

        scores.append(combined_score)

    return np.asarray(scores, dtype=float)