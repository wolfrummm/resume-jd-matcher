"""
Ranking metrics for Resume-JD Matcher evaluation.
"""

from __future__ import annotations

import math


def precision_at_k(
    ranked_relevances: list[int],
    k: int,
    relevance_threshold: int = 1,
) -> float:
    """Precision@K using relevance >= threshold as relevant."""
    if k <= 0:
        return 0.0

    top_k = ranked_relevances[:k]

    if not top_k:
        return 0.0

    relevant = sum(
        relevance >= relevance_threshold
        for relevance in top_k
    )

    return relevant / len(top_k)


def recall_at_k(
    ranked_relevances: list[int],
    k: int,
    relevance_threshold: int = 1,
) -> float:
    """Recall@K using relevance >= threshold as relevant."""
    if k <= 0:
        return 0.0

    total_relevant = sum(
        relevance >= relevance_threshold
        for relevance in ranked_relevances
    )

    if total_relevant == 0:
        return 0.0

    retrieved_relevant = sum(
        relevance >= relevance_threshold
        for relevance in ranked_relevances[:k]
    )

    return retrieved_relevant / total_relevant


def reciprocal_rank(
    ranked_relevances: list[int],
    relevance_threshold: int = 1,
) -> float:
    """
    Reciprocal rank of the first relevant candidate.
    """
    for rank, relevance in enumerate(ranked_relevances, start=1):
        if relevance >= relevance_threshold:
            return 1.0 / rank

    return 0.0


def _dcg_at_k(relevances: list[int], k: int) -> float:
    """Discounted cumulative gain."""
    score = 0.0

    for rank, relevance in enumerate(relevances[:k], start=1):
        score += (
            (2**relevance - 1)
            / math.log2(rank + 1)
        )

    return score


def ndcg_at_k(
    ranked_relevances: list[int],
    k: int,
) -> float:
    """
    Normalized Discounted Cumulative Gain.

    Relevance:
        0 = No Fit
        1 = Potential Fit
        2 = Good Fit
    """
    if k <= 0:
        return 0.0

    actual = _dcg_at_k(ranked_relevances, k)

    ideal = sorted(
        ranked_relevances,
        reverse=True,
    )

    ideal_dcg = _dcg_at_k(ideal, k)

    if ideal_dcg == 0:
        return 0.0

    return actual / ideal_dcg