"""
Evaluation engine for Resume-JD Matcher ranking models.

All rankers follow this interface:

    rank_function(job_description, resumes) -> numpy array

The returned scores must correspond to the resumes
in the same order as the input list.

Current models:
    - TF-IDF
    - BM25
    - MiniLM
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.baselines.current_matcher import (
    rank_candidates as current_matcher_rank,
)
from evaluation.baselines.tfidf import rank_candidates as tfidf_rank
from evaluation.baselines.bm25 import rank_candidates as bm25_rank
from evaluation.baselines.minilm import (
    load_model as load_minilm,
    rank_candidates as minilm_rank,
)

from evaluation.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


# ── Paths ────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent / "results"

RESUMES_FILE = DATA_DIR / "resumes.csv"
JOBS_FILE = DATA_DIR / "jobs.csv"
RELEVANCE_FILE = DATA_DIR / "relevance.csv"

RESULTS_FILE = RESULTS_DIR / "baseline_results.csv"


# ── Benchmark loading ────────────────────────────────────────────────────


def load_benchmark() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Load the prepared benchmark files.

    Returns:
        resumes, jobs, relevance
    """

    required_files = [
        RESUMES_FILE,
        JOBS_FILE,
        RELEVANCE_FILE,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing benchmark file:\n{path}\n\n"
                "Run prepare_benchmark.py first."
            )

    resumes = pd.read_csv(RESUMES_FILE)
    jobs = pd.read_csv(JOBS_FILE)
    relevance = pd.read_csv(RELEVANCE_FILE)

    return resumes, jobs, relevance


# ── Ranker evaluation ────────────────────────────────────────────────────


def evaluate_ranker(
    rank_function,
    resumes: pd.DataFrame,
    jobs: pd.DataFrame,
    relevance: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate one ranking model across all benchmark JDs.

    Parameters
    ----------
    rank_function:
        Function accepting:

            rank_function(job_description, resumes)

        and returning one score per resume.

    resumes:
        DataFrame containing:
            resume_id
            resume_text

    jobs:
        DataFrame containing:
            jd_id
            job_description_text

    relevance:
        DataFrame containing:
            jd_id
            resume_id
            relevance

    Returns
    -------
    pd.DataFrame
        One row of metrics per JD.
    """

    results = []

    # Build a lookup once rather than repeatedly calling set_index().
    resume_lookup = (
        resumes
        .set_index("resume_id")["resume_text"]
        .to_dict()
    )

    for _, job in jobs.iterrows():

        jd_id = job["jd_id"]
        jd_text = job["job_description_text"]

        # Candidates associated with this JD.
        candidates = relevance[
            relevance["jd_id"] == jd_id
        ].copy()

        candidate_ids = candidates["resume_id"].tolist()

        candidate_texts = [
            resume_lookup[resume_id]
            for resume_id in candidate_ids
        ]

        # Run the ranking model.
        scores = rank_function(
            jd_text,
            candidate_texts,
        )

        scores = np.asarray(scores)

        # Validate model output.
        if len(scores) != len(candidate_ids):
            raise ValueError(
                f"{jd_id}: ranker returned {len(scores)} scores "
                f"for {len(candidate_ids)} candidates."
            )

        if not np.all(np.isfinite(scores)):
            raise ValueError(
                f"{jd_id}: ranker returned non-finite scores."
            )

        candidates["score"] = scores

        # Highest score = highest-ranked candidate.
        candidates = candidates.sort_values(
            "score",
            ascending=False,
            kind="mergesort",
        )

        ranked_relevances = (
            candidates["relevance"]
            .astype(int)
            .tolist()
        )

        # Calculate ranking metrics.
        results.append(
            {
                "jd_id": jd_id,
                "num_candidates": len(candidates),

                "ndcg@5": ndcg_at_k(
                    ranked_relevances,
                    5,
                ),

                "ndcg@10": ndcg_at_k(
                    ranked_relevances,
                    10,
                ),

                "mrr": reciprocal_rank(
                    ranked_relevances,
                ),

                "precision@5": precision_at_k(
                    ranked_relevances,
                    5,
                ),

                "recall@5": recall_at_k(
                    ranked_relevances,
                    5,
                ),
            }
        )

    return pd.DataFrame(results)


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:

    print("Loading benchmark...")

    resumes, jobs, relevance = load_benchmark()

    print(f"Resumes: {len(resumes):,}")
    print(f"JDs:     {len(jobs):,}")
    print(f"Pairs:   {len(relevance):,}")

    # Load MiniLM once.
    print("\nLoading MiniLM model...")

    minilm_model = load_minilm()

    # Every ranker follows the same interface.
    rankers = {
    "tfidf": tfidf_rank,
    "bm25": bm25_rank,
    "minilm": lambda jd, resumes: minilm_rank(
        jd,
        resumes,
        minilm_model,
    ),
    "current_matcher": lambda jd, resumes: current_matcher_rank(
        jd,
        resumes,
        minilm_model,
    ),
}

    all_results = []

    # Evaluate every model using the exact same benchmark.
    for name, rank_function in rankers.items():

        print(
            f"\nRunning {name.upper()}..."
        )

        results = evaluate_ranker(
            rank_function,
            resumes,
            jobs,
            relevance,
        )

        results["model"] = name

        all_results.append(results)

    # Combine all model results.
    results = pd.concat(
        all_results,
        ignore_index=True,
    )

    # Create output directory.
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save per-JD results.
    results.to_csv(
        RESULTS_FILE,
        index=False,
    )

    # ── Summary ──────────────────────────────────────────────────────────

    metric_columns = [
        "ndcg@5",
        "ndcg@10",
        "mrr",
        "precision@5",
        "recall@5",
    ]

    summary = (
        results
        .groupby("model")[metric_columns]
        .mean()
        .sort_values(
            "ndcg@5",
            ascending=False,
        )
    )

    print("\n=== BASELINE RESULTS ===")

    print(
        summary
        .round(4)
        .to_string()
    )

    print(
        f"\nPer-JD results written to:\n"
        f"{RESULTS_FILE}"
    )


if __name__ == "__main__":
    main()