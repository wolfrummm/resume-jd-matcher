"""
Prepare the Resume-JD Matcher evaluation benchmark.

Source:
    evaluation/data/raw/test.csv

The source dataset provides three relevance labels:
    No Fit
    Potential Fit
    Good Fit

This script:
    1. Loads the official test split.
    2. Removes duplicate resume-JD pairs.
    3. Groups candidates by job description.
    4. Keeps JDs with at least 5 unique candidates
       and at least 2 relevance levels.
    5. Assigns deterministic IDs.
    6. Writes normalized benchmark files.

The raw and processed datasets are intentionally ignored by Git.
"""

from pathlib import Path

import pandas as pd


RAW_DIR = Path(__file__).parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent / "data" / "processed"

TEST_FILE = RAW_DIR / "test.csv"

RESUMES_FILE = PROCESSED_DIR / "resumes.csv"
JOBS_FILE = PROCESSED_DIR / "jobs.csv"
RELEVANCE_FILE = PROCESSED_DIR / "relevance.csv"


LABEL_MAP = {
    "No Fit": 0,
    "Potential Fit": 1,
    "Good Fit": 2,
}

MIN_CANDIDATES_PER_JD = 5
MIN_LABELS_PER_JD = 2


def normalize_text(text: str) -> str:
    """Normalize whitespace for stable deduplication and IDs."""
    return " ".join(str(text).split())


def main() -> None:
    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Missing source dataset: {TEST_FILE}\n"
            "Place test.csv in evaluation/data/raw/."
        )

    print(f"Loading: {TEST_FILE}")

    df = pd.read_csv(TEST_FILE)

    required_columns = {
        "resume_text",
        "job_description_text",
        "label",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    print(f"Source rows: {len(df):,}")

    # Normalize text before deduplication.
    df["resume_text"] = df["resume_text"].map(normalize_text)
    df["job_description_text"] = df["job_description_text"].map(
        normalize_text
    )

    # Remove exact duplicate resume-JD pairs.
    before_dedup = len(df)

    df = df.drop_duplicates(
        subset=["resume_text", "job_description_text"],
        keep="first",
    ).reset_index(drop=True)

    duplicates_removed = before_dedup - len(df)

    print(f"Duplicate pairs removed: {duplicates_removed:,}")
    print(f"Rows after deduplication: {len(df):,}")

    # Validate labels.
    unknown_labels = set(df["label"]) - set(LABEL_MAP)

    if unknown_labels:
        raise ValueError(
            f"Unknown labels found: {sorted(unknown_labels)}"
        )

    df["relevance"] = df["label"].map(LABEL_MAP)

    # Stable identifiers based on sorted unique text.
    unique_resumes = sorted(df["resume_text"].unique())
    unique_jds = sorted(df["job_description_text"].unique())

    resume_ids = {
        text: f"R{i:04d}"
        for i, text in enumerate(unique_resumes, start=1)
    }

    jd_ids = {
        text: f"J{i:04d}"
        for i, text in enumerate(unique_jds, start=1)
    }

    df["resume_id"] = df["resume_text"].map(resume_ids)
    df["jd_id"] = df["job_description_text"].map(jd_ids)

    # Determine which JDs are suitable ranking queries.
    jd_stats = (
        df.groupby("jd_id")
        .agg(
            candidate_count=("resume_id", "nunique"),
            label_count=("relevance", "nunique"),
        )
        .reset_index()
    )

    eligible_jds = jd_stats[
        (jd_stats["candidate_count"] >= MIN_CANDIDATES_PER_JD)
        & (jd_stats["label_count"] >= MIN_LABELS_PER_JD)
    ]["jd_id"].tolist()

    eligible_jds = sorted(eligible_jds)

    benchmark = df[df["jd_id"].isin(eligible_jds)].copy()

    # Keep only the fields required by each output file.
    resumes = (
        benchmark[["resume_id", "resume_text"]]
        .drop_duplicates()
        .sort_values("resume_id")
        .reset_index(drop=True)
    )

    jobs = (
        benchmark[["jd_id", "job_description_text"]]
        .drop_duplicates()
        .sort_values("jd_id")
        .reset_index(drop=True)
    )

    relevance = (
        benchmark[["jd_id", "resume_id", "relevance"]]
        .sort_values(["jd_id", "resume_id"])
        .reset_index(drop=True)
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    resumes.to_csv(RESUMES_FILE, index=False)
    jobs.to_csv(JOBS_FILE, index=False)
    relevance.to_csv(RELEVANCE_FILE, index=False)

    print("\n=== BENCHMARK SUMMARY ===")
    print(f"Eligible JDs:       {len(jobs):,}")
    print(f"Unique resumes:     {len(resumes):,}")
    print(f"Evaluation pairs:   {len(relevance):,}")

    candidate_counts = relevance.groupby("jd_id")["resume_id"].nunique()

    print(
        f"Candidates per JD:  "
        f"{candidate_counts.min()}–{candidate_counts.max()}"
    )

    print(
        f"Average candidates: "
        f"{candidate_counts.mean():.2f}"
    )

    print("\nLabel distribution:")

    label_counts = relevance["relevance"].value_counts().sort_index()

    label_names = {
        0: "No Fit",
        1: "Potential Fit",
        2: "Good Fit",
    }

    for relevance_value, count in label_counts.items():
        print(
            f"  {relevance_value} - "
            f"{label_names[relevance_value]:<16} "
            f"{count:>6,}"
        )

    print("\nWritten files:")
    print(f"  {RESUMES_FILE}")
    print(f"  {JOBS_FILE}")
    print(f"  {RELEVANCE_FILE}")

    print("\nBenchmark preparation complete.")


if __name__ == "__main__":
    main()