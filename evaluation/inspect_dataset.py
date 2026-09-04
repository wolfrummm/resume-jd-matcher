from collections import defaultdict, deque
from collections import Counter
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).parent / "data" / "raw"


def normalize_text(text: str) -> str:
    """Normalize whitespace for duplicate detection."""
    return " ".join(str(text).lower().split())


def inspect_split(name: str, df: pd.DataFrame) -> dict:
    resumes = df["resume_text"].map(normalize_text)
    jds = df["job_description_text"].map(normalize_text)

    pairs = list(zip(resumes, jds))

    return {
        "name": name,
        "rows": len(df),
        "unique_resumes": resumes.nunique(),
        "unique_jds": jds.nunique(),
        "duplicate_pairs": len(pairs) - len(set(pairs)),
        "label_counts": Counter(df["label"]),
        "resumes": set(resumes),
        "jds": set(jds),
        "pairs": set(pairs),
    }


def main() -> None:
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Expected train.csv and test.csv in evaluation/data/raw/"
        )

    print("Loading local benchmark files...")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    print("\n=== DATASET SCHEMA ===")
    print("Columns:", list(train.columns))

    train_info = inspect_split("train", train)
    test_info = inspect_split("test", test)

    print("\n=== BASIC DATASET STATISTICS ===")

    for info in (train_info, test_info):
        print(f"\n{info['name'].upper()}")
        print(f"Rows:              {info['rows']:,}")
        print(f"Unique resumes:    {info['unique_resumes']:,}")
        print(f"Unique JDs:        {info['unique_jds']:,}")
        print(f"Duplicate pairs:   {info['duplicate_pairs']:,}")

        print("Labels:")
        for label, count in info["label_counts"].items():
            percentage = count / info["rows"] * 100
            print(f"  {label:<16} {count:>6,} ({percentage:5.2f}%)")

    print("\n=== TRAIN / TEST OVERLAP ===")

    resume_overlap = train_info["resumes"] & test_info["resumes"]
    jd_overlap = train_info["jds"] & test_info["jds"]
    pair_overlap = train_info["pairs"] & test_info["pairs"]

    print(f"Resumes in BOTH splits: {len(resume_overlap):,}")
    print(f"JDs in BOTH splits:     {len(jd_overlap):,}")
    print(f"Exact pairs in BOTH:     {len(pair_overlap):,}")

    combined = pd.concat([train, test], ignore_index=True)

    all_resumes = combined["resume_text"].map(normalize_text)
    all_jds = combined["job_description_text"].map(normalize_text)

    print("\n=== COMBINED DATASET ===")
    print(f"Total rows:              {len(combined):,}")
    print(f"Unique resumes:          {all_resumes.nunique():,}")
    print(f"Unique JDs:              {all_jds.nunique():,}")

    pairs = list(zip(all_resumes, all_jds))
    print(f"Unique resume-JD pairs:  {len(set(pairs)):,}")
    print(f"Duplicate pairs:         {len(pairs) - len(set(pairs)):,}")

    print("\n=== CANDIDATES PER JD ===")

    jd_counts = all_jds.value_counts()

    print(f"Minimum candidates/JD: {jd_counts.min()}")
    print(f"Maximum candidates/JD: {jd_counts.max()}")
    print(f"Average candidates/JD: {jd_counts.mean():.2f}")

    for threshold in (3, 5, 10):
        eligible = (jd_counts >= threshold).sum()
        print(
            f"JDs with >= {threshold:2} candidates: "
            f"{eligible:,} / {len(jd_counts):,}"
        )

    print("\n=== LABEL COVERAGE PER JD ===")

    label_groups = (
        combined.assign(_jd_id=all_jds)
        .groupby("_jd_id")["label"]
        .agg(lambda labels: tuple(sorted(set(labels))))
    )

    coverage = Counter(label_groups)

    for labels, count in sorted(
        coverage.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        print(f"{count:>4} JDs: {labels}")

    all_three = sum(len(labels) == 3 for labels in label_groups)

    print(f"\nJDs containing all 3 labels: {all_three:,}")

    print("\n=== RESUME-JD CONNECTIVITY ===")

    resume_to_jds = defaultdict(set)
    jd_to_resumes = defaultdict(set)

    for resume_id, jd_id in zip(all_resumes, all_jds):
        resume_to_jds[resume_id].add(jd_id)
        jd_to_resumes[jd_id].add(resume_id)

    multi_jd_resumes = sum(
        len(jds) > 1
        for jds in resume_to_jds.values()
    )

    print(
        f"Resumes associated with multiple JDs: "
        f"{multi_jd_resumes:,} / {len(resume_to_jds):,}"
    )

    # Build connected components in the bipartite resume-JD graph.
    graph = defaultdict(set)

    for resume_id, jd_id in zip(all_resumes, all_jds):
        resume_node = f"R::{resume_id}"
        jd_node = f"J::{jd_id}"

        graph[resume_node].add(jd_node)
        graph[jd_node].add(resume_node)

    visited = set()
    components = []

    for node in graph:
        if node in visited:
            continue

        queue = deque([node])
        visited.add(node)

        component = set()

        while queue:
            current = queue.popleft()
            component.add(current)

            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        components.append(component)

    component_stats = []

    for component in components:
        resumes = sum(node.startswith("R::") for node in component)
        jds = sum(node.startswith("J::") for node in component)

        component_stats.append((resumes, jds))

    component_stats.sort(reverse=True)

    print(f"Connected components: {len(component_stats):,}")

    print("\nLargest connected components:")
    for resumes, jds in component_stats[:10]:
        print(
            f"  {resumes:>4} resumes × "
            f"{jds:>4} JDs"
        )
    print("\n=== OFFICIAL TEST-JD RANKING ELIGIBILITY ===")

    test_jd_groups = (
        test.assign(_jd_id=test["job_description_text"].map(normalize_text))
        .groupby("_jd_id")
    )

    eligible_test_jds = []

    for jd_id, group in test_jd_groups:
        labels = set(group["label"])
        candidate_count = group["resume_text"].map(normalize_text).nunique()

        if len(labels) >= 2 and candidate_count >= 3:
            eligible_test_jds.append(
                {
                    "jd_id": jd_id,
                    "candidates": candidate_count,
                    "labels": labels,
                }
            )

    print(f"Official test JDs: {len(test_jd_groups):,}")
    print(
        f"Eligible ranking JDs: "
        f"{len(eligible_test_jds):,}"
    )

    print(
        f"Excluded JDs: "
        f"{len(test_jd_groups) - len(eligible_test_jds):,}"
    )

    print("\nEligible test JDs by label coverage:")

    coverage_counts = Counter(
        tuple(sorted(item["labels"]))
        for item in eligible_test_jds
    )

    for labels, count in sorted(
        coverage_counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        print(f"{count:>4} JDs: {labels}")

    if eligible_test_jds:
        candidate_counts = [
            item["candidates"]
            for item in eligible_test_jds
        ]

        print("\nCandidate counts for eligible JDs:")
        print(f"  Minimum: {min(candidate_counts)}")
        print(f"  Maximum: {max(candidate_counts)}")
        print(
            f"  Average: "
            f"{sum(candidate_counts) / len(candidate_counts):.2f}"
        )
    print("\n=== DATASET AUDIT COMPLETE ===")


if __name__ == "__main__":
    main()