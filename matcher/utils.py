"""
utils.py — Shared helpers (improved)

Fixes:
  - #10 Streamlit theme config (generated separately)
  - #12 Metric-aware resume bullet generator
"""


def score_to_label(score: float) -> tuple[str, str]:
    """Convert 0–100 similarity score to (label, color hex)."""
    if score >= 70:
        return "Strong Match", "#22c55e"
    elif score >= 50:
        return "Moderate Match", "#f59e0b"
    elif score >= 35:
        return "Weak Match", "#ef4444"
    else:
        return "Poor Match", "#991b1b"


def impact_score_to_color(score: int) -> str:
    if score >= 75:
        return "#22c55e"
    elif score >= 50:
        return "#f59e0b"
    else:
        return "#ef4444"


def format_section_name(name: str) -> str:
    """Convert section key to display name."""
    return name.replace("_", " ").title()


def generate_resume_bullets(
    matched_skills: list[str],
    impact_count: int,
    semantic_score: float,
    keyword_rate: float,
) -> dict[str, str]:
    """
    #12 Generate track-specific resume bullets with real numbers.
    """
    skill_str = ", ".join(matched_skills[:6]) if matched_skills else "NLP, transformer embeddings, cosine similarity"
    sem = f"{semantic_score:.0f}"
    kw = f"{keyword_rate:.0f}"
    n_skills = len(matched_skills)

    return {
        "AIML / ML Engineer": (
            f"Built a Resume-JD Matcher using HuggingFace `sentence-transformers` (`all-MiniLM-L6-v2`) "
            f"with chunked mean-pool embeddings and cosine similarity; achieved {sem}% semantic alignment "
            f"on test pairs and {kw}% keyword extraction accuracy across {n_skills}+ tech skills "
            f"spanning {skill_str}."
        ),
        "Data Science / Analyst": (
            f"Developed a semantic text-similarity pipeline (resume vs. JD) using transformer embeddings; "
            f"built skill taxonomy extraction across 10 categories ({n_skills} matched skills in testing), "
            f"JD required/preferred section parsing, and an Impact Score metric detecting {impact_count}+ "
            f"quantified achievements per resume."
        ),
        "Full Stack / SDE": (
            f"Engineered a modular Streamlit app with PDF parsing (PyMuPDF), chunked NLP inference "
            f"(HuggingFace sentence-transformers), multi-resume comparison mode, and session-state caching; "
            f"deployed on Streamlit Cloud with <2s cold-start after model warmup."
        ),
    }