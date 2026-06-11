"""
analyzer.py — Skill gap analysis + Impact Score + JD section parsing (improved)

Fixes:
  - #2  Expanded taxonomy (300+ skills, India-specific, aliases)
  - #5  Impact Score: checks for quantified achievements in experience/projects
  - #6  JD section parsing: Required vs Nice-to-have skill weighting
  - #7  Score calibration context
"""

import json
import re
from pathlib import Path


_SKILL_FILE = Path(__file__).parent.parent / "data" / "skill_keywords.json"
with open(_SKILL_FILE, "r") as f:
    SKILL_TAXONOMY: dict = json.load(f)

ALL_SKILLS: list[str] = [
    skill for category in SKILL_TAXONOMY.values() for skill in category
]

# Skill aliases: normalize common abbreviations to canonical form
SKILL_ALIASES: dict[str, str] = {
    "ml": "machine learning",
    "dl": "deep learning",
    "ai": "machine learning",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "tf": "tensorflow",
    "sk": "scikit-learn",
    "sklearn": "scikit-learn",
    "hf": "huggingface",
    "llms": "large language model",
    "llm": "large language model",
    "oop": "oops",
    "oop's": "oops",
    "dsa": "data structures",
    "ds": "data structures",
    "algo": "algorithms",
    "k8s": "kubernetes",
    "k8": "kubernetes",
    "pg": "postgresql",
    "postgres": "postgresql",
    "mongo": "mongodb",
    "gh": "github",
    "gha": "github actions",
    "sd": "system design",
    "lld": "low level design",
    "hld": "high level design",
    "next": "next.js",
    "node": "node.js",
    "vue3": "vue",
    "angular2": "angular",
}

# Patterns that signal "required" vs "nice to have" in JD sections
_REQUIRED_PATTERNS = re.compile(
    r"\b(required|must have|mandatory|essential|minimum|you (must|should|will|need)|"
    r"we (require|expect|need)|responsibilities|you are|strong (in|background|experience|knowledge))\b",
    re.IGNORECASE,
)
_PREFERRED_PATTERNS = re.compile(
    r"\b(preferred|nice to have|bonus|plus|good to have|advantageous|ideally|"
    r"familiarity|exposure|desired|optional)\b",
    re.IGNORECASE,
)

# Impact indicators: numbers, percentages, metrics, scale words
_IMPACT_PATTERNS = [
    r"\b\d+[\+%]?\s*(users?|customers?|requests?|queries|transactions|records|rows|samples?|"
    r"models?|endpoints?|apis?|services?|clients?|teams?|members?|students?)\b",
    r"\b(reduced?|improved?|increased?|decreased?|optimized?|boosted?|cut|halved?|doubled?|tripled?)\b.{0,40}\b\d+[\+%]?\b",
    r"\b\d+[\+%]?\s*(x|times|fold|faster|slower|accurate|accuracy|precision|recall|f1|auc|mse|rmse|mae)\b",
    r"\b(latency|throughput|uptime|availability|scalab)\w*\b.{0,30}\d+",
    r"\$\s*\d+",
    r"\b\d{4,}\b",  # large numbers (10000+)
    r"\b(top|rank|position|place|winner|finalist)\s+\d+\b",
]
_IMPACT_RE = re.compile("|".join(_IMPACT_PATTERNS), re.IGNORECASE)


def _apply_aliases(text: str) -> str:
    """Expand known abbreviations in text before skill extraction."""
    words = text.split()
    expanded = []
    for word in words:
        clean = re.sub(r"[^\w]", "", word.lower())
        expanded.append(SKILL_ALIASES.get(clean, word))
    return " ".join(expanded)


def extract_skills_from_text(text: str) -> set[str]:
    """Return the set of known skills found in a block of text (with alias expansion)."""
    expanded = _apply_aliases(text)
    text_lower = expanded.lower()
    found = set()
    for skill in ALL_SKILLS:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            found.add(skill)
    return found


def parse_jd_sections(jd_text: str) -> dict[str, str]:
    """
    Split JD into required vs preferred blocks for weighted skill analysis.
    Returns {"required": text, "preferred": text, "full": full_text}
    """
    lines = jd_text.split("\n")
    required_lines, preferred_lines = [], []
    mode = "required"  # default: treat everything as required unless flagged

    for line in lines:
        if _PREFERRED_PATTERNS.search(line):
            mode = "preferred"
        elif _REQUIRED_PATTERNS.search(line):
            mode = "required"
        if mode == "preferred":
            preferred_lines.append(line)
        else:
            required_lines.append(line)

    return {
        "required": "\n".join(required_lines),
        "preferred": "\n".join(preferred_lines),
        "full": jd_text,
    }


def analyze_skill_gap(resume_text: str, jd_text: str) -> dict:
    """
    Compare skills in JD vs resume, with required/preferred weighting.

    Returns:
        {
            "jd_skills": [...],
            "required_skills": [...],
            "preferred_skills": [...],
            "matched": [...],
            "matched_required": [...],
            "matched_preferred": [...],
            "missing": [...],
            "missing_required": [...],
            "resume_only": [...],
            "match_rate": float,
            "required_match_rate": float,
        }
    """
    jd_sections = parse_jd_sections(jd_text)

    jd_skills = extract_skills_from_text(jd_text)
    required_skills = extract_skills_from_text(jd_sections["required"])
    preferred_skills = extract_skills_from_text(jd_sections["preferred"])
    resume_skills = extract_skills_from_text(resume_text)

    matched = jd_skills & resume_skills
    missing = jd_skills - resume_skills
    matched_required = required_skills & resume_skills
    missing_required = required_skills - resume_skills
    matched_preferred = preferred_skills & resume_skills
    resume_only = resume_skills - jd_skills

    match_rate = (len(matched) / len(jd_skills) * 100) if jd_skills else 0.0
    req_match_rate = (len(matched_required) / len(required_skills) * 100) if required_skills else match_rate

    return {
        "jd_skills": sorted(jd_skills),
        "required_skills": sorted(required_skills),
        "preferred_skills": sorted(preferred_skills),
        "matched": sorted(matched),
        "matched_required": sorted(matched_required),
        "matched_preferred": sorted(matched_preferred),
        "missing": sorted(missing),
        "missing_required": sorted(missing_required),
        "resume_only": sorted(resume_only),
        "match_rate": round(match_rate, 1),
        "required_match_rate": round(req_match_rate, 1),
    }


def compute_impact_score(resume_text: str, sections: dict) -> dict:
    """
    #5 Impact Score: detect quantified achievements in experience/projects.

    Returns:
        {
            "score": 0-100,
            "hits": [...matched snippets...],
            "count": int,
            "label": str,
            "advice": str,
        }
    """
    # Focus on experience + projects sections
    target_text = ""
    for key in ["experience", "projects", "full", "header"]:
        if key in sections:
            target_text += sections[key] + "\n"

    if not target_text:
        target_text = resume_text

    hits = []
    for match in _IMPACT_RE.finditer(target_text):
        snippet = target_text[max(0, match.start() - 30): match.end() + 30].strip()
        snippet = re.sub(r"\s+", " ", snippet)
        hits.append(snippet)

    # Deduplicate approximate hits
    seen = set()
    unique_hits = []
    for h in hits:
        key = h[:40]
        if key not in seen:
            seen.add(key)
            unique_hits.append(h)

    count = len(unique_hits)

    # Score: 0 hits = 0, 1-2 = 30, 3-4 = 55, 5-7 = 75, 8+ = 90+
    if count == 0:
        score = 0
        label = "No Metrics Found"
        advice = "Add quantified achievements: 'Reduced inference latency by 40%', 'Trained model on 50K samples', 'Served 1000+ users'."
    elif count <= 2:
        score = 30
        label = "Minimal Impact Language"
        advice = f"Found {count} metric(s). Aim for 5+ quantified results across experience and projects."
    elif count <= 4:
        score = 55
        label = "Some Impact Language"
        advice = f"Found {count} metrics. Add more numbers to project outcomes and internship contributions."
    elif count <= 7:
        score = 75
        label = "Good Impact Language"
        advice = f"Found {count} quantified metrics. Strong habit — keep it consistent across all bullet points."
    else:
        score = min(95, 75 + (count - 7) * 3)
        label = "Strong Impact Language"
        advice = f"Found {count} quantified metrics. Excellent use of data-driven bullet points."

    return {
        "score": score,
        "hits": unique_hits[:8],  # top 8 for display
        "count": count,
        "label": label,
        "advice": advice,
    }


def get_suggestions(gap_result: dict, section_scores: dict, impact: dict) -> list[str]:
    """Generate actionable improvement suggestions."""
    suggestions = []
    missing_req = gap_result["missing_required"]
    missing_all = gap_result["missing"]

    # Required skills first
    if missing_req:
        top = missing_req[:5]
        suggestions.append(
            f"🚨 **Required skills missing**: {', '.join(f'`{s}`' for s in top)}. "
            f"These appear in the required section — address them first."
        )

    if missing_all and not missing_req:
        top = missing_all[:5]
        suggestions.append(
            f"Add these preferred skills where you have the knowledge: {', '.join(f'`{s}`' for s in top)}"
        )

    if len(missing_all) > 5:
        suggestions.append(
            f"You're missing {len(missing_all)} JD keywords total. Mirror the JD's exact terminology "
            f"where you genuinely have the skill — ATS systems do exact-string matching."
        )

    # Section scores
    if section_scores:
        sorted_secs = sorted(section_scores.items(), key=lambda x: x[1])
        weakest_name, weakest_score = sorted_secs[0]
        strongest_name, strongest_score = sorted_secs[-1]

        if weakest_score < 35:
            suggestions.append(
                f"Your **{weakest_name.title()}** section scores {weakest_score}% against the JD. "
                f"Rewrite it using language from the job description."
            )
        if strongest_name in ["projects", "experience"] and strongest_score > 55:
            suggestions.append(
                f"Your **{strongest_name.title()}** section ({strongest_score}%) is your strongest — "
                f"ensure it appears early and is detailed."
            )

    # Impact
    if impact["score"] < 55:
        suggestions.append(impact["advice"])

    # Overall semantic
    req_rate = gap_result["required_match_rate"]
    if req_rate < 40:
        suggestions.append(
            "Required keyword match is below 40%. The resume may get filtered by ATS before a human sees it. "
            "Tailor this resume specifically for this role."
        )
    elif req_rate > 80:
        suggestions.append(
            "Strong required-skill match (>80%). Focus now on the interview: make sure you can speak to "
            "every keyword you've listed."
        )

    return suggestions


def get_score_calibration(semantic_score: float, keyword_rate: float) -> str:
    """
    #7 Return a human-readable calibration note explaining what the scores mean.
    """
    notes = []
    notes.append(
        f"**Semantic score ({semantic_score:.0f}%)**: Measures how conceptually similar your resume "
        f"is to the JD using transformer embeddings. "
    )
    if semantic_score >= 65:
        notes.append("65%+ is strong — your experience aligns well with the role's language.")
    elif semantic_score >= 45:
        notes.append("45–65% is typical for candidates who meet most but not all requirements.")
    else:
        notes.append("Below 45% suggests a significant mismatch in role or domain. Consider if this role fits your profile.")

    notes.append(
        f"**Keyword match ({keyword_rate:.0f}%)**: Fraction of JD skills found verbatim in your resume. "
        f"ATS systems primarily use this. Most shortlisted candidates score 50–80%."
    )
    return " ".join(notes)


def categorize_skills_by_domain(skills: list[str]) -> dict:
    """Group a list of skills back into their taxonomy categories."""
    categorized = {}
    for category, category_skills in SKILL_TAXONOMY.items():
        hits = [s for s in skills if s in category_skills]
        if hits:
            categorized[category] = hits
    return categorized