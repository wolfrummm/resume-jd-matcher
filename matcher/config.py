"""Configuration for the Resume-JD Matcher."""

# Embedding configuration
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_CHAR_LIMIT = 1800

# Semantic skill matching
SEMANTIC_SKILL_THRESHOLD = 0.62

# Section-level scoring weights
SECTION_WEIGHTS = {
    "skills": 0.35,
    "experience": 0.35,
    "projects": 0.20,
    "education": 0.05,
    "summary": 0.05,
}

# Input validation
MIN_RESUME_TEXT_LENGTH = 150
