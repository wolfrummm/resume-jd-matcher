"""
extractor.py — PDF parsing and resume section splitting (improved)

Fixes:
  - #1  Robust section splitter: regex + ALL-CAPS heuristic + short-line heuristic
  - #3  Min-length guard with clear user messages
  - #11 Scanned/image PDF detection
"""

import re
import fitz  # PyMuPDF


MIN_RESUME_CHARS = 150
MIN_JD_CHARS = 100

# Section header patterns (case-insensitive)
SECTION_PATTERNS = {
    "summary":        r"\b(summary|objective|profile|about me|career objective|professional summary)\b",
    "skills":         r"\b(skills|technical skills|core competencies|technologies|tech stack|key skills|tools)\b",
    "experience":     r"\b(experience|work experience|employment|professional experience|internship|internships|work history)\b",
    "projects":       r"\b(projects|personal projects|academic projects|key projects|portfolio)\b",
    "education":      r"\b(education|academic background|qualifications|degrees|academics)\b",
    "certifications": r"\b(certifications|certificates|courses|training|achievements|awards)\b",
    "publications":   r"\b(publications|research|papers|conferences)\b",
}

# Heuristic: a line is likely a section header if it's short, has no sentence-ending
# punctuation, and is ALL CAPS or Title Case with few words
def _looks_like_header(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 55:
        return False
    if stripped.endswith((".", ",", ";", ")")):
        return False
    word_count = len(stripped.split())
    if word_count > 5:
        return False
    # ALL CAPS check (e.g. "WORK EXPERIENCE", "SKILLS")
    if stripped.isupper() and word_count <= 4:
        return True
    # Title Case with few words (e.g. "Work Experience", "Key Projects")
    if stripped.istitle() and word_count <= 3:
        return True
    return False


def _classify_header(line: str) -> str | None:
    """Return section key if the line matches a known section, else None."""
    stripped = line.strip()
    for section_name, pattern in SECTION_PATTERNS.items():
        if re.search(pattern, stripped, re.IGNORECASE):
            return section_name
    return None


def extract_text_from_pdf(uploaded_file) -> tuple[str, bool]:
    """
    Extract plain text from a Streamlit UploadedFile (PDF).
    Returns (text, is_scanned).
    Raises ValueError on parse failure.
    """
    try:
        pdf_bytes = uploaded_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text_parts = []
        total_chars = 0

        for page in doc:
            page_text = page.get_text("text")
            text_parts.append(page_text)
            total_chars += len(page_text.strip())

        doc.close()
        full_text = "\n".join(text_parts).strip()

        # Scanned PDF heuristic: very little text extracted despite pages existing
        page_count = len(text_parts)
        avg_chars_per_page = total_chars / max(page_count, 1)
        is_scanned = avg_chars_per_page < 80 and page_count >= 1

        return full_text, is_scanned

    except Exception as e:
        raise ValueError(f"Failed to parse PDF: {e}")


def split_into_sections(text: str) -> dict:
    """
    Split resume text into named sections using:
      1. Known-pattern matching
      2. ALL CAPS / Title Case heuristic fallback

    Returns dict: { section_name: section_text }
    Falls back to {"full": text} if fewer than 2 sections detected.
    """
    lines = text.split("\n")
    sections = {}
    current_section = "header"
    buffer = []

    for line in lines:
        stripped = line.strip()

        # Try known-pattern match first
        matched_section = None
        if stripped and len(stripped) < 60:
            matched_section = _classify_header(stripped)

        # Fall back to typographic heuristic
        if matched_section is None and _looks_like_header(stripped):
            # Use the raw line as section name (lowercase, underscored)
            candidate = re.sub(r"[^a-z0-9 ]", "", stripped.lower()).strip().replace(" ", "_")
            if candidate:
                matched_section = candidate

        if matched_section and matched_section != current_section:
            if buffer:
                content = "\n".join(buffer).strip()
                if content:
                    sections[current_section] = content
            current_section = matched_section
            buffer = []
        else:
            buffer.append(line)

    # Flush last buffer
    if buffer:
        content = "\n".join(buffer).strip()
        if content:
            sections[current_section] = content

    # Fallback if barely any sections found
    if len([k for k in sections if sections[k]]) <= 1:
        sections = {"full": text}

    return sections


def clean_text(text: str) -> str:
    """Normalize whitespace and remove non-printable characters."""
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def validate_inputs(resume_text: str, jd_text: str) -> list[str]:
    """
    Validate both inputs and return a list of error messages.
    Empty list = all good.
    """
    errors = []
    if len(resume_text.strip()) < MIN_RESUME_CHARS:
        errors.append(
            f"Resume text is too short ({len(resume_text.strip())} chars). "
            f"Paste at least {MIN_RESUME_CHARS} characters for meaningful analysis."
        )
    if len(jd_text.strip()) < MIN_JD_CHARS:
        errors.append(
            f"Job description is too short ({len(jd_text.strip())} chars). "
            f"Paste the full JD for accurate results."
        )
    return errors