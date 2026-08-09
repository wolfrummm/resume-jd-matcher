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

_BOLD_FLAG = 1 << 4  # PyMuPDF bold bit in a span's "flags"


def _looks_like_header(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 55:
        return False
    if stripped.endswith((".", ",", ";", ")")):
        return False
    if any(ch.isdigit() for ch in stripped):
        return False
    word_count = len(stripped.split())
    if word_count > 5:
        return False
    if stripped.isupper() and word_count <= 4:
        return True
    if stripped.istitle() and word_count <= 3:
        return True
    return False


def _classify_header(line: str) -> str | None:
    stripped = line.strip()
    for section_name, pattern in SECTION_PATTERNS.items():
        if re.search(pattern, stripped, re.IGNORECASE):
            return section_name
    return None


def _extract_page_lines_with_style(page) -> list[dict]:
    """Extract lines with font size/bold/position metadata (not just plain text)."""
    raw = page.get_text("dict")
    lines_info = []
    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            text = "".join(s["text"] for s in spans).strip()
            if not text:
                continue
            max_size = max(s["size"] for s in spans)
            is_bold = any((s["flags"] & _BOLD_FLAG) or "bold" in s["font"].lower() for s in spans)
            x0 = min(s["bbox"][0] for s in spans)
            y0 = min(s["bbox"][1] for s in spans)
            y1 = max(s["bbox"][3] for s in spans)
            lines_info.append({"text": text, "size": max_size, "bold": is_bold, "x0": x0, "y0": y0, "y1": y1})
    return lines_info


def _reading_order_lines(lines_info: list[dict]) -> list[dict]:
    """Group lines into rows by vertical overlap, sort left-to-right within each row."""
    lines_info = sorted(lines_info, key=lambda l: l["y0"])
    rows = []
    for l in lines_info:
        placed = False
        for row in rows:
            row_y0 = min(r["y0"] for r in row)
            row_y1 = max(r["y1"] for r in row)
            height = row_y1 - row_y0
            overlap = min(l["y1"], row_y1) - max(l["y0"], row_y0)
            if height > 0 and overlap / height > 0.4:
                row.append(l)
                placed = True
                break
        if not placed:
            rows.append([l])
    rows.sort(key=lambda row: min(r["y0"] for r in row))
    ordered = []
    for row in rows:
        row.sort(key=lambda r: r["x0"])
        ordered.extend(row)
    return ordered


def extract_text_from_pdf(uploaded_file) -> tuple[str, bool, set]:
    """
    Extract plain text, scanned-PDF flag, and a set of lines that are
    stylistically real headers (larger font and/or bold) — used to avoid
    mistaking short address/date lines for section headers.
    Returns (text, is_scanned, heading_lines).
    """
    try:
        pdf_bytes = uploaded_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        all_lines = []
        total_chars = 0
        page_count = 0

        for page in doc:
            page_count += 1
            lines_info = _extract_page_lines_with_style(page)
            ordered = _reading_order_lines(lines_info)
            all_lines.extend(ordered)
            total_chars += sum(len(l["text"]) for l in ordered)

        doc.close()

        if not all_lines:
            return "", True, set()

        sizes = [round(l["size"], 1) for l in all_lines]
        body_size = max(set(sizes), key=sizes.count)  # most common font size = body text baseline

        heading_lines = set()
        for l in all_lines:
            is_larger = l["size"] > body_size * 1.15
            if (is_larger or l["bold"]) and len(l["text"]) <= 45 and len(l["text"].split()) <= 6:
                heading_lines.add(l["text"].strip())

        # If we found fewer than 2 stylistically-distinct lines, this PDF
        # likely doesn't differentiate headers visually — font signal is
        # unreliable here, so fall back to shape-only heuristics instead.
        if len(heading_lines) < 2:
            heading_lines = None

        full_text = "\n".join(l["text"] for l in all_lines).strip()
        avg_chars_per_page = total_chars / max(page_count, 1)
        is_scanned = avg_chars_per_page < 80 and page_count >= 1

        return full_text, is_scanned, heading_lines

    except Exception as e:
        raise ValueError(f"Failed to parse PDF: {e}")


def split_into_sections(text: str, heading_lines: set | None = None) -> dict:
    """
    Split resume text into named sections using:
      1. Known-pattern matching
      2. Font-based heading detection (if available, from PDF) combined with
         shape heuristics — much stricter, avoids false positives like an
         address line ("Patiala Punjab") being mistaken for a new section.
      3. Shape heuristic alone as a fallback when no font info exists
         (e.g. pasted text, or a PDF with no visual header distinction).
    """
    lines = text.split("\n")
    sections = {}
    current_section = "header"
    buffer = []
    seen_known_section = False

    for line in lines:
        stripped = line.strip()

        matched_section = None
        if stripped and len(stripped) < 60:
            matched_section = _classify_header(stripped)

        if matched_section is not None:
            seen_known_section = True
        elif seen_known_section:
            is_shape_heading = _looks_like_header(stripped)
            if heading_lines:
                # Require BOTH text-shape AND font-style evidence
                is_style_heading = stripped in heading_lines
                if is_shape_heading and is_style_heading:
                    candidate = re.sub(r"[^a-z0-9 ]", "", stripped.lower()).strip().replace(" ", "_")
                    if candidate:
                        matched_section = candidate
            elif is_shape_heading:
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

    if buffer:
        content = "\n".join(buffer).strip()
        if content:
            sections[current_section] = content

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