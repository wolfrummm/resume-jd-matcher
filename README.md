# 🎯 Resume-JD Matcher

Semantic resume-to-job-description matcher using HuggingFace sentence-transformers, cosine similarity, and skill gap analysis. Built as part of a placement prep project pipeline targeting SDE, AIML, and DS roles.

## Features

**Semantic Similarity** — `all-MiniLM-L6-v2` embeddings with chunked mean-pool encoding (no lossy truncation), scored against the full JD via cosine similarity.

**Skill Gap Analysis** — 402 curated skills across 10 categories including an `india_specific` category (DSA, OOPS, DBMS, LLD, HLD, system design). Alias expansion handles abbreviations like `K8s`, `NLP`, `LLD` before matching. Missing skills are split into Required vs Preferred based on JD section parsing.

**Section-Level Scoring** — resume is split into named sections (Skills, Experience, Projects, Education) using regex + ALL-CAPS/Title Case heuristics. Each section is independently embedded and scored against the JD.

**Impact Score** — scans experience and projects for quantified achievements (numbers, percentages, scale words, metric verbs). Returns a 0–100 score with targeted advice.

**Multi-Resume Comparison** — paste one JD, upload up to 3 resumes, get a side-by-side grouped chart and weighted recommendation.

**Score Calibration** — plain-English note explaining what the semantic score and keyword rate mean, with typical ranges for shortlisted candidates.

**Scanned PDF Detection** — detects image-based PDFs by average chars/page and redirects the user to paste text instead of silently returning empty results.

**Resume Bullets Generator** — auto-generates track-specific resume bullet points with real numbers from the analysis run (matched skill count, impact count, semantic score).

## Tech Stack

| Layer | Tool |
|---|---|
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Similarity | `scikit-learn` cosine_similarity |
| PDF Parsing | `PyMuPDF` (fitz) |
| Visualization | `plotly` |
| UI | `Streamlit` |

## Local Setup

```bash
git clone https://github.com/wolfrummm/resume-jd-matcher
cd resume-jd-matcher
pip install -r requirements.txt
streamlit run app.py
```

First run downloads `all-MiniLM-L6-v2` (~90MB) to `~/.cache/huggingface/`. Subsequent runs load from cache instantly.

## Deploy to Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set `app.py` as entry point
4. Deploy — cold start ~30s (model download), warm requests <2s

## Project Structure

```
resume-jd-matcher/
├── app.py                       # Streamlit UI — single + compare modes
├── matcher/
│   ├── embedder.py              # Chunked mean-pool embedding + cosine similarity
│   ├── extractor.py             # PDF parsing, section splitting, input validation
│   ├── analyzer.py              # Skill gap, JD section parsing, Impact Score, calibration
│   └── utils.py                 # Score labels, resume bullet generator
├── data/
│   └── skill_keywords.json      # 402 skills across 10 categories
├── .streamlit/
│   └── config.toml              # Dark theme config
└── requirements.txt
```

## Resume Bullet Points

**AIML / ML Engineer:**
> Built a Resume-JD Matcher using HuggingFace `sentence-transformers` (`all-MiniLM-L6-v2`) with chunked mean-pool embeddings and cosine similarity; implemented JD required/preferred section parsing, Impact Score detection across quantified metrics, and skill alias expansion across 402 tech skills in 10 categories.

**Data Science / Analyst:**
> Developed a semantic text-similarity pipeline (resume vs. JD) with transformer embeddings; built skill taxonomy extraction (402 skills, 10 categories including DSA/OOPS/system design), JD section classification for required vs. preferred weighting, and an Impact Score metric detecting quantified achievements per resume section.

**Full Stack / SDE:**
> Engineered a modular Streamlit app with PDF parsing (PyMuPDF), scanned-PDF detection, chunked NLP inference (HuggingFace sentence-transformers), multi-resume comparison mode, and session-state result caching; deployed on Streamlit Cloud with dark theme and <2s warm inference.