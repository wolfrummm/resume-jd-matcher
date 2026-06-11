"""
app.py — Resume-JD Matcher (v2)

Improvements implemented:
  #1  Robust section splitter (extractor.py)
  #2  Expanded taxonomy 300+ skills (skill_keywords.json)
  #3  Min-length input guard
  #4  Chunked mean-pool embedding (embedder.py)
  #5  Impact Score tab
  #6  JD required vs preferred parsing (analyzer.py)
  #7  Score calibration tooltip
  #8  Multi-resume comparison mode
  #9  Session state — results persist across reruns
  #10 Dark theme via .streamlit/config.toml
  #11 Scanned PDF detection
  #12 Resume bullets with real metrics
"""

import streamlit as st
import plotly.graph_objects as go

from matcher import (
    load_model,
    extract_text_from_pdf,
    split_into_sections,
    clean_text,
    validate_inputs,
    compute_similarity,
    compute_section_scores,
    analyze_skill_gap,
    get_suggestions,
    categorize_skills_by_domain,
    compute_impact_score,
    get_score_calibration,
    score_to_label,
    impact_score_to_color,
    format_section_name,
    generate_resume_bullets,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Resume-JD Matcher",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.score-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border-radius: 16px; padding: 1.6rem; text-align: center;
    border: 1px solid #334155; height: 100%;
}
.score-number { font-size: 3.2rem; font-weight: 700; line-height: 1; }
.score-label  { font-size: 0.95rem; margin-top: 0.3rem; font-weight: 600; letter-spacing: 0.04em; }
.score-sub    { color: #64748b; font-size: 0.75rem; margin-top: 0.4rem; }

.compare-card {
    background: #1e293b; border-radius: 12px; padding: 1.2rem;
    border: 1px solid #334155; margin-bottom: 0.8rem;
}
.compare-winner {
    border: 2px solid #22c55e !important;
    background: linear-gradient(135deg, #0f2b1b 0%, #1e293b 100%) !important;
}
.compare-label { font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; }
.compare-val   { font-size: 1.5rem; font-weight: 700; }

.section-header {
    font-size: 0.85rem; font-weight: 600; color: #94a3b8;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin: 1.2rem 0 0.6rem; border-bottom: 1px solid #1e293b; padding-bottom: 0.3rem;
}
.pill {
    display: inline-block; padding: 3px 10px; border-radius: 999px;
    font-size: 0.76rem; font-weight: 500; margin: 3px;
}
.pill-green  { background: #dcfce7; color: #166534; }
.pill-red    { background: #fee2e2; color: #991b1b; }
.pill-blue   { background: #dbeafe; color: #1e40af; }
.pill-amber  { background: #fef3c7; color: #92400e; }

.suggestion-box {
    background: #0f172a; border-left: 3px solid #6366f1;
    border-radius: 0 8px 8px 0; padding: 0.75rem 1rem;
    margin: 0.4rem 0; font-size: 0.88rem; color: #e2e8f0;
}
.impact-hit {
    background: #1e293b; border-radius: 6px; padding: 0.4rem 0.7rem;
    margin: 0.3rem 0; font-size: 0.82rem; color: #cbd5e1;
    border-left: 2px solid #f59e0b;
}
.calibration-box {
    background: #0f172a; border-radius: 8px; padding: 0.9rem 1.1rem;
    margin: 0.6rem 0; font-size: 0.83rem; color: #94a3b8;
    border: 1px solid #1e293b;
}
.bullet-box {
    background: #0f172a; border-radius: 8px; padding: 0.8rem 1rem;
    margin: 0.5rem 0; font-size: 0.85rem; color: #e2e8f0;
    border-left: 3px solid #8b5cf6; font-family: monospace;
}
.req-badge {
    background: #fee2e2; color: #991b1b; border-radius: 4px;
    padding: 1px 6px; font-size: 0.7rem; font-weight: 700;
    margin-left: 4px; vertical-align: middle;
}
.pref-badge {
    background: #fef3c7; color: #92400e; border-radius: 4px;
    padding: 1px 6px; font-size: 0.7rem; font-weight: 700;
    margin-left: 4px; vertical-align: middle;
}
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white; border: none; border-radius: 8px;
    padding: 0.6rem 2rem; font-weight: 600; font-size: 1rem;
    width: 100%; transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }
</style>
""", unsafe_allow_html=True)


# ── Session state init ─────────────────────────────────────────────────────────
# #9 Persist results across reruns so clicking tabs doesn't re-trigger analysis
if "results" not in st.session_state:
    st.session_state.results = None
if "mode" not in st.session_state:
    st.session_state.mode = "single"


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("## 🎯 Resume-JD Matcher")
st.markdown("Semantic similarity · Skill gap analysis · Impact scoring — powered by HuggingFace sentence-transformers")
st.divider()

# ── Model ──────────────────────────────────────────────────────────────────────
with st.spinner("Loading model (first run ~30s, then cached)..."):
    model = load_model()

# ── Mode toggle ───────────────────────────────────────────────────────────────
mode_col, _ = st.columns([2, 5])
with mode_col:
    mode = st.radio(
        "Mode",
        ["Single Resume", "Compare Resumes"],
        horizontal=True,
        label_visibility="collapsed",
    )
st.session_state.mode = mode

# ══════════════════════════════════════════════════════════════════════════════
# SINGLE MODE
# ══════════════════════════════════════════════════════════════════════════════
if mode == "Single Resume":

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.markdown("#### 📄 Resume")
        rt1, rt2 = st.tabs(["Upload PDF", "Paste Text"])
        with rt1:
            uploaded_pdf = st.file_uploader("PDF", type=["pdf"], label_visibility="collapsed")
        with rt2:
            pasted_resume = st.text_area("Text", height=260, placeholder="Paste resume here...", label_visibility="collapsed")

    with col_right:
        st.markdown("#### 💼 Job Description")
        jd_input = st.text_area("JD", height=300, placeholder="Paste job description here...", label_visibility="collapsed")

    st.markdown("")
    _, btn_col, _ = st.columns([2, 1, 2])
    with btn_col:
        analyze_btn = st.button("Analyze Match →")

    if analyze_btn:
        # Resolve resume text
        resume_text = ""
        if uploaded_pdf is not None:
            try:
                raw_text, is_scanned = extract_text_from_pdf(uploaded_pdf)
                # #11 Scanned PDF warning
                if is_scanned:
                    st.warning(
                        "⚠️ This PDF appears to be image-based (scanned). "
                        "Very little text was extracted. Please paste your resume text in the 'Paste Text' tab instead."
                    )
                    st.stop()
                resume_text = clean_text(raw_text)
            except ValueError as e:
                st.error(str(e))
                st.stop()
        elif pasted_resume.strip():
            resume_text = clean_text(pasted_resume)

        jd_text = clean_text(jd_input)

        # #3 Validate inputs
        errors = validate_inputs(resume_text, jd_text)
        if errors:
            for err in errors:
                st.warning(err)
            st.stop()

        with st.spinner("Analyzing..."):
            sections = split_into_sections(resume_text)
            section_scores = compute_section_scores(sections, jd_text, model)
            overall_score = round(compute_similarity(resume_text, jd_text, model) * 100, 1)
            gap = analyze_skill_gap(resume_text, jd_text)
            impact = compute_impact_score(resume_text, sections)
            suggestions = get_suggestions(gap, section_scores, impact)
            calibration = get_score_calibration(overall_score, gap["match_rate"])
            bullets = generate_resume_bullets(gap["matched"], impact["count"], overall_score, gap["match_rate"])

        st.session_state.results = {
            "type": "single",
            "overall_score": overall_score,
            "gap": gap,
            "section_scores": section_scores,
            "sections": sections,
            "impact": impact,
            "suggestions": suggestions,
            "calibration": calibration,
            "bullets": bullets,
        }

    # ── Render results ─────────────────────────────────────────────────────────
    res = st.session_state.results
    if res and res.get("type") == "single":
        overall_score = res["overall_score"]
        gap = res["gap"]
        section_scores = res["section_scores"]
        sections = res["sections"]
        impact = res["impact"]

        st.divider()
        st.markdown("## Results")

        # Score cards row
        label, sem_color = score_to_label(overall_score)
        imp_color = impact_score_to_color(impact["score"])
        req_color = score_to_label(res["gap"]["required_match_rate"])[1]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="score-card">
                <div class="score-number" style="color:{sem_color}">{overall_score}%</div>
                <div class="score-label" style="color:{sem_color}">{label}</div>
                <div class="score-sub">Semantic Similarity</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="score-card">
                <div class="score-number" style="color:{req_color}">{gap['required_match_rate']:.0f}%</div>
                <div class="score-label" style="color:{req_color}">Required Skills</div>
                <div class="score-sub">{len(gap['matched_required'])} / {len(gap['required_skills'])} matched</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="score-card">
                <div class="score-number" style="color:{imp_color}">{impact['score']}</div>
                <div class="score-label" style="color:{imp_color}">{impact['label']}</div>
                <div class="score-sub">Impact Score</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="score-card">
                <div class="score-number" style="color:#94a3b8">{gap['match_rate']:.0f}%</div>
                <div class="score-label" style="color:#94a3b8">All Keywords</div>
                <div class="score-sub">{len(gap['matched'])} / {len(gap['jd_skills'])} matched</div>
            </div>""", unsafe_allow_html=True)

        # #7 Calibration
        st.markdown("")
        st.markdown(f'<div class="calibration-box">ℹ️ {res["calibration"]}</div>', unsafe_allow_html=True)
        st.markdown("")

        # Tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Section Breakdown", "✅ Skill Gap", "⚡ Impact Score",
            "💡 Suggestions", "📝 Resume Bullets", "🔍 Raw Sections"
        ])

        # ── Tab 1: Section Breakdown ───────────────────────────────────────────
        with tab1:
            if section_scores:
                sorted_s = sorted(section_scores.items(), key=lambda x: x[1], reverse=True)
                names  = [format_section_name(s[0]) for s in sorted_s]
                values = [s[1] for s in sorted_s]
                colors = ["#22c55e" if v >= 70 else "#f59e0b" if v >= 45 else "#ef4444" for v in values]

                fig = go.Figure(go.Bar(
                    x=values, y=names, orientation="h",
                    marker_color=colors,
                    text=[f"{v}%" for v in values], textposition="outside",
                ))
                fig.update_layout(
                    xaxis=dict(range=[0, 115], showgrid=False, zeroline=False, visible=False),
                    yaxis=dict(autorange="reversed"),
                    margin=dict(l=10, r=60, t=20, b=10),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    height=max(200, len(names) * 58), font=dict(size=13),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Each section is independently embedded and scored against the full JD.")
            else:
                st.info("No distinct sections detected. Add clear headers (SKILLS, EXPERIENCE, PROJECTS, EDUCATION) to your resume for section-level analysis.")

        # ── Tab 2: Skill Gap ──────────────────────────────────────────────────
        with tab2:
            if gap["jd_skills"]:
                # #6 Required vs preferred labelling
                required_set = set(gap["required_skills"])
                preferred_set = set(gap["preferred_skills"])

                def skill_badge(skill):
                    if skill in required_set:
                        return f'<span class="req-badge">REQ</span>'
                    if skill in preferred_set:
                        return f'<span class="pref-badge">PREF</span>'
                    return ""

                gcol1, gcol2 = st.columns(2)
                with gcol1:
                    st.markdown('<div class="section-header">✅ Matched Skills</div>', unsafe_allow_html=True)
                    if gap["matched"]:
                        cat = categorize_skills_by_domain(gap["matched"])
                        for c_name, skills in cat.items():
                            st.markdown(f"**{c_name.replace('_', ' ').title()}**")
                            pills = " ".join([
                                f'<span class="pill pill-green">{s}{skill_badge(s)}</span>'
                                for s in skills
                            ])
                            st.markdown(pills, unsafe_allow_html=True)
                    else:
                        st.caption("No matched skills found.")

                with gcol2:
                    st.markdown('<div class="section-header">❌ Missing Skills</div>', unsafe_allow_html=True)
                    if gap["missing"]:
                        # Sort: required first
                        req_missing = [s for s in gap["missing"] if s in required_set]
                        pref_missing = [s for s in gap["missing"] if s in preferred_set and s not in required_set]
                        other_missing = [s for s in gap["missing"] if s not in required_set and s not in preferred_set]

                        if req_missing:
                            st.markdown("**🚨 Required**")
                            pills = " ".join([f'<span class="pill pill-red">{s}</span>' for s in req_missing])
                            st.markdown(pills, unsafe_allow_html=True)
                        if pref_missing:
                            st.markdown("**⚠️ Preferred**")
                            pills = " ".join([f'<span class="pill pill-amber">{s}</span>' for s in pref_missing])
                            st.markdown(pills, unsafe_allow_html=True)
                        if other_missing:
                            st.markdown("**Other**")
                            pills = " ".join([f'<span class="pill pill-red">{s}</span>' for s in other_missing])
                            st.markdown(pills, unsafe_allow_html=True)
                    else:
                        st.success("No missing skills — great alignment!")

                if gap["resume_only"]:
                    st.markdown('<div class="section-header">🔵 In Resume, Not in JD</div>', unsafe_allow_html=True)
                    pills = " ".join([f'<span class="pill pill-blue">{s}</span>' for s in gap["resume_only"][:25]])
                    st.markdown(pills, unsafe_allow_html=True)
            else:
                st.info("No recognized skills found in the JD. Works best with technical JDs.")

        # ── Tab 3: Impact Score ───────────────────────────────────────────────
        with tab3:
            imp = res["impact"]
            imp_col, _ = st.columns([1, 2])
            with imp_col:
                color = impact_score_to_color(imp["score"])
                st.markdown(f"""
                <div class="score-card">
                    <div class="score-number" style="color:{color}">{imp['score']}</div>
                    <div class="score-label" style="color:{color}">{imp['label']}</div>
                    <div class="score-sub">{imp['count']} quantified metric(s) found</div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f'<div class="suggestion-box">💡 {imp["advice"]}</div>', unsafe_allow_html=True)

            if imp["hits"]:
                st.markdown('<div class="section-header">Detected Metrics</div>', unsafe_allow_html=True)
                for hit in imp["hits"]:
                    st.markdown(f'<div class="impact-hit">…{hit}…</div>', unsafe_allow_html=True)
            else:
                st.markdown("**No quantified metrics detected.**")
                st.markdown("""
Add bullet points like:
- *"Trained LSTM on 50,000 ECG samples, achieving 94% F1 score"*
- *"Reduced API latency by 38% via Redis caching"*
- *"Deployed to 1,200+ users across 3 colleges"*
                """)

        # ── Tab 4: Suggestions ────────────────────────────────────────────────
        with tab4:
            if res["suggestions"]:
                for s in res["suggestions"]:
                    st.markdown(f'<div class="suggestion-box">{s}</div>', unsafe_allow_html=True)
            else:
                st.success("Your resume looks well-aligned with this JD!")

        # ── Tab 5: Resume Bullets ─────────────────────────────────────────────
        with tab5:
            st.markdown("Copy these into your resume — numbers are derived from this analysis run.")
            for track, bullet in res["bullets"].items():
                st.markdown(f"**{track}**")
                st.markdown(f'<div class="bullet-box">{bullet}</div>', unsafe_allow_html=True)
                st.markdown("")

        # ── Tab 6: Raw Sections ───────────────────────────────────────────────
        with tab6:
            if sections:
                for sname, scontent in sections.items():
                    with st.expander(f"{format_section_name(sname)}  ({len(scontent)} chars)"):
                        st.text(scontent[:1000] + ("…" if len(scontent) > 1000 else ""))
            else:
                st.info("No sections to display.")


# ══════════════════════════════════════════════════════════════════════════════
# COMPARE MODE  (#8)
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("#### 💼 Job Description")
    jd_input_cmp = st.text_area("JD (compare)", height=180, placeholder="Paste job description here...", label_visibility="collapsed")

    st.markdown("#### 📄 Resumes to Compare")
    st.caption("Upload up to 3 resume PDFs or paste text. Leave unused slots empty.")

    resume_inputs = []
    cols = st.columns(3)
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"**Resume {i+1}**")
            label_input = st.text_input(f"Label {i+1}", value=f"Resume {i+1}", key=f"label_{i}")
            pdf_input = st.file_uploader(f"PDF {i+1}", type=["pdf"], key=f"pdf_{i}", label_visibility="collapsed")
            text_input = st.text_area(f"Text {i+1}", height=160, placeholder="Or paste text...", key=f"text_{i}", label_visibility="collapsed")
            resume_inputs.append({"label": label_input, "pdf": pdf_input, "text": text_input})

    st.markdown("")
    _, btn_col_cmp, _ = st.columns([2, 1, 2])
    with btn_col_cmp:
        compare_btn = st.button("Compare Resumes →")

    if compare_btn:
        jd_text_cmp = clean_text(jd_input_cmp)
        if not jd_text_cmp or len(jd_text_cmp) < 100:
            st.warning("Please paste a job description (at least 100 characters).")
            st.stop()

        compare_results = []
        for inp in resume_inputs:
            rtext = ""
            if inp["pdf"]:
                try:
                    raw, is_scanned = extract_text_from_pdf(inp["pdf"])
                    if is_scanned:
                        st.warning(f"⚠️ '{inp['label']}' appears to be a scanned PDF — skipping.")
                        continue
                    rtext = clean_text(raw)
                except ValueError as e:
                    st.error(f"Error reading '{inp['label']}': {e}")
                    continue
            elif inp["text"].strip():
                rtext = clean_text(inp["text"])

            if not rtext or len(rtext) < 150:
                continue

            with st.spinner(f"Analyzing {inp['label']}..."):
                sections = split_into_sections(rtext)
                overall = round(compute_similarity(rtext, jd_text_cmp, model) * 100, 1)
                gap = analyze_skill_gap(rtext, jd_text_cmp)
                impact = compute_impact_score(rtext, sections)
                compare_results.append({
                    "label": inp["label"],
                    "overall": overall,
                    "req_rate": gap["required_match_rate"],
                    "kw_rate": gap["match_rate"],
                    "impact": impact["score"],
                    "impact_label": impact["label"],
                    "matched": len(gap["matched"]),
                    "missing_req": gap["missing_required"],
                    "gap": gap,
                })

        if not compare_results:
            st.warning("No valid resumes to compare. Please add at least one resume.")
            st.stop()

        st.session_state.results = {"type": "compare", "data": compare_results}

    # Render compare results
    res = st.session_state.results
    if res and res.get("type") == "compare":
        data = res["data"]
        if not data:
            st.info("No results to display.")
        else:
            st.divider()
            st.markdown("## Comparison Results")

            # Find winner per metric
            best_overall = max(data, key=lambda x: x["overall"])["label"]
            best_req     = max(data, key=lambda x: x["req_rate"])["label"]
            best_impact  = max(data, key=lambda x: x["impact"])["label"]

            # Summary table
            fig = go.Figure(data=[
                go.Bar(name="Semantic Match %",   x=[d["label"] for d in data], y=[d["overall"]  for d in data], marker_color="#6366f1"),
                go.Bar(name="Required Skills %",  x=[d["label"] for d in data], y=[d["req_rate"] for d in data], marker_color="#22c55e"),
                go.Bar(name="Impact Score",        x=[d["label"] for d in data], y=[d["impact"]   for d in data], marker_color="#f59e0b"),
            ])
            fig.update_layout(
                barmode="group",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=30, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                yaxis=dict(range=[0, 110]),
                height=320,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Per-card breakdown
            card_cols = st.columns(len(data))
            for i, (d, col) in enumerate(zip(data, card_cols)):
                is_winner = d["overall"] == max(x["overall"] for x in data)
                winner_class = "compare-winner" if is_winner else ""
                with col:
                    badge = "🏆 " if is_winner else ""
                    st.markdown(f"""
                    <div class="compare-card {winner_class}">
                        <div style="font-weight:700;font-size:1rem;margin-bottom:0.8rem">{badge}{d['label']}</div>
                        <div class="compare-label">Semantic</div>
                        <div class="compare-val" style="color:{score_to_label(d['overall'])[1]}">{d['overall']}%</div>
                        <div style="margin-top:0.6rem">
                        <div class="compare-label">Required Skills</div>
                        <div class="compare-val" style="color:{score_to_label(d['req_rate'])[1]}">{d['req_rate']:.0f}%</div>
                        </div>
                        <div style="margin-top:0.6rem">
                        <div class="compare-label">Impact Score</div>
                        <div class="compare-val" style="color:{impact_score_to_color(d['impact'])}">{d['impact']}</div>
                        </div>
                        <div style="margin-top:0.6rem">
                        <div class="compare-label">Keywords Matched</div>
                        <div style="font-size:1rem;font-weight:600">{d['matched']}</div>
                        </div>
                    </div>""", unsafe_allow_html=True)

                    if d["missing_req"]:
                        st.caption(f"Missing required: {', '.join(d['missing_req'][:4])}" + ("..." if len(d["missing_req"]) > 4 else ""))

            # Recommendation
            st.markdown("")
            winner = max(data, key=lambda x: x["overall"] * 0.4 + x["req_rate"] * 0.4 + x["impact"] * 0.2)
            st.success(f"**Recommendation: Use '{winner['label']}'** — highest combined score across semantic match, required skill coverage, and impact language.")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Resume-JD Matcher v2 · HuggingFace sentence-transformers · all-MiniLM-L6-v2")