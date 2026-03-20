"""
app.py – Streamlit dashboard for CreditLens.

A dark-themed professional UI that calls the FastAPI backend.
All ML processing happens over HTTP via requests.
"""

import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="CreditLens Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS for Dark Theme Polish ──────────────────────────────────────────
st.markdown("""
<style>
/* Adjust metric card styling */
div[data-testid="stMetricValue"] {
    font-size: 1.8rem;
    font-weight: bold;
}
/* Subtitle */
.subtitle {
    font-size: 1.2rem;
    color: #a0aab5;
    margin-top: -10px;
    margin-bottom: 30px;
}
/* Section headers */
.section-header {
    border-bottom: 1px solid #334;
    padding-bottom: 10px;
    margin-top: 30px;
    margin-bottom: 20px;
    color: #e0e5eb;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("CreditLens")
st.markdown('<p class="subtitle">SME Credit Risk Assessment Intelligence</p>', unsafe_allow_html=True)

# ── Sidebar Form ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("New Assessment")
    
    with st.form("assessment_form"):
        b_name = st.text_input("Business Name", value="GlobeTech Logistics")
        b_desc = st.text_area(
            "Business Description",
            value="Mid-size logistics and trucking firm in Mumbai, 12 years in operation, 85 employees.",
            height=120
        )
        b_fin = st.text_area(
            "Financial Information",
            value="Annual revenue ₹15 crore, consistent margins but requested a $50,000 loan for fleet expansion.",
            height=120
        )
        fetch_news = st.checkbox("Fetch live news", value=True)
        
        submit_btn = st.form_submit_button("Analyze Credit Risk", use_container_width=True)


# ── Helper for color coding ──────────────────────────────────────────────────
def get_rating_color(rating: str) -> str:
    r = rating.upper()
    if r == "LOW": return "green"
    if r == "MEDIUM": return "orange"
    if r in ("HIGH", "REJECT", "UNACCEPTABLE"): return "red"
    return "normal"


# ── Main Content Area (Executes on Submit) ──────────────────────────────────
if submit_btn:
    if not b_name or not b_desc or not b_fin:
        st.warning("Please fill in all input fields.")
    else:
        with st.spinner(f"Analyzing credit profile for {b_name} via AI models..."):
            try:
                payload = {
                    "business_name": b_name,
                    "description": b_desc,
                    "financial_text": b_fin,
                    "fetch_news": fetch_news
                }
                resp = requests.post(f"{API_URL}/analyze", json=payload, timeout=60)
                resp.raise_for_status()
                result = resp.json()
                
                # Extract components
                cb = result.get("credit_brief", {})
                
                # ── ROW 1: Metrics ──────────────────────────────────────────
                st.markdown('<h3 class="section-header">Credit Decision</h3>', unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                
                rating = cb.get("risk_rating", "Unknown").upper()
                score = cb.get("risk_score", 50)
                rec = cb.get("recommendation", "Unknown").upper()
                
                m1.metric("Risk Rating", rating)
                m2.metric("Risk Score", f"{score}/100")
                m3.metric("Recommendation", rec)
                
                st.progress(score / 100, text=f"Risk Score Progression ({score})")
                
                # ── ROW 2: Strengths & Risks ────────────────────────────────
                col_left, col_right = st.columns(2)
                with col_left:
                    st.subheader("Key Mitigants / Strengths")
                    for m in cb.get("mitigants", []):
                        st.markdown(f"✅ {m}")
                        
                with col_right:
                    st.subheader("Key Risks")
                    for r in cb.get("key_risks", []):
                        st.markdown(f"❌ {r}")
                
                st.info(f"**Executive Summary:** {cb.get('summary', '')}")
                
                # ── ROW 3: Insights Text ────────────────────────────────────
                st.markdown('<h3 class="section-header">AI Extraction Insights</h3>', unsafe_allow_html=True)
                
                s_label = result.get("sentiment", {}).get("label", "N/A").title()
                s_conf = result.get("sentiment", {}).get("score", 0)
                st.write(f"**Financial Sentiment:** {s_label} (Confidence: {s_conf:.1%})")
                
                ner = result.get("ner", {})
                orgs = ", ".join(ner.get("org_mentions", []))
                revs = ", ".join(ner.get("revenue_mentions", []))
                st.write(f"**Organizations Detected:** {orgs if orgs else 'None'}")
                st.write(f"**Revenue Figures Extracted:** {revs if revs else 'None'}")
                
                # ── ROW 4: Similar Companies Table ─────────────────────────
                st.markdown('<h3 class="section-header">Comparable Historical Cases</h3>', unsafe_allow_html=True)
                sim_profiles = result.get("similar_profiles", [])
                if sim_profiles:
                    # Flatten for neat dataframe
                    table_data = []
                    for sp in sim_profiles:
                        table_data.append({
                            "Score (Similarity)": f"{(1 - sp.get('distance', 1))*100:.1f}%",
                            "Historical Grade": sp.get("loan_grade", "?"),
                            "Risk Label": sp.get("risk_label", "?").title()
                        })
                    st.dataframe(table_data, use_container_width=True)
                else:
                    st.write("No similar profiles found in database.")
                
                # ── ROW 5: Diagnostics ─────────────────────────────────────────
                st.markdown('<h3 class="section-header">Diagnostics</h3>', unsafe_allow_html=True)
                with st.expander("View Diagnostic Details"):
                    st.write("**Processed Timestamp:**", result.get("timestamp", "N/A"))
                    
                    sentiment = result.get("sentiment", {})
                    st.markdown("**Sentiment Analysis Details:**")
                    st.write(f"- Classification: {sentiment.get('label', 'N/A').capitalize()}")
                    st.write(f"- Confidence Score: {sentiment.get('score', 0):.4f}")
                    st.write(f"- Text Analyzed: *\"{sentiment.get('text_preview', 'N/A')}\"*")
                    
                    ner = result.get("ner", {})
                    st.markdown("**Entity Extraction Engine:**")
                    st.write(f"- Organizations: {', '.join(ner.get('org_mentions', [])) if ner.get('org_mentions') else 'None detected'}")
                    st.write(f"- People: {', '.join(ner.get('person_mentions', [])) if ner.get('person_mentions') else 'None detected'}")
                    st.write(f"- Financials: {', '.join(ner.get('revenue_mentions', [])) if ner.get('revenue_mentions') else 'None detected'}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to API. Is the FastAPI backend running on port 8000?")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
else:
    # Empty state instructions
    st.info("👈 Fill out the business profile in the sidebar and click **Analyze Credit Risk** to begin.")
