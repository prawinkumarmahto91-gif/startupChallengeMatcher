"""
streamlit_app.py

Interactive demo UI for presenting the Startup-Challenge Matching system
to judges/reviewers. This calls match_startups() DIRECTLY as a Python
function (not over HTTP) - so this is for demo/presentation purposes,
not a substitute for the api.py REST API that the backend team uses.

Run locally:
    streamlit run streamlit_app.py

Deploy for free (no credit card):
    1. Push this project to a public GitHub repo
    2. Go to https://share.streamlit.io, sign in with GitHub
    3. "New app" -> select this repo -> main file: streamlit_app.py
    4. Deploy (takes 2-5 minutes)
"""

import streamlit as st

from src.matcher import match_startups
from src.similarity import load_ready_data

st.set_page_config(page_title="Startup-Challenge Matching", page_icon="🚀", layout="wide")

st.title("🚀 Startup–Challenge Matching System")
st.caption("SIH 2026 — Government Innovation Procurement Platform")

with st.expander("ℹ️ How this system works"):
    st.markdown(
        """
        1. **Semantic Retrieval** — BGE-M3 embeddings find startups whose profile
           semantically matches the challenge, even with different wording.
        2. **Eligibility Filtering** — Hard requirements (DPIIT recognition,
           certifications, minimum experience) are checked. A startup fails
           here regardless of how high its similarity score is.
        3. **Hybrid Ranking** — Semantic similarity is combined with technology
           match, sector match, experience, budget fit, and location fit into
           one final weighted score.
        4. **Explainability** — Every recommendation includes real,
           data-derived reasons — never invented.

        **This system recommends candidates for expert review.
        It does not make the final procurement decision.**
        """
    )


@st.cache_data
def get_challenges():
    _, challenges = load_ready_data()
    return challenges


challenges_df = get_challenges()

challenge_options = {
    f"{row['challenge_id']} — {row['title']}": row["challenge_id"]
    for _, row in challenges_df.iterrows()
}

col_select, col_slider = st.columns([3, 1])
with col_select:
    selected_label = st.selectbox("Select a Government Challenge", list(challenge_options.keys()))
with col_slider:
    top_n = st.slider("Number of recommendations", min_value=3, max_value=10, value=5)

selected_id = challenge_options[selected_label]

if st.button("🔍 Find Matching Startups", type="primary", use_container_width=True):
    with st.spinner("Running semantic matching, eligibility checks, and ranking..."):
        result = match_startups(challenge_id=selected_id, top_n_final=top_n)

    st.subheader(f"Results for: {result['challenge_title']}")

    if not result["recommendations"]:
        st.warning("No eligible startups found for this challenge.")
    else:
        st.success(f"{len(result['recommendations'])} eligible startups found.")

        for rec in result["recommendations"]:
            with st.container(border=True):
                header_col, score_col = st.columns([3, 1])
                with header_col:
                    st.markdown(f"### #{rec['rank']} — {rec['startup_name']}")
                with score_col:
                    st.metric("Final Score", f"{rec['final_score']:.1f} / 100")

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Semantic", f"{rec['semantic_score']:.2f}")
                m2.metric("Technology", f"{rec['technology_match']:.2f}")
                m3.metric("Sector", f"{rec['sector_match']:.2f}")
                m4.metric("Experience", f"{rec['experience_score']:.2f}")
                m5.metric("Budget", f"{rec['budget_score']:.2f}")

                with st.expander("Why this recommendation?"):
                    for reason in rec["reasons"]:
                        st.markdown(f"- {reason}")

st.divider()
st.caption(
    "Built with BAAI/bge-m3 semantic embeddings, deterministic eligibility "
    "rules, and a transparent hybrid ranking formula."
)