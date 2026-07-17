from pathlib import Path

import streamlit as st

from utils.loader import IdeaLoader
from components.sidebar import render_sidebar
from components.metrics import render_metrics
from components.pipeline import simulate_pipeline
from components.cards import render_cards


# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="Research Gap Finder",
    page_icon="🔬",
    layout="wide"
)


# ======================================================
# LOAD DATA
# ======================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

JSON_PATH = (
    PROJECT_ROOT /
    "data" /
    "processed" /
    "ranked_ideas.json"
)

loader = IdeaLoader(JSON_PATH)


# ======================================================
# SESSION STATE
# ======================================================

if "generated" not in st.session_state:
    st.session_state.generated = False

if "results" not in st.session_state:
    st.session_state.results = []

if "query" not in st.session_state:
    st.session_state.query = ""


# ======================================================
# HEADER
# ======================================================

st.markdown(
    """
    <h1 style="
        text-align:center;
        font-size:3.3rem;
        font-weight:800;
        margin-bottom:0;
    ">
        🔬
        <span style="
            background: linear-gradient(
                90deg,
                #FFFFFF 0%,
                #BFCBFF 45%,
                #7C8CF8 75%,
                #A855F7 100%
            );
            -webkit-background-clip:text;
            -webkit-text-fill-color:transparent;
            background-clip:text;
        ">
            Research Gap Finder
        </span>
    </h1>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
<div style="text-align:center; margin-bottom:30px; font-size:1.35rem; font-weight:600;">
AI-powered Research Gap Discovery using
<span style="background:linear-gradient(90deg,#BDE7FF 0%,#7DB7FF 35%,#6F7CFF 70%,#B16CFF 100%);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
font-weight:700;">
Knowledge Graphs and Multi-Agent Analysis
</span>
</div>
""",
    unsafe_allow_html=True,
)

# ======================================================
# SEARCH
# ======================================================

query = st.text_input(
    "Enter a research topic",
    placeholder="Example: RAG"
)

st.markdown("""
<style>

div.stButton > button {
    width: 100%;
    height: 3.3rem;

    border-radius: 12px;
    border: none;

    font-size: 20px;
    font-weight: 600;

    color: white;

    background: linear-gradient(
        90deg,
        #5D5FEF 0%,
        #7C5CF5 55%,
        #A855F7 100%
    );

    transition: all 0.25s ease;
}

div.stButton > button:hover {
    transform: translateY(-2px);

    background: linear-gradient(
        90deg,
        #6D6FFF 0%,
        #8A63FF 55%,
        #B45CFF 100%
    );

    box-shadow: 0 8px 18px rgba(124,92,245,.35);
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

div[data-testid="stVerticalBlockBorderWrapper"]{
    border-radius:16px !important;
    background:#11141B !important;
}

</style>
""", unsafe_allow_html=True)

if st.button(
    "🚀 Generate Research Ideas",
    use_container_width=True
):

    st.session_state.query = query

    st.session_state.results = loader.search(query)

    st.session_state.generated = True

    simulate_pipeline()


def render_pipeline_overview():

    cards = [
        ("01", "📄", "Paper Collection", "ArXiv + OpenAlex", "#6C63FF"),
        ("02", "🕸️", "Knowledge Graph", "Entity Extraction", "#E056B5"),
        ("03", "🔍", "Semantic Search", "FAISS Retrieval", "#3FA9F5"),
        ("04", "🤖", "Gap Detection", "Multi-Agent Analysis", "#84F18A"),
        ("05", "💡", "Hypothesis Gen.", "Idea Ranking", "#F5A623"),
    ]

    cols = st.columns(5, gap="medium")

    for col, (num, icon, title, subtitle, color) in zip(cols, cards):

        with col:

            with st.container(border=True):

                st.markdown(
                    f"""
<div style="text-align:right;
            color:#8A8A8A;
            font-size:13px;
            font-weight:600;">
{num}
</div>
""",
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
<div style="
text-align:center;
font-size:46px;
color:{color};
margin-top:10px;
margin-bottom:8px;
">
{icon}
</div>
""",
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
<div style="
text-align:center;
font-size:20px;
font-weight:700;
color:white;
margin-bottom:6px;
">
{title}
</div>
""",
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
<div style="
text-align:center;
font-size:13px;
color:#AFAFAF;
margin-bottom:12px;
">
{subtitle}
</div>
""",
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
<div style="
height:4px;
width:100%;
background:{color};
border-radius:4px;
margin-top:10px;
">
</div>
""",
                    unsafe_allow_html=True,
                )
                
# Show pipeline overview only before generation
if not st.session_state.generated:

    st.markdown(
        """
<div style="text-align:center; margin-bottom:30px;">

<h2 style="margin-bottom:5px;font-size:1.5rem;color:#7C8CF8;">
Our AI-Powered Pipeline
</h2>

</div>
""",
        unsafe_allow_html=True,
    )

    render_pipeline_overview()

    st.divider()


# ======================================================
# RESULTS
# ======================================================

if st.session_state.generated:

    direction = render_sidebar(loader)

    ideas = loader.filter_direction(
        st.session_state.results,
        direction
    )

    render_metrics(
        loader.get_statistics()
    )

    st.success(
        f"Showing {len(ideas)} research ideas for "
        f'"{st.session_state.query}"'
    )

    render_cards(ideas)