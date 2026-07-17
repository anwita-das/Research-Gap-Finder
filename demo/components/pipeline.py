import time
import streamlit as st


PIPELINE_STAGES = [

    (
        "📄 Paper Collection",
        "Searching ArXiv...",
        0.45
    ),

    (
        "📄 Paper Collection",
        "Searching OpenAlex...",
        0.45
    ),

    (
        "📚 Dataset Integration",
        "Merging datasets & removing duplicates...",
        0.45
    ),

    (
        "🧠 Entity Extraction",
        "Extracting Methods, Models, Datasets, Tasks and Claims...",
        0.60
    ),

    (
        "🕸️ Knowledge Graph",
        "Building heterogeneous knowledge graph...",
        0.60
    ),

    (
        "🔍 Semantic Search",
        "Generating embeddings and building FAISS index...",
        0.55
    ),

    (
        "📈 Graph Analysis",
        "Running Motif Analysis...",
        0.55
    ),

    (
        "📈 Graph Analysis",
        "Running GraphSAGE Link Prediction...",
        0.60
    ),

    (
        "📑 Context Retrieval",
        "Retrieving paper metadata and graph neighborhood...",
        0.55
    ),

    (
        "📖 Paper Enrichment",
        "Analyzing papers and extracting structured summaries...",
        0.70
    ),

    (
        "📅 Temporal Analysis",
        "Analyzing publication trends and concept evolution...",
        0.55
    ),

    (
        "⚖️ Semantic Comparison",
        "Comparing methodologies, datasets and limitations...",
        0.60
    ),

    (
        "🎯 Gap Detection",
        "Validating candidate research gaps...",
        0.70
    ),

    (
        "💡 Hypothesis Generation",
        "Generating novel research hypotheses...",
        0.60
    ),

    (
        "⭐ Novelty Scoring",
        "Computing novelty, impact and feasibility scores...",
        0.55
    ),

    (
        "🏆 Ranking",
        "Ranking final research ideas...",
        0.70
    ),
]


def simulate_pipeline():
    """
    Simulates execution of the entire
    Research Gap Finder pipeline.
    """

    st.subheader("Pipeline Execution")

    progress_bar = st.progress(0)

    status = st.empty()

    log_box = st.empty()

    logs = []

    total = len(PIPELINE_STAGES)

    for i, (stage, message, delay) in enumerate(PIPELINE_STAGES):

        status.markdown(
            f"### {stage}"
        )

        logs.append(
            f"✅ {message}"
        )

        log_box.code(
            "\n".join(logs),
            language="text"
        )

        progress_bar.progress(
            (i + 1) / total
        )

        time.sleep(delay)

    status.success(
        "Pipeline completed successfully!"
    )

    st.success(
        "Research ideas generated successfully."
    )

    time.sleep(0.5)