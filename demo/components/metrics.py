import streamlit as st


def render_metrics(stats: dict):
    """
    Display the top dashboard metrics.

    Parameters
    ----------
    stats : dict
        Dictionary returned by
        IdeaLoader.get_statistics()
    """

    st.subheader("Project Statistics")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="Research Papers",
            value=stats["papers"]
        )

    with col2:
        st.metric(
            label="Graph Nodes",
            value=stats["graph_nodes"]
        )

    with col3:
        st.metric(
            label="Relationships",
            value=stats["graph_edges"]
        )

    with col4:
        st.metric(
            label="Candidate Gaps",
            value=stats["candidate_gaps"]
        )

    with col5:
        st.metric(
            label="Research Ideas",
            value=stats["research_ideas"]
        )

    st.divider()