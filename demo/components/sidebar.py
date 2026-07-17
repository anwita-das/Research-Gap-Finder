import streamlit as st


def render_sidebar(loader):
    """
    Render the sidebar controls.

    Returns:
        selected_direction
    """

    st.sidebar.title("Filters")

    directions = ["All"] + loader.get_directions()

    selected_direction = st.sidebar.selectbox(
        "Research Direction",
        directions
    )

    st.sidebar.markdown("---")

    return selected_direction