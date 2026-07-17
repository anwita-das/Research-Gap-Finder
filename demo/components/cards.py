import streamlit as st


def _score_color(score: float):
    """
    Returns a colored emoji based on score.
    """

    if score >= 0.73:
        return "🟢"

    if score >= 0.60:
        return "🟡"

    return "🔴"


def _render_score(label: str, value: float):

    st.write(f"**{label}:** {value:.2f}")

    st.progress(float(value))


def render_cards(ideas):
    """
    Render all ranked research ideas.
    """

    st.header("🏆 Ranked Research Ideas")

    if not ideas:

        st.warning(
            "No research ideas found."
        )

        return

    for idea in ideas:

        rank = idea["rank"]

        score = idea["final_score"]

        color = _score_color(score)

        title = (
            f"{color} "
            f"Rank #{rank}  |  "
            f"{idea['research_direction']}"
        )

        with st.expander(title, expanded=(rank == 1)):

            col1, col2 = st.columns([3, 1])

            with col1:

                st.subheader("Research Question")

                st.write(
                    idea["research_question"]
                )

            with col2:

                st.metric(
                    "Final Score",
                    f"{score:.3f}"
                )

            st.markdown("---")

            st.subheader("Hypothesis")

            st.write(
                idea["hypothesis"]
            )

            st.subheader("Proposed Methodology")

            st.write(
                idea["proposed_methodology"]
            )

            st.subheader("Expected Contribution")

            st.write(
                idea["expected_contribution"]
            )

            st.subheader("Supporting Explanation")

            st.write(
                idea["supporting_explanation"]
            )

            st.markdown("---")

            st.subheader("Evaluation Scores")

            c1, c2 = st.columns(2)

            with c1:

                _render_score(
                    "Novelty",
                    idea["novelty_score"]
                )

                _render_score(
                    "Impact",
                    idea["impact_score"]
                )

            with c2:

                _render_score(
                    "Feasibility",
                    idea["feasibility_score"]
                )

                _render_score(
                    "Competition",
                    idea["competition_score"]
                )

            st.markdown("---")

            st.subheader("Score Justification")

            st.markdown(
                f"""
**Novelty**

{idea["novelty_reason"]}

---

**Impact**

{idea["impact_reason"]}

---

**Feasibility**

{idea["feasibility_reason"]}

---

**Competition**

{idea["competition_reason"]}
"""
            )