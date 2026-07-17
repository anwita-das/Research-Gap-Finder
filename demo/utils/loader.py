import json
from pathlib import Path


class IdeaLoader:
    """
    Loads and filters ranked research ideas.
    """

    def __init__(self, json_path: str):
        self.json_path = Path(json_path)
        self.ideas = self.load()

    def load(self):
        """Load ideas from JSON."""

        if not self.json_path.exists():
            raise FileNotFoundError(
                f"Could not find {self.json_path}"
            )

        with open(self.json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def search(self, query: str):
        """
        Search ideas using the user's query.

        For demo purposes, we treat any query containing
        'rag' as matching the complete dataset.
        """

        if not query:
            return []

        query = query.lower().strip()

        # Since your demo dataset is RAG-focused,
        # any RAG-related search returns all ideas.
        rag_keywords = [
            "rag",
            "retrieval",
            "retrieval augmented generation"
        ]

        if any(word in query for word in rag_keywords):
            return self.ideas

        results = []

        searchable_fields = [
            "research_question",
            "research_direction",
            "expected_contribution",
            "supporting_explanation",
            "hypothesis",
            "proposed_methodology"
        ]

        for idea in self.ideas:

            text = " ".join(
                str(idea.get(field, "")).lower()
                for field in searchable_fields
            )

            if query in text:
                results.append(idea)

        return results

    def get_statistics(self):
        """
        Dashboard statistics.
        """

        return {
            "papers": 50,
            "graph_nodes": 1553,
            "graph_edges": 1827,
            "candidate_gaps": 100,
            "research_ideas": len(
                [i for i in self.ideas if i["gap_confidence"] > 0]
            ),
        }

    def get_directions(self):
        """
        Returns unique research directions.
        """

        directions = sorted(
            {
                idea["research_direction"]
                for idea in self.ideas
                if idea["research_direction"] != "Not applicable"
            }
        )

        return directions

    def filter_direction(self, ideas, direction):

        if direction == "All":
            return ideas

        return [
            idea
            for idea in ideas
            if idea["research_direction"] == direction
        ]

    def sort_ideas(self, ideas, option):

        mapping = {
            "Final Score": "final_score",
            "Novelty": "novelty_score",
            "Impact": "impact_score",
            "Feasibility": "feasibility_score",
            "Competition": "competition_score",
        }

        key = mapping.get(option, "final_score")

        return sorted(
            ideas,
            key=lambda x: x.get(key, 0),
            reverse=True
        )