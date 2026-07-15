from pathlib import Path
import json

from .schemas import (
    CandidateEdge,
    PaperContext,
    NeighborNode
)


class ContextRetriever:
    """
    Loads all project data once and builds PaperContext objects
    for candidate gaps.
    """

    def __init__(self):

        # -----------------------------
        # Project Paths
        # -----------------------------

        self.project_root = Path(__file__).resolve().parents[2]

        self.data_dir = self.project_root / "data" / "processed"

        self.paper_path = (
            self.data_dir /
            "merged" /
            "merged_RAG.json"
        )

        self.entity_dir = (
            self.data_dir /
            "entities"
        )

        self.graph_path = (
            self.data_dir /
            "knowledge_graph.json"
        )

        self.candidate_path = (
            self.data_dir /
            "final_gap_candidates.json"
        )

        # -----------------------------
        # Cached Data
        # -----------------------------

        self.paper_map = self._load_papers()

        self.entity_map = self._load_entities()

        self.graph_nodes = {}

        self.graph_edges = []

        self.adjacency = {}

        self._load_graph()

        self._build_adjacency_map()

    # ==========================================================
    # Paper Loading
    # ==========================================================

    def _load_papers(self):
        """
        Loads merged_RAG.json into memory.

        Returns
        -------
        dict

            {
                paper_id -> paper_json
            }
        """

        with open(
            self.paper_path,
            "r",
            encoding="utf-8"
        ) as f:

            papers = json.load(f)

        paper_map = {}

        for paper in papers:

            paper_map[
                paper["paper_id"]
            ] = paper

        print(f"Loaded {len(paper_map)} papers.")

        return paper_map

    # ==========================================================
    # Entity Loading
    # ==========================================================

    def _load_entities(self):
        """
        Loads every entity JSON into memory.

        Returns
        -------
        dict

            {
                paper_id -> entity_json
            }
        """

        entity_map = {}

        for file in self.entity_dir.glob("*.json"):

            with open(
                file,
                "r",
                encoding="utf-8"
            ) as f:

                entity = json.load(f)

            entity_map[
                entity["paper_id"]
            ] = entity

        print(f"Loaded {len(entity_map)} entity files.")

        return entity_map
    
    # ==========================================================
    # Candidate Loading
    # ==========================================================

    def load_candidates(self) -> list[CandidateEdge]:
        """
        Loads all candidate gaps generated in Phase 5.
        """

        with open(
            self.candidate_path,
            "r",
            encoding="utf-8"
        ) as f:

            candidates = json.load(f)

        candidate_objects = []

        for candidate in candidates:

            candidate_objects.append(

                CandidateEdge(

                    gap_id=candidate["gap_id"],

                    source_node=candidate["source_node"],

                    target_node=candidate["target_node"],

                    relation=candidate["relation"],

                    shared_entity=candidate["shared_entity"],

                    motif_score=candidate["motif_score"],

                    graphsage_score=candidate["graphsage_score"],

                    confidence=candidate["confidence"],

                    status=candidate["status"]

                )

            )

        print(
            f"Loaded {len(candidate_objects)} candidate gaps."
        )

        return candidate_objects
    
    # ==========================================================
    # Knowledge Graph Loading
    # ==========================================================

    def _load_graph(self):
        """
        Loads the exported knowledge graph.
        """

        with open(
            self.graph_path,
            "r",
            encoding="utf-8"
        ) as f:

            graph = json.load(f)

        self.graph_nodes = {
            node["node_id"]: node
            for node in graph["nodes"]
        }

        self.graph_edges = graph["edges"]

        print(
            f"Loaded {len(self.graph_nodes)} graph nodes "
            f"and {len(self.graph_edges)} edges."
        )

    # ==========================================================
    # Adjacency Map
    # ==========================================================

    def _build_adjacency_map(self):
        """
        Builds an adjacency map for fast neighbor lookup.

        adjacency[node_id] -> list of connected edges
        """

        self.adjacency = {}

        for edge in self.graph_edges:

            source = edge["source"]
            target = edge["target"]

            self.adjacency.setdefault(source, []).append(edge)
            self.adjacency.setdefault(target, []).append(edge)

        print(
            f"Built adjacency map for "
            f"{len(self.adjacency)} nodes."
        )

    # ==========================================================
    # Neighbor Extraction
    # ==========================================================

    def _extract_neighbors(
        self,
        paper_node: str
    ) -> list[NeighborNode]:
        """
        Returns all immediate neighbors of a paper node.
        """

        neighbors = []

        edges = self.adjacency.get(
            paper_node,
            []
        )

        for edge in edges:

            if edge["source"] == paper_node:
                neighbor_id = edge["target"]
            else:
                neighbor_id = edge["source"]

            node = self.graph_nodes.get(
                neighbor_id,
                {}
            )

            neighbors.append(

                NeighborNode(

                    node_id=neighbor_id,

                    node_type=node.get(
                        "label",
                        "Unknown"
                    ),

                    relation=edge["relation"]

                )

            )

        return neighbors
    
    # ==========================================================
    # Shared Entity Context
    # ==========================================================

    def _get_shared_entity_context(
        self,
        entity_node: str
    ):
        """
        Returns information about the shared entity and
        every paper connected to it.
        """

        entity = self.graph_nodes.get(
            entity_node,
            {}
        )

        connected_papers = []

        for edge in self.adjacency.get(
            entity_node,
            []
        ):

            if edge["source"] == entity_node:
                other = edge["target"]
            else:
                other = edge["source"]

            if other.startswith("paper:"):

                connected_papers.append(other)

        return {

            "node": entity,

            "connected_papers": connected_papers,

            "paper_count": len(
                connected_papers
            )

        }
    
    # ==========================================================
    # Helpers
    # ==========================================================

    def _paper_id_from_node(
        self,
        node_id: str
    ) -> str:
        """
        Converts

        paper:arxiv_2402.07483v2

        →

        arxiv_2402.07483v2
        """

        return node_id.replace(
            "paper:",
            "",
            1
        )
    
    # ==========================================================
    # Context Builder
    # ==========================================================

    def build_context(
        self,
        candidate: CandidateEdge
    ) -> PaperContext:
        """
        Builds the complete PaperContext required for
        Phase 6 reasoning.
        """

        source_paper_id = self._paper_id_from_node(
            candidate.source_node
        )

        target_paper_id = self._paper_id_from_node(
            candidate.target_node
        )

        source_paper = self.paper_map.get(
            source_paper_id
        )

        target_paper = self.paper_map.get(
            target_paper_id
        )

        if source_paper is None:
            raise ValueError(
                f"Paper not found: {source_paper_id}"
            )

        if target_paper is None:
            raise ValueError(
                f"Paper not found: {target_paper_id}"
            )

        source_entities = self.entity_map.get(
            source_paper_id,
            {}
        )

        target_entities = self.entity_map.get(
            target_paper_id,
            {}
        )

        source_neighbors = self._extract_neighbors(
            candidate.source_node
        )

        target_neighbors = self._extract_neighbors(
            candidate.target_node
        )

        shared_entity_context = self._get_shared_entity_context(
            candidate.shared_entity
        )

        return PaperContext(

            candidate=candidate,

            source_paper=source_paper,
            target_paper=target_paper,

            source_entities=source_entities,
            target_entities=target_entities,

            shared_entity_context=shared_entity_context,

            source_neighbors=source_neighbors,
            target_neighbors=target_neighbors

        )
    
if __name__ == "__main__":

    retriever = ContextRetriever()

    candidate = retriever.load_candidates()[0]

    context = retriever.build_context(candidate)

    print()

    print("=" * 50)

    print("Source Paper")

    print(context.source_paper["title"])

    print()

    print("Target Paper")

    print(context.target_paper["title"])

    print()

    print("Shared Entity")

    print(context.shared_entity_context["node"]["label"])

    print()

    print("Source Neighbors")

    print(len(context.source_neighbors))

    print()

    print("Target Neighbors")

    print(len(context.target_neighbors))