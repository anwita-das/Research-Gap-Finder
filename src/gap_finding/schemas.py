from dataclasses import dataclass, field, asdict
from typing import Any


# ==========================================================
# Phase 5 Output
# ==========================================================

@dataclass
class CandidateEdge:
    gap_id: str

    source_node: str
    target_node: str

    relation: str
    shared_entity: str

    motif_score: float
    graphsage_score: float
    confidence: float

    status: str

# ==========================================================
# Graph Neighbor
# ==========================================================

@dataclass
class NeighborNode:
    """
    Represents one neighboring node in the knowledge graph.
    """

    node_id: str
    node_type: str
    relation: str

# ==========================================================
# Phase 6 - Context Retrieval Output
# ==========================================================

@dataclass
class PaperContext:
    """
    Everything required to reason about a candidate edge.
    Produced by the Context Retrieval Agent.
    """

    candidate: CandidateEdge

    source_paper: dict[str, Any]
    target_paper: dict[str, Any]

    source_entities: dict[str, Any]
    target_entities: dict[str, Any]

    shared_entity_context: dict[str, Any] = field(default_factory=dict)

    source_neighbors: list[NeighborNode] = field(default_factory=list)
    target_neighbors: list[NeighborNode] = field(default_factory=list)


# ==========================================================
# Phase 6 - Paper Reader Output
# ==========================================================

@dataclass
class PaperSections:
    """
    Evidence collected from all chunks of a paper.
    """

    methodology: list[str] = field(default_factory=list)

    experimental_setup: list[str] = field(default_factory=list)

    experimental_results: list[str] = field(default_factory=list)

    discussion: list[str] = field(default_factory=list)

    limitations: list[str] = field(default_factory=list)

    future_work: list[str] = field(default_factory=list)

    conclusion: list[str] = field(default_factory=list)

@dataclass
class PaperSummary:
    """
    Final enriched representation of a paper.

    Existing entity extraction is reused.
    The Paper Reader Agent only fills the enrichment fields.
    """

    paper_id: str

    # ---------- Existing metadata ----------
    title: str
    year: int
    authors: list[str]
    abstract: str

    # ---------- Existing entity extraction ----------
    methods: list[str]
    models: list[str]
    algorithms: list[str]
    datasets: list[str]
    tasks: list[str]
    metrics: list[str]
    claims: list[str]
    keywords: list[str]
    summary: str

    # ---------- Newly enriched information ----------
    experimental_results: list[str] = field(default_factory=list)

    limitations: list[str] = field(default_factory=list)

    future_work: list[str] = field(default_factory=list)

    key_contributions: list[str] = field(default_factory=list)

    novelty_points: list[str] = field(default_factory=list)

    experimental_setup: str = ""

    implementation_details: list[str] = field(default_factory=list)


# ==========================================================
# Utility
# ==========================================================

def to_dict(obj):
    """
    Convert any dataclass into a JSON serializable dictionary.
    """
    return asdict(obj)