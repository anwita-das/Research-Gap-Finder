from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ==========================================================
# Phase 5 Output
# ==========================================================


class CandidateEdge(BaseModel):
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


class NeighborNode(BaseModel):
    """
    Represents one neighboring node in the knowledge graph.
    """

    node_id: str

    node_type: str

    relation: str


# ==========================================================
# Phase 6 Context Retrieval
# ==========================================================


class PaperContext(BaseModel):
    """
    Context required for research gap reasoning.
    """

    candidate: CandidateEdge

    source_paper: dict[str, Any]

    target_paper: dict[str, Any]


    source_entities: dict[str, Any]

    target_entities: dict[str, Any]


    shared_entity_context: dict[str, Any] = Field(
        default_factory=dict
    )


    source_neighbors: list[NeighborNode] = Field(
        default_factory=list
    )


    target_neighbors: list[NeighborNode] = Field(
        default_factory=list
    )


# ==========================================================
# Section Extraction Output
# ==========================================================


class PaperSections(BaseModel):
    """
    Structured scientific information extracted
    from a paper.
    """

    methodology: list[str] = Field(
        default_factory=list
    )


    experimental_setup: list[str] = Field(
        default_factory=list
    )


    experimental_results: list[str] = Field(
        default_factory=list
    )


    discussion: list[str] = Field(
        default_factory=list
    )


    limitations: list[str] = Field(
        default_factory=list
    )


    future_work: list[str] = Field(
        default_factory=list
    )


    conclusion: list[str] = Field(
        default_factory=list
    )


# ==========================================================
# Enriched Paper Output
# ==========================================================


class PaperSummary(BaseModel):
    """
    Final enriched representation of a paper.
    """


    paper_id: str


    # Metadata

    title: str

    year: int

    authors: list[str]

    abstract: str


    # Existing extracted entities

    methods: list[str] = Field(
        default_factory=list
    )

    models: list[str] = Field(
        default_factory=list
    )

    algorithms: list[str] = Field(
        default_factory=list
    )

    datasets: list[str] = Field(
        default_factory=list
    )

    tasks: list[str] = Field(
        default_factory=list
    )

    metrics: list[str] = Field(
        default_factory=list
    )

    claims: list[str] = Field(
        default_factory=list
    )

    keywords: list[str] = Field(
        default_factory=list
    )


    summary: str = ""


    # Enrichment fields

    experimental_results: list[str] = Field(
        default_factory=list
    )


    limitations: list[str] = Field(
        default_factory=list
    )


    future_work: list[str] = Field(
        default_factory=list
    )


    key_contributions: list[str] = Field(
        default_factory=list
    )


    novelty_points: list[str] = Field(
        default_factory=list
    )


    experimental_setup: str = ""


    implementation_details: list[str] = Field(
        default_factory=list
    )
# ==========================================================
# Temporal Analysis Output
# ==========================================================


class TemporalSummary(BaseModel):
    """
    Output produced by the Temporal Analysis Agent.
    """

    entity: str

    paper_count: int

    publication_counts: dict[int, int] = Field(
        default_factory=dict
    )

    first_year: int

    latest_year: int

    trend: str

    citation_growth: float

    trend_summary: str = ""


# ==========================================================
# Comparison Output
# ==========================================================


class ComparisonSummary(BaseModel):
    """
    Output produced by the Comparison Agent.
    """

    shared_methods: list[str] = Field(
        default_factory=list
    )

    shared_datasets: list[str] = Field(
        default_factory=list
    )

    shared_metrics: list[str] = Field(
        default_factory=list
    )

    different_methods: list[str] = Field(
        default_factory=list
    )

    different_datasets: list[str] = Field(
        default_factory=list
    )

    different_metrics: list[str] = Field(
        default_factory=list
    )

    shared_limitations: list[str] = Field(
        default_factory=list
    )

    complementary_strengths: list[str] = Field(
        default_factory=list
    )

    missing_connections: list[str] = Field(
        default_factory=list
    )

    comparison_summary: str = ""


# ==========================================================
# Research Gap Output
# ==========================================================


class ResearchGap(BaseModel):
    """
    Final output produced by the Gap Detector Agent.
    """

    gap_id: str

    candidate_id: str

    gap_type: str

    title: str

    description: str

    supporting_evidence: list[str] = Field(
        default_factory=list
    )

    comparison_summary: str = ""

    temporal_evidence: TemporalSummary | dict = Field(
        default_factory=dict
    )

    confidence: float = 0.0



# ==========================================================
# Utility
# ==========================================================


def to_dict(obj: BaseModel) -> dict:
    """
    Convert Pydantic model into JSON dictionary.
    """

    return obj.model_dump()