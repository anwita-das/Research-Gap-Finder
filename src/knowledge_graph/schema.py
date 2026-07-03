"""
Knowledge Graph schema constants for semantic entities.

Defines node labels, relationship types, and collections used throughout
the knowledge graph construction pipeline.
"""

# ============================================================================
# Node Labels for Semantic Entities
# ============================================================================

METHOD: str = "Method"
"""Label for research method nodes."""

MODEL: str = "Model"
"""Label for machine learning model nodes."""

ALGORITHM: str = "Algorithm"
"""Label for algorithm nodes."""

DATASET: str = "Dataset"
"""Label for dataset nodes."""

TASK: str = "Task"
"""Label for research task nodes."""

METRIC: str = "Metric"
"""Label for evaluation metric nodes."""

KEYWORD: str = "Keyword"
"""Label for keyword nodes."""

FIELD: str = "Field"
"""Label for research field nodes."""

CLAIM: str = "Claim"
"""Label for research claim nodes."""


# ============================================================================
# Relationship Types
# ============================================================================

USES_METHOD: str = "USES_METHOD"
"""Relationship: Paper uses a method."""

USES_MODEL: str = "USES_MODEL"
"""Relationship: Paper uses a model."""

USES_ALGORITHM: str = "USES_ALGORITHM"
"""Relationship: Paper uses an algorithm."""

USES_DATASET: str = "USES_DATASET"
"""Relationship: Paper uses a dataset."""

ADDRESSES_TASK: str = "ADDRESSES_TASK"
"""Relationship: Paper addresses a task."""

EVALUATED_BY: str = "EVALUATED_BY"
"""Relationship: Paper is evaluated by a metric."""

HAS_KEYWORD: str = "HAS_KEYWORD"
"""Relationship: Paper has a keyword."""

BELONGS_TO_FIELD: str = "BELONGS_TO_FIELD"
"""Relationship: Paper belongs to a field."""

MAKES_CLAIM: str = "MAKES_CLAIM"
"""Relationship: Paper makes a claim."""


# ============================================================================
# Collections
# ============================================================================

NODE_TYPES: frozenset[str] = frozenset(
    [METHOD, MODEL, ALGORITHM, DATASET, TASK, METRIC, KEYWORD, FIELD, CLAIM]
)
"""All available node types in the knowledge graph."""

RELATION_TYPES: frozenset[str] = frozenset(
    [
        USES_METHOD,
        USES_MODEL,
        USES_ALGORITHM,
        USES_DATASET,
        ADDRESSES_TASK,
        EVALUATED_BY,
        HAS_KEYWORD,
        BELONGS_TO_FIELD,
        MAKES_CLAIM,
    ]
)
"""All available relationship types in the knowledge graph."""
