"""Knowledge Graph module for semantic entity management and graph construction."""

from src.knowledge_graph.builder import GraphBuilder
from src.knowledge_graph.edge_factory import EdgeFactory
from src.knowledge_graph.entity_resolver import EntityResolver
from src.knowledge_graph.node_factory import NodeFactory
from src.knowledge_graph.schema import (
    ADDRESSES_TASK,
    ALGORITHM,
    BELONGS_TO_FIELD,
    CLAIM,
    DATASET,
    EVALUATED_BY,
    FIELD,
    HAS_KEYWORD,
    KEYWORD,
    MAKES_CLAIM,
    METHOD,
    METRIC,
    MODEL,
    NODE_TYPES,
    RELATION_TYPES,
    TASK,
    USES_ALGORITHM,
    USES_DATASET,
    USES_METHOD,
    USES_MODEL,
)

__all__ = [
    "GraphBuilder",
    "EdgeFactory",
    "EntityResolver",
    "NodeFactory",
    "NODE_TYPES",
    "RELATION_TYPES",
    "METHOD",
    "MODEL",
    "ALGORITHM",
    "DATASET",
    "TASK",
    "METRIC",
    "KEYWORD",
    "FIELD",
    "CLAIM",
    "USES_METHOD",
    "USES_MODEL",
    "USES_ALGORITHM",
    "USES_DATASET",
    "ADDRESSES_TASK",
    "EVALUATED_BY",
    "HAS_KEYWORD",
    "BELONGS_TO_FIELD",
    "MAKES_CLAIM",
]