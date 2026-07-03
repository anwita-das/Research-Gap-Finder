"""
Unit tests for semantic graph builder components.

Tests entity deduplication, node creation, edge management, and
the high-level add_semantic_entities_to_paper integration.
"""

import hashlib

import pytest

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
    TASK,
    USES_ALGORITHM,
    USES_DATASET,
    USES_METHOD,
    USES_MODEL,
)


class TestEntityResolver:
    """Tests for EntityResolver normalization and deduplication."""

    def test_normalize_basic_lowercase(self):
        """Test that normalization converts to lowercase."""
        resolver = EntityResolver()
        assert resolver.normalize("ImageNet") == "imagenet"
        assert resolver.normalize("IMAGENET") == "imagenet"

    def test_normalize_whitespace(self):
        """Test that normalization handles whitespace."""
        resolver = EntityResolver()
        assert resolver.normalize(" ImageNet ") == "imagenet"
        assert resolver.normalize("  image   net  ") == "image_net"

    def test_normalize_spaces_to_underscores(self):
        """Test that spaces are replaced with underscores."""
        resolver = EntityResolver()
        assert resolver.normalize("image net") == "image_net"
        assert resolver.normalize("cross validation") == "cross_validation"

    def test_normalize_punctuation_removal(self):
        """Test that punctuation is removed."""
        resolver = EntityResolver()
        assert resolver.normalize("ResNet-50") == "resnet50"
        assert resolver.normalize("F1-Score") == "f1score"

    def test_entity_existence_check(self):
        """Test entity_exists method."""
        resolver = EntityResolver()
        assert not resolver.entity_exists(DATASET, "imagenet")

        resolver.register_entity(DATASET, "imagenet", "dataset:imagenet")
        assert resolver.entity_exists(DATASET, "imagenet")

    def test_register_and_retrieve_entity(self):
        """Test entity registration and retrieval."""
        resolver = EntityResolver()
        resolver.register_entity(METHOD, "cross_validation", "method:cross_validation")

        assert resolver.get_entity(METHOD, "cross_validation") == "method:cross_validation"
        assert resolver.get_entity(METHOD, "nonexistent") is None

    def test_invalid_entity_type_raises_error(self):
        """Test that invalid entity types raise ValueError."""
        resolver = EntityResolver()
        with pytest.raises(ValueError):
            resolver.entity_exists("INVALID_TYPE", "test")

    def test_stats_tracks_registrations(self):
        """Test that stats method tracks entity counts."""
        resolver = EntityResolver()
        resolver.register_entity(DATASET, "imagenet", "dataset:imagenet")
        resolver.register_entity(METHOD, "cross_validation", "method:cross_validation")

        stats = resolver.stats()
        assert stats[DATASET] == 1
        assert stats[METHOD] == 1
        assert stats[ALGORITHM] == 0


class TestNodeFactory:
    """Tests for NodeFactory node creation."""

    def test_create_method_node(self):
        """Test method node creation."""
        node = NodeFactory.create_method("Cross-Validation", "cross_validation")

        assert node["node_id"] == "method:cross_validation"
        assert node["label"] == METHOD
        assert node["name"] == "Cross-Validation"

    def test_create_dataset_node(self):
        """Test dataset node creation."""
        node = NodeFactory.create_dataset("ImageNet", "imagenet")

        assert node["node_id"] == "dataset:imagenet"
        assert node["label"] == DATASET
        assert node["name"] == "ImageNet"

    def test_create_task_node(self):
        """Test task node creation."""
        node = NodeFactory.create_task("Image Classification", "image_classification")

        assert node["node_id"] == "task:image_classification"
        assert node["label"] == TASK
        assert node["name"] == "Image Classification"

    def test_create_claim_node(self):
        """Test claim node creation with SHA1 hash."""
        claim_text = "CNNs improve accuracy"
        normalized = "cnns_improve_accuracy"

        node = NodeFactory.create_claim(claim_text, normalized)

        # Verify structure
        assert node["label"] == CLAIM
        assert node["text"] == claim_text
        assert node["normalized_text"] == normalized

        # Verify node_id uses SHA1 hash
        expected_hash = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
        assert node["node_id"] == f"claim:{expected_hash}"

    def test_create_entity_generic_method(self):
        """Test generic create_entity method."""
        node = NodeFactory.create_entity(ALGORITHM, "Gradient Descent", "gradient_descent")

        assert node["node_id"] == "algorithm:gradient_descent"
        assert node["label"] == ALGORITHM
        assert node["name"] == "Gradient Descent"

    def test_create_entity_invalid_type_raises_error(self):
        """Test that invalid entity type raises ValueError."""
        with pytest.raises(ValueError):
            NodeFactory.create_entity("INVALID", "test", "test")


class TestGraphBuilderDuplication:
    """Tests for duplicate entity detection and deduplication."""

    def test_duplicate_datasets_create_single_node(self):
        """Test that duplicate dataset names create only one node."""
        builder = GraphBuilder()

        id1 = builder.add_dataset("ImageNet")
        id2 = builder.add_dataset("ImageNet")

        assert id1 == id2
        assert builder.node_count() == 1

    def test_duplicate_methods_create_single_node(self):
        """Test that duplicate method names create only one node."""
        builder = GraphBuilder()

        id1 = builder.add_method("Cross-Validation")
        id2 = builder.add_method("Cross-Validation")

        assert id1 == id2
        assert builder.node_count() == 1

    def test_duplicate_tasks_create_single_node(self):
        """Test that duplicate task names create only one node."""
        builder = GraphBuilder()

        id1 = builder.add_task("Image Classification")
        id2 = builder.add_task("Image Classification")

        assert id1 == id2
        assert builder.node_count() == 1

    def test_normalization_deduplication_variants(self):
        """Test that all normalized variants map to same node."""
        builder = GraphBuilder()

        # All these should normalize to the same value
        id1 = builder.add_dataset("ImageNet")
        id2 = builder.add_dataset("imagenet")
        id3 = builder.add_dataset(" IMAGE NET ")

        assert id1 == id2 == id3
        assert builder.node_count() == 1

    def test_duplicate_claims_deduplicated_by_hash(self):
        """Test that duplicate claims are deduplicated by SHA1 hash."""
        builder = GraphBuilder()

        # These should normalize to the same claim
        id1 = builder.add_claim("CNNs improve accuracy")
        id2 = builder.add_claim("cnns improve accuracy")
        id3 = builder.add_claim(" CNNS IMPROVE ACCURACY ")

        assert id1 == id2 == id3
        assert builder.node_count() == 1


class TestGraphBuilderNodeCreation:
    """Tests for node creation and attributes."""

    def test_method_node_has_correct_attributes(self):
        """Test that method node has all required attributes."""
        builder = GraphBuilder()
        node_id = builder.add_method("Attention Mechanism")

        node = builder.get_node(node_id)
        assert node["label"] == METHOD
        assert node["name"] == "Attention Mechanism"
        assert node["normalized_name"] == "attention_mechanism"

    def test_dataset_node_has_correct_attributes(self):
        """Test that dataset node has all required attributes."""
        builder = GraphBuilder()
        node_id = builder.add_dataset("CIFAR-10")

        node = builder.get_node(node_id)
        assert node["label"] == DATASET
        assert node["name"] == "CIFAR-10"
        assert node["normalized_name"] == "cifar10"

    def test_claim_node_has_correct_attributes(self):
        """Test that claim node has all required attributes."""
        builder = GraphBuilder()
        node_id = builder.add_claim("Transfer learning helps with limited data")

        node = builder.get_node(node_id)
        assert node["label"] == CLAIM
        assert node["text"] == "Transfer learning helps with limited data"
        assert "normalized_text" in node

    def test_node_exists_check(self):
        """Test node_exists method."""
        builder = GraphBuilder()
        method_id = builder.add_method("Grid Search")

        assert builder.node_exists(method_id)
        assert not builder.node_exists("nonexistent:id")

    def test_get_nodes_by_type(self):
        """Test retrieval of nodes by type."""
        builder = GraphBuilder()
        builder.add_method("Method1")
        builder.add_method("Method2")
        builder.add_dataset("Dataset1")

        methods = builder.get_nodes_by_type(METHOD)
        assert len(methods) == 2

        datasets = builder.get_nodes_by_type(DATASET)
        assert len(datasets) == 1


class TestEdgeFactory:
    """Tests for edge creation and relationships."""

    def test_connect_method_creates_edge(self):
        """Test that connect_method creates USES_METHOD edge."""
        builder = GraphBuilder()
        method_id = builder.add_method("Cross-Validation")
        paper_id = "paper:arxiv123"

        # Add paper node to graph
        builder.graph.add_node(paper_id, label="Paper")

        edge_factory = EdgeFactory(builder.graph)
        edge = edge_factory.connect_method(paper_id, method_id)

        assert edge is not None
        assert edge[0] == paper_id
        assert edge[1] == method_id
        assert edge[2] == USES_METHOD

    def test_connect_dataset_creates_edge(self):
        """Test that connect_dataset creates USES_DATASET edge."""
        builder = GraphBuilder()
        dataset_id = builder.add_dataset("ImageNet")
        paper_id = "paper:arxiv456"

        builder.graph.add_node(paper_id, label="Paper")
        edge_factory = EdgeFactory(builder.graph)
        edge = edge_factory.connect_dataset(paper_id, dataset_id)

        assert edge is not None
        assert edge[2] == USES_DATASET

    def test_connect_task_creates_edge(self):
        """Test that connect_task creates ADDRESSES_TASK edge."""
        builder = GraphBuilder()
        task_id = builder.add_task("Image Classification")
        paper_id = "paper:arxiv789"

        builder.graph.add_node(paper_id, label="Paper")
        edge_factory = EdgeFactory(builder.graph)
        edge = edge_factory.connect_task(paper_id, task_id)

        assert edge is not None
        assert edge[2] == ADDRESSES_TASK

    def test_edge_has_correct_attributes(self):
        """Test that edges have correct attributes."""
        builder = GraphBuilder()
        method_id = builder.add_method("Method1")
        paper_id = "paper:test"

        builder.graph.add_node(paper_id, label="Paper")
        edge_factory = EdgeFactory(builder.graph)
        edge_factory.connect_method(paper_id, method_id)

        # Get edge data
        edge_data = builder.graph.get_edge_data(paper_id, method_id, key=USES_METHOD)
        assert edge_data["relation"] == USES_METHOD
        assert edge_data["weight"] == 1.0
        assert edge_data["key"] == USES_METHOD

    def test_duplicate_edges_not_inserted(self):
        """Test that duplicate edges are not inserted."""
        builder = GraphBuilder()
        method_id = builder.add_method("Method1")
        paper_id = "paper:test"

        builder.graph.add_node(paper_id, label="Paper")
        edge_factory = EdgeFactory(builder.graph)

        # Add same edge twice
        edge1 = edge_factory.connect_method(paper_id, method_id)
        edge2 = edge_factory.connect_method(paper_id, method_id)

        # First should succeed, second should return None
        assert edge1 is not None
        assert edge2 is None

        # Only one edge should exist
        assert builder.graph.out_degree(paper_id) == 1

    def test_all_relationship_types_created_correctly(self):
        """Test that all relationship types are created correctly."""
        builder = GraphBuilder()
        paper_id = "paper:test"
        builder.graph.add_node(paper_id, label="Paper")
        edge_factory = EdgeFactory(builder.graph)

        # Create one of each relationship type
        method_id = builder.add_method("Method1")
        model_id = builder.add_model("Model1")
        algorithm_id = builder.add_algorithm("Algorithm1")
        dataset_id = builder.add_dataset("Dataset1")
        task_id = builder.add_task("Task1")
        metric_id = builder.add_metric("Metric1")
        keyword_id = builder.add_keyword("Keyword1")
        field_id = builder.add_field("Field1")
        claim_id = builder.add_claim("Claim1")

        edge_factory.connect_method(paper_id, method_id)
        edge_factory.connect_model(paper_id, model_id)
        edge_factory.connect_algorithm(paper_id, algorithm_id)
        edge_factory.connect_dataset(paper_id, dataset_id)
        edge_factory.connect_task(paper_id, task_id)
        edge_factory.connect_metric(paper_id, metric_id)
        edge_factory.connect_keyword(paper_id, keyword_id)
        edge_factory.connect_field(paper_id, field_id)
        edge_factory.connect_claim(paper_id, claim_id)

        # Verify all edges exist with correct types
        assert builder.graph.has_edge(paper_id, method_id, key=USES_METHOD)
        assert builder.graph.has_edge(paper_id, model_id, key=USES_MODEL)
        assert builder.graph.has_edge(paper_id, algorithm_id, key=USES_ALGORITHM)
        assert builder.graph.has_edge(paper_id, dataset_id, key=USES_DATASET)
        assert builder.graph.has_edge(paper_id, task_id, key=ADDRESSES_TASK)
        assert builder.graph.has_edge(paper_id, metric_id, key=EVALUATED_BY)
        assert builder.graph.has_edge(paper_id, keyword_id, key=HAS_KEYWORD)
        assert builder.graph.has_edge(paper_id, field_id, key=BELONGS_TO_FIELD)
        assert builder.graph.has_edge(paper_id, claim_id, key=MAKES_CLAIM)


class TestAddSemanticEntitiesToPaper:
    """Tests for the high-level add_semantic_entities_to_paper integration."""

    def test_adds_entities_and_creates_edges(self):
        """Test that entities are added and edges are created."""
        builder = GraphBuilder()
        paper_id = "paper:test123"
        builder.graph.add_node(paper_id, label="Paper")

        extracted = {
            "methods": ["Cross-Validation"],
            "datasets": ["ImageNet"],
            "tasks": ["Image Classification"],
            "models": [],
            "algorithms": None,
            "metrics": [],
            "keywords": [],
            "fields": [],
            "claims": [],
        }

        summary = builder.add_semantic_entities_to_paper(paper_id, extracted)

        # Check summary
        assert summary["methods_added"] == 1
        assert summary["datasets_added"] == 1
        assert summary["tasks_added"] == 1
        assert summary["total"] == 3

        # Check nodes exist
        assert builder.node_count() == 3

    def test_ignores_empty_lists(self):
        """Test that empty lists are ignored."""
        builder = GraphBuilder()
        paper_id = "paper:test"
        builder.graph.add_node(paper_id, label="Paper")

        extracted = {
            "methods": [],
            "models": [],
            "algorithms": [],
            "datasets": [],
            "tasks": [],
            "metrics": [],
            "keywords": [],
            "fields": [],
            "claims": [],
        }

        summary = builder.add_semantic_entities_to_paper(paper_id, extracted)

        assert summary["total"] == 0
        assert builder.node_count() == 0

    def test_ignores_none_lists(self):
        """Test that None values are ignored."""
        builder = GraphBuilder()
        paper_id = "paper:test"
        builder.graph.add_node(paper_id, label="Paper")

        extracted = {
            "methods": None,
            "models": None,
            "algorithms": None,
            "datasets": None,
            "tasks": None,
            "metrics": None,
            "keywords": None,
            "fields": None,
            "claims": None,
        }

        summary = builder.add_semantic_entities_to_paper(paper_id, extracted)

        assert summary["total"] == 0

    def test_ignores_empty_strings(self):
        """Test that empty strings are ignored."""
        builder = GraphBuilder()
        paper_id = "paper:test"
        builder.graph.add_node(paper_id, label="Paper")

        extracted = {
            "methods": ["", "  ", "Cross-Validation"],
            "models": [],
            "algorithms": [],
            "datasets": [],
            "tasks": [],
            "metrics": [],
            "keywords": [],
            "fields": [],
            "claims": [],
        }

        summary = builder.add_semantic_entities_to_paper(paper_id, extracted)

        # Only one method should be added
        assert summary["methods_added"] == 1
        assert builder.node_count() == 1

    def test_ignores_duplicates_in_extracted_list(self):
        """Test that duplicates within a single extraction list are ignored."""
        builder = GraphBuilder()
        paper_id = "paper:test"
        builder.graph.add_node(paper_id, label="Paper")

        extracted = {
            "methods": ["Cross-Validation", "cross-validation", "CROSS VALIDATION"],
            "models": [],
            "algorithms": [],
            "datasets": [],
            "tasks": [],
            "metrics": [],
            "keywords": [],
            "fields": [],
            "claims": [],
        }

        summary = builder.add_semantic_entities_to_paper(paper_id, extracted)

        # All three should deduplicate to one method
        assert summary["methods_added"] == 1
        assert builder.node_count() == 1

    def test_all_entity_types_processed(self):
        """Test that all entity types are processed."""
        builder = GraphBuilder()
        paper_id = "paper:complete"
        builder.graph.add_node(paper_id, label="Paper")

        extracted = {
            "methods": ["Method1"],
            "models": ["Model1"],
            "algorithms": ["Algorithm1"],
            "datasets": ["Dataset1"],
            "tasks": ["Task1"],
            "metrics": ["Metric1"],
            "keywords": ["Keyword1"],
            "fields": ["Field1"],
            "claims": ["Claim1"],
        }

        summary = builder.add_semantic_entities_to_paper(paper_id, extracted)

        assert summary["methods_added"] == 1
        assert summary["models_added"] == 1
        assert summary["algorithms_added"] == 1
        assert summary["datasets_added"] == 1
        assert summary["tasks_added"] == 1
        assert summary["metrics_added"] == 1
        assert summary["keywords_added"] == 1
        assert summary["fields_added"] == 1
        assert summary["claims_added"] == 1
        assert summary["total"] == 9

    def test_reuses_existing_entities(self):
        """Test that existing entities are reused."""
        builder = GraphBuilder()
        paper_id1 = "paper:first"
        paper_id2 = "paper:second"

        builder.graph.add_node(paper_id1, label="Paper")
        builder.graph.add_node(paper_id2, label="Paper")

        extracted = {
            "methods": ["Grid Search"],
            "models": [],
            "algorithms": [],
            "datasets": [],
            "tasks": [],
            "metrics": [],
            "keywords": [],
            "fields": [],
            "claims": [],
        }

        # Add to first paper
        summary1 = builder.add_semantic_entities_to_paper(paper_id1, extracted)
        assert summary1["methods_added"] == 1
        assert builder.node_count() == 1

        # Add same entity to second paper
        summary2 = builder.add_semantic_entities_to_paper(paper_id2, extracted)
        assert summary2["methods_added"] == 1
        assert builder.node_count() == 1  # No new node created

    def test_complex_extraction_scenario(self):
        """Test a complex extraction with mixed valid and invalid data."""
        builder = GraphBuilder()
        paper_id = "paper:complex"
        builder.graph.add_node(paper_id, label="Paper")

        extracted = {
            "methods": ["Cross-Validation", None, "", "  ", "Grid Search", "cross-validation"],
            "models": ["ResNet", "ResNet", "VGG"],  # ResNet duplicated
            "algorithms": ["Gradient Descent"],
            "datasets": [None, "ImageNet", "  IMAGENET  "],  # ImageNet duplicated
            "tasks": ["Image Classification"],
            "metrics": [],
            "keywords": ["Deep Learning", "CNN", "deep learning"],  # deep learning duplicated
            "fields": None,
            "claims": ["CNNs improve accuracy", "  CNNS IMPROVE ACCURACY  "],  # Duplicated
        }

        summary = builder.add_semantic_entities_to_paper(paper_id, extracted)

        # Expected counts:
        # methods: Cross-Validation, Grid Search (cross-validation deduplicated) = 2
        # models: ResNet, VGG (one ResNet deduplicated) = 2
        # algorithms: Gradient Descent = 1
        # datasets: ImageNet (duplicated and deduplicated) = 1
        # tasks: Image Classification = 1
        # keywords: Deep Learning, CNN (deep learning deduplicated) = 2
        # claims: CNNs improve accuracy (deduplicated) = 1
        # Total = 10

        assert summary["methods_added"] == 2
        assert summary["models_added"] == 2
        assert summary["algorithms_added"] == 1
        assert summary["datasets_added"] == 1
        assert summary["tasks_added"] == 1
        assert summary["metrics_added"] == 0
        assert summary["keywords_added"] == 2
        assert summary["fields_added"] == 0
        assert summary["claims_added"] == 1
        assert summary["total"] == 10


class TestGraphBuilderUtilities:
    """Tests for utility methods."""

    def test_node_count(self):
        """Test node count method."""
        builder = GraphBuilder()
        assert builder.node_count() == 0

        builder.add_method("Method1")
        assert builder.node_count() == 1

        builder.add_dataset("Dataset1")
        assert builder.node_count() == 2

    def test_type_counts(self):
        """Test type_counts method."""
        builder = GraphBuilder()
        builder.add_method("Method1")
        builder.add_method("Method2")
        builder.add_dataset("Dataset1")
        builder.add_task("Task1")

        counts = builder.type_counts()
        assert counts[METHOD] == 2
        assert counts[DATASET] == 1
        assert counts[TASK] == 1

    def test_resolver_stats(self):
        """Test resolver stats method."""
        builder = GraphBuilder()
        builder.add_method("Method1")
        builder.add_dataset("Dataset1")
        builder.add_dataset("Dataset2")

        stats = builder.resolver_stats()
        assert stats[METHOD] == 1
        assert stats[DATASET] == 2
        assert stats[ALGORITHM] == 0
