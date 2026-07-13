"""Phase 5 GraphSAGE preprocessing and link prediction for the existing knowledge graph.

This module reuses the existing motif-analysis helpers from src.graph_analysis.motif_analysis
for candidate generation, then scores those candidates with a lightweight GraphSAGE model.
The implementation is intentionally simple and safe for heterogeneous node types such as
paper, author, dataset, method, model, task, metric, and claim.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, Union

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from src.knowledge_graph.graph_loader import load_existing_graph

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class GraphSAGELinkPredictor(Module):
    """A lightweight GraphSAGE model for edge prediction."""

    def __init__(self, in_channels: int, hidden_channels: int = 64) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.out = torch.nn.Linear(hidden_channels * 2, 1)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return x


def load_graph(
    graph_path: Optional[Union[str, Path]] = None,
    *,
    graph_obj: Optional[nx.Graph] = None,
) -> nx.MultiDiGraph:
    """Load the existing graph from disk or reuse an in-memory graph object."""
    if graph_obj is not None:
        if isinstance(graph_obj, nx.MultiDiGraph):
            return graph_obj
        if isinstance(graph_obj, nx.DiGraph):
            return nx.MultiDiGraph(graph_obj)
        raise TypeError("graph_obj must be a NetworkX graph")
    return load_existing_graph(graph_path)


def _node_type_one_hot(node_id: str, graph: nx.Graph) -> Tensor:
    """Create a simple type-aware feature vector for heterogeneous nodes."""
    attrs = graph.nodes.get(node_id, {})
    node_type = str(attrs.get("type") or attrs.get("label") or "unknown").lower()
    type_order = [
        "paper",
        "author",
        "dataset",
        "method",
        "model",
        "task",
        "metric",
        "claim",
        "keyword",
        "unknown",
    ]
    index = type_order.index(node_type) if node_type in type_order else len(type_order) - 1
    vector = torch.zeros(len(type_order), dtype=torch.float)
    vector[index] = 1.0
    return vector


def _structural_features(node_id: str, graph: nx.Graph) -> Tensor:
    """Append simple degree-based structural features."""
    degree = float(graph.degree(node_id))
    in_degree = float(graph.in_degree(node_id)) if hasattr(graph, "in_degree") else degree
    out_degree = float(graph.out_degree(node_id)) if hasattr(graph, "out_degree") else degree
    return torch.tensor([degree, in_degree, out_degree], dtype=torch.float)


def _metadata_features(node_id: str, graph: nx.Graph) -> Tensor:
    """Append optional metadata-derived features when present."""
    attrs = graph.nodes.get(node_id, {})
    properties = {
        **attrs,
        **attrs.get("properties", {})
    }
    if not isinstance(properties, dict):
        properties = {}

    feature_values: list[float] = []
    for key in ("year", "citation_count", "reference_count"):
        value = properties.get(key)
        if isinstance(value, (int, float)):
            feature_values.append(float(value))
        else:
            feature_values.append(0.0)
    return torch.tensor(feature_values, dtype=torch.float)


def build_node_features(graph: nx.Graph) -> tuple[dict[str, int], Tensor]:
    """Create a node-id to integer index mapping and a feature matrix."""
    node_ids = list(graph.nodes())
    mapping = {node_id: index for index, node_id in enumerate(node_ids)}
    if not node_ids:
        return mapping, torch.empty((0, 0), dtype=torch.float)

    features = []
    for node_id in node_ids:
        type_feature = _node_type_one_hot(node_id, graph)
        structural = _structural_features(node_id, graph)
        metadata = _metadata_features(node_id, graph)
        features.append(torch.cat([type_feature, structural, metadata], dim=0))

    feature_tensor = torch.stack(features, dim=0)

    feature_tensor = (
        feature_tensor - feature_tensor.mean(dim=0)
    ) / (
        feature_tensor.std(dim=0) + 1e-8
    )

    return mapping, feature_tensor


def build_edge_index(graph: nx.Graph, mapping: dict[str, int]) -> Tensor:
    """Create a PyTorch Geometric edge_index tensor from the graph."""
    edges = []
    for source, target in graph.edges():
        if source in mapping and target in mapping:
            edges.append((mapping[source], mapping[target]))

    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _sample_negative_edges(
    graph: nx.Graph,
    mapping: dict[str,int],
    num_samples:int
) -> Tensor:

    negatives = []

    nodes = list(mapping.keys())

    rng = np.random.default_rng(0)

    attempts = 0
    max_attempts = num_samples * 20

    while len(negatives) < num_samples and attempts < max_attempts:

        src, dst = rng.choice(nodes, 2, replace=False)
        attempts += 1

        if not graph.has_edge(src, dst):
            negatives.append(
                [
                    mapping[src],
                    mapping[dst]
                ]
            )

    if not negatives:
        return torch.empty((2,0), dtype=torch.long)

    return torch.tensor(
        negatives,
        dtype=torch.long
    ).t().contiguous()


def build_link_prediction_data(
    graph: nx.Graph,
    *,
    mapping: Optional[dict[str, int]] = None,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
) -> tuple[Data, dict[str, Any]]:
    """Build a PyTorch Geometric Data object with train/validation/test splits."""
    if mapping is None:
        mapping, _ = build_node_features(graph)

    edge_index = build_edge_index(graph, mapping)
    positive_edges = edge_index.t().tolist() if edge_index.numel() > 0 else []

    rng = np.random.default_rng(0)
    shuffled = rng.permutation(len(positive_edges)) if positive_edges else np.array([], dtype=int)

    split_test = max(1, int(len(positive_edges) * test_ratio)) if positive_edges else 0
    split_val = max(1, int(len(positive_edges) * val_ratio)) if positive_edges else 0

    test_edges = [positive_edges[idx] for idx in shuffled[:split_test]]
    val_edges = [positive_edges[idx] for idx in shuffled[split_test : split_test + split_val]]
    train_edges = [positive_edges[idx] for idx in shuffled[split_test + split_val :]]

    _, features = build_node_features(graph)
    data = Data(
        x=features,
        edge_index=edge_index,
        train_pos_edge_index=torch.tensor(train_edges, dtype=torch.long).t().contiguous() if train_edges else torch.empty((2, 0), dtype=torch.long),
        val_pos_edge_index=torch.tensor(val_edges, dtype=torch.long).t().contiguous() if val_edges else torch.empty((2, 0), dtype=torch.long),
        test_pos_edge_index=torch.tensor(test_edges, dtype=torch.long).t().contiguous() if test_edges else torch.empty((2, 0), dtype=torch.long),
    )
    return data, {"mapping": mapping, "train_edges": train_edges, "val_edges": val_edges, "test_edges": test_edges}


def train_link_prediction_model(
    data: Data,
    graph: nx.Graph,
    mapping: dict[str, int],
    *,
    epochs: int = 5,
    lr: float = 0.001,
) -> GraphSAGELinkPredictor:
    """Train a simple GraphSAGE link-prediction model on the prepared data."""
    model = GraphSAGELinkPredictor(in_channels=data.x.size(1))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(max(1, epochs)):
        model.train()
        optimizer.zero_grad()

        node_embeddings = model(data.x, data.edge_index)
        pos_edges = data.train_pos_edge_index
        if pos_edges.numel() == 0:
            pos_edges = torch.empty((2, 0), dtype=torch.long)

        num_samples = max(1, pos_edges.size(1))
        neg_edges = _sample_negative_edges(
            graph,
            mapping,
            num_samples
        )
        if pos_edges.size(1) > 0:
            edge_pairs = torch.cat([pos_edges, neg_edges], dim=1)
            labels = torch.cat([torch.ones(pos_edges.size(1)), torch.zeros(neg_edges.size(1))], dim=0).float()
        else:
            edge_pairs = neg_edges
            labels = torch.zeros(neg_edges.size(1), dtype=torch.float)

        src = edge_pairs[0]
        dst = edge_pairs[1]
        pair_emb = torch.cat([node_embeddings[src], node_embeddings[dst]], dim=1)
        logits = model.out(pair_emb).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    return model


def score_candidates(
    graph: nx.Graph,
    model: GraphSAGELinkPredictor,
    mapping: dict[str, int],
    candidates: Optional[list[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    """Score candidate missing edges with the trained GraphSAGE model."""
    if graph is None or model is None:
        return []

    _, features = build_node_features(graph)
    edge_index = build_edge_index(graph, mapping)
    model.eval()
    with torch.no_grad():
        embeddings = model(features, edge_index)

    scored_candidates: list[dict[str, Any]] = []
    for candidate in candidates or []:
        src_idx = mapping.get(candidate.get("source_node"))
        dst_idx = mapping.get(candidate.get("target_node"))
        if src_idx is None or dst_idx is None:
            continue

        pair_emb = torch.cat([embeddings[src_idx], embeddings[dst_idx]], dim=0).unsqueeze(0)
        raw_score = model.out(pair_emb).item()

        score = float(
            torch.sigmoid(
                torch.tensor(raw_score)
            )
        )
        updated = dict(candidate)
        updated["graphsage_score"] = round(float(score), 4)
        scored_candidates.append(updated)

    return scored_candidates


def save_predictions(predictions: list[dict[str, Any]], output_path: Union[str, Path]) -> Path:
    """Persist GraphSAGE predictions to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(predictions, handle, indent=2, ensure_ascii=False)
    return path


def preprocess_and_train(
    graph_path: Optional[Union[str, Path]] = None,
    *,
    candidates: Optional[list[dict[str, Any]]] = None,
    output_path: Optional[Union[str, Path]] = None,
    graph_obj: Optional[nx.Graph] = None,
) -> dict[str, Any]:
    """Run the full preprocessing and training pipeline and save predictions."""
    graph = load_graph(graph_path, graph_obj=graph_obj)
    mapping, features = build_node_features(graph)
    edge_index = build_edge_index(graph, mapping)

    data, split_info = build_link_prediction_data(graph, mapping=mapping)
    model = train_link_prediction_model(data,graph,mapping)

    print("Candidates received:", len(candidates))

    scored_candidates = score_candidates(
        graph,
        model,
        mapping,
        candidates or []
    )

    if output_path is None:
        output_path = Path("data/processed/graphsage_predictions.json")
    save_predictions(scored_candidates, output_path)

    return {
        "mapping": mapping,
        "features_shape": list(features.shape),
        "edge_index_shape": list(edge_index.shape),
        "split_info": split_info,
        "predictions_path": str(output_path),
        "prediction_count": len(scored_candidates),
    }


def main() -> None:

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s"
    )

    with open(
        "data/processed/motif_candidates.json",
        "r",
        encoding="utf-8"
    ) as file:
        candidates = json.load(file)


    result = preprocess_and_train(
        candidates=candidates
    )


    print(
        json.dumps(
            result,
            indent=2
        )
    )


if __name__ == "__main__":
    main()

