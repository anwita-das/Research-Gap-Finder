import json
import shutil
from pathlib import Path

import numpy as np

from src.graph_analysis.graphsage_preprocess import preprocess_graph


def test_preprocess_graph_writes_expected_artifacts(tmp_path):
    graph_path = Path("data/processed/knowledge_graph.json")
    output_dir = tmp_path / "outputs"

    preprocess_graph(str(graph_path), str(output_dir))

    assert (output_dir / "node_mapping.json").exists()
    assert (output_dir / "edge_index.npy").exists()
    assert (output_dir / "node_features.npy").exists()

    with (output_dir / "node_mapping.json").open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)

    edge_index = np.load(output_dir / "edge_index.npy")
    node_features = np.load(output_dir / "node_features.npy")

    assert isinstance(mapping, dict)
    assert edge_index.ndim == 2
    assert edge_index.shape[0] == 2
    assert node_features.ndim == 2
    assert node_features.shape[0] == len(mapping)
