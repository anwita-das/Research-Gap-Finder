"""
Visualize the knowledge graph using PyVis.
"""

from pathlib import Path

from pyvis.network import Network

from src.knowledge_graph.builder import GraphBuilder
from src.knowledge_graph.build_graph import (
    _load_papers,
    _load_entity_files,
    _add_paper_to_builder,
    _add_entities_to_builder,
)


def build_graph():
    """
    Build the NetworkX graph from stored papers and entity files.
    """

    project_root = Path(__file__).resolve().parents[2]

    merged_dir = project_root / "data" / "processed" / "merged"
    entities_dir = project_root / "data" / "processed" / "entities"

    builder = GraphBuilder()

    # Add papers
    for paper in _load_papers(merged_dir):
        _add_paper_to_builder(builder, paper)

    # Add extracted entities
    for entity_file in _load_entity_files(entities_dir):
        _add_entities_to_builder(builder, entity_file)

    return builder.build_graph()


def visualize_graph():

    graph = build_graph()

    net = Network(
        height="900px",
        width="100%",
        bgcolor="#ffffff",
        font_color="black",
        directed=True,
    )

    graph = graph.copy()

    for node_id, attrs in list(graph.nodes(data=True)):
        if attrs.get("label") in {"Author", "Venue"}:
            graph.remove_node(node_id)

    net.from_nx(graph)
    for node in net.nodes:
        label = node.get("label", "")
        if label == "Paper":
            node["label"] = node.get("title", node["id"])
        else:
            node["label"] = (
                node.get("name")
                or node.get("text")      # for Claim nodes
                or node["id"]
            )
    for edge in net.edges:
        relation = edge.get("relation")
        if relation:
            edge["label"] = relation
            edge["title"] = relation  # optional: show on hover

    output_dir = Path(__file__).resolve().parents[2] / "data" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "knowledge_graph.html"

    net.save_graph(str(output_file))

    print("\nVisualization saved to:")
    print(output_file.resolve())


if __name__ == "__main__":
    visualize_graph()