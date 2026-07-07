import json
from pathlib import Path
from typing import Union
import networkx as nx
from networkx.readwrite import json_graph


class GraphExporter:
    """Export a NetworkX MultiDiGraph to GraphML, GEXF, or JSON."""

    def export_graphml(
        self,
        graph: nx.MultiDiGraph,
        output_path: Union[str, Path],
        ) -> Path:
        """Export the graph to GraphML and return the file path."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        nx.write_graphml(graph, path)
        return path

    def export_gexf(
        self,
        graph: nx.MultiDiGraph,
        output_path: Union[str, Path],
        ) -> Path:
        """Export the graph to GEXF and return the file path."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        nx.write_gexf(graph, path)
        return path

    def export_json(
        self,
        graph: nx.MultiDiGraph,
        output_path: Union[str, Path],
    ) -> Path:
        """Export the graph to JSON using node-link format and return
        the file path."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json_graph.node_link_data(graph)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
        return path