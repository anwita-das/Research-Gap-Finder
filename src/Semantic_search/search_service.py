"""High-level semantic search service orchestrating indexing, retrieval, ranking, and graph enrichment."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .config import SemanticSearchConfig
from .embedding_service import EmbeddingService
from .faiss_index import FaissIndex
from .graph_connector import GraphConnector
from .graph_loader import GraphLoader
from .ranking import RankingService
from .retrieval import SemanticRetriever
from .types import PaperDocument, SearchQuery, SearchResult


class SearchService:
    """Orchestrate semantic search over the existing knowledge graph."""

    def __init__(
        self,
        config: Optional[SemanticSearchConfig] = None,
        graph_loader: Optional[GraphLoader] = None,
        embedding_service: Optional[EmbeddingService] = None,
        faiss_index: Optional[FaissIndex] = None,
        ranking_service: Optional[RankingService] = None,
        graph_connector: Optional[GraphConnector] = None,
    ) -> None:
        self.config = config or SemanticSearchConfig()
        self.graph_loader = graph_loader or GraphLoader()
        self.embedding_service = embedding_service or EmbeddingService(self.config)
        self.faiss_index = faiss_index or FaissIndex(self.config.index_path)
        self.ranking_service = ranking_service or RankingService(self.config)
        self.graph_connector = graph_connector or GraphConnector()
        self.retriever = SemanticRetriever(self.embedding_service, self.faiss_index, self.config)

    def build_index(self, documents: Optional[List[PaperDocument]] = None) -> None:
        """Build or refresh the semantic index from paper documents."""
        if documents is None:
            documents = self._documents_from_graph()
        self.retriever.build_index(documents)

    def search(self, query: SearchQuery) -> List[SearchResult]:
        documents = self._documents_from_graph()
        print("Documents:", len(documents))

        retrieved_documents = self.retriever.retrieve(query, documents)
        print("Retrieved:", len(retrieved_documents))

        candidates = [
            (
                document,
                float(document.metadata.get("semantic_similarity", 0.0)),
            )
            for document in retrieved_documents
        ]

        ranked_results = self.ranking_service.rank(candidates)
        print("Ranked:", len(ranked_results))

        enriched_results = self.graph_connector.enrich_results(ranked_results)
        print("Enriched:", len(enriched_results))

        return enriched_results
    def load_graph(self, graph_path: Optional[Path] = None) -> None:
        """Load the existing graph export from disk for read-only graph enrichment."""
        self.graph_loader.load_graph(graph_path)
        self.graph_connector.graph = self.graph_loader.graph

    def _documents_from_graph(self) -> List[PaperDocument]:
        """Create paper documents enriched with connected graph entities."""
        graph = self.graph_loader.load_graph(self.config.graph_path)
        count = 0
        for u, v, data in graph.edges(data=True):
                if data.get("relation") == "CITES":
                    print(u, "->", v)
                    count += 1
                    if count == 5:
                        break

        print("Total CITES found:", count)
        nodes = self.graph_loader.load_paper_nodes()
        print("Nodes loaded:", len(nodes))

        documents: List[PaperDocument] = []

        for node in nodes:
            paper_id = str(node.get("paper_id") or node.get("node_id") or "")
            title = str(node.get("title") or "")
            
            abstract = str(node.get("abstract") or "")
            # Skip placeholder paper nodes
            if node.get("placeholder"):
                continue

            # Skip papers without any searchable text
            if not title.strip() and not abstract.strip():
                continue
            print(node)
            

            
            metadata = {
                "year": node.get("year"),
                "citation_count": node.get("citation_count"),
                "reference_count": node.get("reference_count"),
                "source": node.get("source"),
                "connectivity": (
                    min(graph.degree(paper_id) / 20, 1.0)
                    if graph.has_node(paper_id)
                    else 0.0
                ),

                
            }

            methods = []
            datasets = []
            tasks = []
            models = []

            if graph.has_node(paper_id):
                for _, target, edge_data in graph.out_edges(paper_id, data=True):
                    relation = str(edge_data.get("relation", "")).upper()

                    target_attrs = graph.nodes[target]

                    entity_name = (
                        target_attrs.get("name")
                        or target_attrs.get("title")
                        or target_attrs.get("label")
                        or target
                    ).strip()
                

                    if relation == "USES_METHOD":
                        methods.append(entity_name)

                    elif relation == "USES_DATASET":
                        datasets.append(entity_name)

                    elif relation == "USES_MODEL":
                        models.append(entity_name)

                    elif relation == "ADDRESSES_TASK":
                        tasks.append(entity_name)
                

            text_parts = [
                title,
                abstract,
            ]

            if methods:
                text_parts.append("Methods: " + ", ".join(methods))

            if models:
                text_parts.append("Models: " + ", ".join(models))

            if datasets:
                text_parts.append("Datasets: " + ", ".join(datasets))

            if tasks:
                text_parts.append("Tasks: " + ", ".join(tasks))

            text = "\n\n".join(part for part in text_parts if part)

            documents.append(
            PaperDocument(
                paper_id=paper_id,
                title=title,
                abstract=abstract,
                text=text,
                metadata=metadata,
                entity_names=methods + models + datasets + tasks,
            )
        )

        print("Added:", paper_id, repr(title))
            

        return documents