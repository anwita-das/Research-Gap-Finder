"""Command-line interface for the semantic search subsystem."""

from __future__ import annotations

import argparse
from typing import Optional

from .config import SemanticSearchConfig
from .search_service import SearchService
from .types import SearchQuery


def main(query: Optional[str] = None) -> None:
    """Run a semantic search query from the command line."""
    parser = argparse.ArgumentParser(description="Semantic search over the knowledge graph")
    parser.add_argument("query", nargs="?", default=query)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args([] if query is not None else None)

    if args.query is None:
        raise SystemExit("Please provide a query")

    config = SemanticSearchConfig(top_k=args.top_k).resolve_paths()
    service = SearchService(config=config)
    service.load_graph(config.graph_path)
    results = service.search(SearchQuery(text=args.query, top_k=args.top_k))

    print(f"Semantic search results for: {args.query}")
    for index, result in enumerate(results, start=1):
        print(result)
        print(f"{index}. {result.title} (score={result.score:.3f})")
        if result.graph_context:
            context = result.graph_context
            details = []
            if context.authors:
                details.append(f"authors: {', '.join(context.authors[:3])}")
            if context.methods:
                details.append(f"methods: {', '.join(context.methods[:3])}")
            if context.models:
                details.append(f"models: {', '.join(context.models[:3])}")
            if context.datasets:
                details.append(f"datasets: {', '.join(context.datasets[:3])}")
            if context.tasks:
                details.append(f"tasks: {', '.join(context.tasks[:3])}")
            if context.citation_neighbors:
                details.append(f"citations: {', '.join(context.citation_neighbors[:3])}")
            if details:
                print("   " + " | ".join(details))


if __name__ == "__main__":
    main()
