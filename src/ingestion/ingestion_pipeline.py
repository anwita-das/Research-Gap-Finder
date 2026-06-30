"""Orchestrate the ingestion workflow for ArXiv and Semantic Scholar papers."""

from __future__ import annotations

from datetime import datetime
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.ingestion.semantic_scholar_client import SemanticScholarClient

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

try:
    from src.arxiv_fetcher import fetch_papers as _fetch_papers
except ModuleNotFoundError:
    _fetch_papers = None

fetch_papers = _fetch_papers


def enrich_paper(arxiv_paper: Dict[str, Any], semantic_client: Optional[SemanticScholarClient] = None) -> Dict[str, Any]:
    """Enrich an ArXiv paper with Semantic Scholar metadata when available.

    Args:
        arxiv_paper: The ArXiv-derived paper schema object.
        semantic_client: Optional Semantic Scholar client instance.

    Returns:
        A merged paper schema dictionary with ArXiv fields preserved and
        Semantic Scholar metadata merged in where it is non-empty.
    """
    if not isinstance(arxiv_paper, dict):
        raise TypeError("arxiv_paper must be a dictionary")

    if semantic_client is None:
        semantic_client = SemanticScholarClient()

    doi = arxiv_paper.get("doi") or ""
    arxiv_id = arxiv_paper.get("arxiv_id") or ""
    title = arxiv_paper.get("title") or ""

    logger.info("Enriching paper %s using Semantic Scholar", arxiv_paper.get("paper_id", title))
    semantic_paper = semantic_client.get_paper_by_identifier(
        doi=doi or None,
        arxiv_id=arxiv_id or None,
        title=title or None,
    )

    if semantic_paper is None:
        logger.warning("No Semantic Scholar metadata found for paper %s", arxiv_paper.get("paper_id", title))
        return arxiv_paper

    return merge_paper_metadata(arxiv_paper, semantic_paper)


def merge_paper_metadata(arxiv_paper: Dict[str, Any], semantic_paper: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge ArXiv and Semantic Scholar paper data without overwriting valid values.

    ArXiv values take precedence for title, abstract, pdf_url, and categories
    while Semantic Scholar values take precedence for citation-related fields and
    semantic scholar identifiers.
    """
    if not isinstance(arxiv_paper, dict):
        raise TypeError("arxiv_paper must be a dictionary")

    if semantic_paper is None:
        return arxiv_paper

    if not isinstance(semantic_paper, dict):
        raise TypeError("semantic_paper must be a dictionary")

    merged = dict(arxiv_paper)
    merged["paper_id"] = arxiv_paper.get(
        "paper_id",
        semantic_paper.get("paper_id", "")
    )
    merged["metadata"] = dict(arxiv_paper.get("metadata", {}))

    def _prefer_existing(existing: Any, incoming: Any) -> Any:
        if existing not in (None, "", [], {}):
            return existing
        return incoming

    preferred_fields = {
        "title": "title",
        "abstract": "abstract",
        "pdf_url": "pdf_url",
        "fields_of_study": "fields_of_study",
    }

    for field in preferred_fields:
        if field in {"title", "abstract", "pdf_url"}:
            merged[field] = _prefer_existing(arxiv_paper.get(field), semantic_paper.get(field))
        else:
            merged[field] = _prefer_existing(semantic_paper.get(field), arxiv_paper.get(field))

    for field in ["authors", "year", "venue", "doi", "arxiv_id", "url", "source", "keywords"]:
        if field in ["authors", "year", "venue", "doi", "arxiv_id", "url", "source", "keywords"]:
            merged[field] = _prefer_existing(arxiv_paper.get(field), semantic_paper.get(field))

    for field in ["semantic_scholar_id", "citations", "references"]:
        merged[field] = _prefer_existing(semantic_paper.get(field), arxiv_paper.get(field))

    metadata = merged["metadata"]
    metadata["citation_count"] = _prefer_existing(
        semantic_paper.get("metadata", {}).get("citation_count"),
        arxiv_paper.get("metadata", {}).get("citation_count"),
    )
    metadata["reference_count"] = _prefer_existing(
        semantic_paper.get("metadata", {}).get("reference_count"),
        arxiv_paper.get("metadata", {}).get("reference_count"),
    )
    metadata["publication_date"] = _prefer_existing(
        arxiv_paper.get("metadata", {}).get("publication_date"),
        semantic_paper.get("metadata", {}).get("publication_date"),
    )
    metadata["updated_at"] = datetime.now().isoformat()

    sources = set()

    arxiv_sources = arxiv_paper.get("source", [])
    semantic_sources = semantic_paper.get("source", [])

    if isinstance(arxiv_sources, str):
        sources.add(arxiv_sources)
    else:
        sources.update(arxiv_sources)

    if isinstance(semantic_sources, str):
        sources.add(semantic_sources)
    else:
        sources.update(semantic_sources)

    merged["source"] = list(sources)

    return merged


def process_topic(research_topic: str, max_papers: int, semantic_client: Optional[SemanticScholarClient] = None) -> List[Dict[str, Any]]:
    """Process a research topic by fetching ArXiv papers and enriching them.

    Args:
        research_topic: The research topic string to search on ArXiv.
        max_papers: Maximum number of papers to retrieve and enrich.
        semantic_client: Optional Semantic Scholar client instance.

    Returns:
        A list of merged paper schema objects.
    """
    if not isinstance(research_topic, str) or not research_topic.strip():
        raise ValueError("research_topic must not be empty")

    if max_papers <= 0:
        raise ValueError("max_papers must be a positive integer")

    if semantic_client is None:
        semantic_client = SemanticScholarClient()

    logger.info("Processing topic '%s' with max_papers=%s", research_topic, max_papers)

    if fetch_papers is None:
        raise ModuleNotFoundError("ArXiv fetcher dependency is unavailable")

    arxiv_papers = fetch_papers(research_topic, total=max_papers)

    enriched_papers: List[Dict[str, Any]] = []
    for arxiv_paper in arxiv_papers:
        enriched_paper = enrich_paper(arxiv_paper, semantic_client=semantic_client)
        enriched_papers.append(enriched_paper)

    logger.info("Completed enrichment for %s papers", len(enriched_papers))
    return enriched_papers


def save_merged_dataset(papers: List[Dict[str, Any]], query: str) -> Path:
    """Save merged paper data to disk as a JSON dataset.

    Args:
        papers: The merged paper schema objects to persist.
        query: The research topic used to generate the dataset.

    Returns:
        The path to the saved JSON file.
    """
    if not isinstance(papers, list):
        raise TypeError("papers must be a list")

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must not be empty")

    project_root = Path(__file__).resolve().parents[2]
    destination_dir = project_root / "data" / "processed" / "merged"
    destination_dir.mkdir(parents=True, exist_ok=True)

    sanitized_query = re.sub(r"\s+", "_", query.strip())
    sanitized_query = re.sub(r"[^A-Za-z0-9_\-]", "", sanitized_query)
    output_path = destination_dir / f"merged_{sanitized_query}.json"

    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(papers, file_handle, indent=4, ensure_ascii=False)

    logger.info("Saved %s merged papers to %s", len(papers), output_path)
    return output_path


def main() -> None:
    """Run the ingestion pipeline from the command line."""
    try:
        research_topic = input("Research Topic: ").strip()
        max_papers_input = input("Maximum Papers: ").strip()

        if not research_topic:
            raise ValueError("Research Topic cannot be empty")

        max_papers = int(max_papers_input) if max_papers_input else 10

        semantic_client = SemanticScholarClient()
        try:
            papers = process_topic(research_topic, max_papers, semantic_client=semantic_client)
            enriched_count = sum(1 for paper in papers if paper.get("semantic_scholar_id"))
            arxiv_only_count = len(papers) - enriched_count
            output_path = save_merged_dataset(papers, research_topic)

            print("---------------------------------------")
            print("Topic")
            print(research_topic)
            print("Papers Retrieved")
            print(len(papers))
            print("Successfully Enriched")
            print(enriched_count)
            print("ArXiv-only Papers")
            print(arxiv_only_count)
            print("Output File")
            print(output_path)
            print("---------------------------------------")
        finally:
            semantic_client.close()
    except Exception as exc:  # pragma: no cover - CLI guard
        logger.exception("Ingestion pipeline failed: %s", exc)
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()
