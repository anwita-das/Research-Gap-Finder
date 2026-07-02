"""Orchestrate the ingestion workflow for ArXiv and openalex papers."""

from __future__ import annotations

from datetime import datetime, UTC
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.ingestion.openalex_client import OpenAlexClient

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

try:
    from src.arxiv_fetcher import fetch_papers as _fetch_papers
except ModuleNotFoundError:
    _fetch_papers = None

fetch_papers = _fetch_papers


def enrich_paper(arxiv_paper: Dict[str, Any], openalex_client: Optional[OpenAlexClient] = None) -> Dict[str, Any]:
    """Enrich an ArXiv paper with OpenAlex metadata when available.

    Args:
        arxiv_paper: The ArXiv-derived paper schema object.
        openalex_client: Optional OpenAlex client instance.

    Returns:
        A merged paper schema dictionary with ArXiv fields preserved and
        OpenAlex metadata merged in where it is non-empty.
    """
    if not isinstance(arxiv_paper, dict):
        raise TypeError("arxiv_paper must be a dictionary")

    if openalex_client is None:
        openalex_client = OpenAlexClient()

    doi = arxiv_paper.get("doi") or ""
    arxiv_id = arxiv_paper.get("arxiv_id") or ""
    title = arxiv_paper.get("title") or ""

    logger.info("Enriching paper %s using OpenAlex", arxiv_paper.get("paper_id", title))
    openalex_paper = openalex_client.get_paper_by_identifier(
        doi=doi or None,
        arxiv_id=arxiv_id or None,
        title=title or None,
    )

    if openalex_paper is None:
        logger.warning("No OpenAlex metadata found for paper %s", arxiv_paper.get("paper_id", title))
        return arxiv_paper

    return merge_paper_metadata(arxiv_paper, openalex_paper)


def merge_paper_metadata(arxiv_paper: Dict[str, Any], openalex_paper: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge ArXiv and OpenAlex paper data without overwriting valid values.

    ArXiv values take precedence for title, abstract, pdf_url, and categories
    while OpenAlex values take precedence for citation-related fields and
    OpenAlex identifiers.
    """
    if not isinstance(arxiv_paper, dict):
        raise TypeError("arxiv_paper must be a dictionary")

    if openalex_paper is None:
        return arxiv_paper

    if not isinstance(openalex_paper, dict):
        raise TypeError("openalex_paper must be a dictionary")

    merged = dict(arxiv_paper)
    merged["paper_id"] = arxiv_paper.get(
        "paper_id",
        openalex_paper.get("paper_id", "")
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
            merged[field] = _prefer_existing(arxiv_paper.get(field), openalex_paper.get(field))
        else:
            merged[field] = _prefer_existing(openalex_paper.get(field), arxiv_paper.get(field))

    for field in ["authors", "year", "venue", "doi", "arxiv_id", "url", "source", "keywords"]:
        if field in ["authors", "year", "venue", "doi", "arxiv_id", "url", "source", "keywords"]:
            merged[field] = _prefer_existing(arxiv_paper.get(field), openalex_paper.get(field))

    for field in ["openalex_id", "citations", "references"]:
        merged[field] = _prefer_existing(openalex_paper.get(field), arxiv_paper.get(field))

    metadata = merged["metadata"]
    metadata["citation_count"] = _prefer_existing(
        openalex_paper.get("metadata", {}).get("citation_count"),
        arxiv_paper.get("metadata", {}).get("citation_count"),
    )
    metadata["reference_count"] = _prefer_existing(
        openalex_paper.get("metadata", {}).get("reference_count"),
        arxiv_paper.get("metadata", {}).get("reference_count"),
    )
    metadata["publication_date"] = _prefer_existing(
        arxiv_paper.get("metadata", {}).get("publication_date"),
        openalex_paper.get("metadata", {}).get("publication_date"),
    )
    metadata["updated_at"] = datetime.now(UTC).isoformat()

    sources = set()

    arxiv_sources = arxiv_paper.get("source", [])
    openalex_sources = openalex_paper.get("source", [])

    if isinstance(arxiv_sources, str):
        sources.add(arxiv_sources)
    else:
        sources.update(arxiv_sources)

    if isinstance(openalex_sources, str):
        sources.add(openalex_sources)
    else:
        sources.update(openalex_sources)

    merged["source"] = sorted(sources)

    return merged


def process_topic(research_topic: str, max_papers: int, openalex_client: Optional[OpenAlexClient] = None) -> List[Dict[str, Any]]:
    """Process a research topic by fetching ArXiv papers and enriching them.

    Args:
        research_topic: The research topic string to search on ArXiv.
        max_papers: Maximum number of papers to retrieve and enrich.
        openalex_client: Optional OpenAlex client instance.

    Returns:
        A list of merged paper schema objects.
    """
    if not isinstance(research_topic, str) or not research_topic.strip():
        raise ValueError("research_topic must not be empty")

    if max_papers <= 0:
        raise ValueError("max_papers must be a positive integer")

    if openalex_client is None:
        openalex_client = OpenAlexClient()

    logger.info("Processing topic '%s' with max_papers=%s", research_topic, max_papers)

    if fetch_papers is None:
        raise ModuleNotFoundError("ArXiv fetcher dependency is unavailable")

    arxiv_papers = fetch_papers(research_topic, total=max_papers)

    enriched_papers: List[Dict[str, Any]] = []
    for i, arxiv_paper in enumerate(arxiv_papers, start=1):
        print(f"Enriching paper {i}: {arxiv_paper.get('title')}")
        enriched_paper = enrich_paper(arxiv_paper, openalex_client=openalex_client)
        print(f"Finished paper {i}")
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

        openalex_client = OpenAlexClient()
        try:
            papers = process_topic(research_topic, max_papers, openalex_client=openalex_client)
            enriched_count = sum(1 for paper in papers if paper.get("openalex_id"))
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
            openalex_client.close()
    except Exception as exc:  # pragma: no cover - CLI guard
        logger.exception("Ingestion pipeline failed: %s", exc)
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()
