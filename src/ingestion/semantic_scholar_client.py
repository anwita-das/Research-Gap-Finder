"""Semantic Scholar API client for the Research Knowledge Graph Navigator."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from typing import Final

import requests
from dotenv import load_dotenv


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class SemanticScholarClient:
    """Client for interacting with the Semantic Scholar API.

    Supports authenticated requests when an API key is available and
    unauthenticated requests otherwise.
    """

    API_BASE_URL: Final = "https://api.semanticscholar.org/graph/v1"
    DEFAULT_TIMEOUT: int = 30
    MAX_RETRIES: Final = 5
    BACKOFF_BASE: Final = 1.0
    SEARCH_ENDPOINT: Final = "paper/search"

    def __init__(self, api_key: Optional[str] = None) -> None:
        """Initialize the Semantic Scholar client.

        Args:
            api_key: Optional API key to use for authenticated requests.
                If not provided, the client will attempt to read the key
                from the environment using python-dotenv.
        """
        self._load_environment()
        self.api_key = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        self.session = requests.Session()
        self.session.headers.update(self._default_headers())
        logger.debug("Initialized SemanticScholarClient: authenticated=%s", bool(self.api_key))

    @staticmethod
    def _load_environment() -> None:
        """Load environment variables from a .env file if it exists."""
        project_root = Path(__file__).resolve().parents[1]
        dotenv_path = project_root / ".env"

        if dotenv_path.exists():
            load_dotenv(dotenv_path)
            logger.debug("Loaded environment variables from %s", dotenv_path)
        else:
            load_dotenv()
            logger.debug("Loaded environment variables from default locations")

    def _default_headers(self) -> Dict[str, str]:
        """Return the default headers used for all requests."""
        headers = {
            "Accept": "application/json",
            "User-Agent": "ResearchKnowledgeGraphNavigator/1.0",
        }

        if self.api_key:
            headers["x-api-key"] = self.api_key
            logger.debug("Configured authenticated headers for Semantic Scholar API")
        else:
            logger.debug("Configured unauthenticated headers for Semantic Scholar API")

        return headers

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self.session.close()
        logger.debug("Closed SemanticScholarClient session")

    def _build_url(self, path: str) -> str:
        """Build a full API URL for the given endpoint path."""
        return f"{self.API_BASE_URL.rstrip('/')}/{path.lstrip('/')}"

    def request(self, method: str, path: str, **kwargs) -> requests.Response:
        """Send a request to the Semantic Scholar API.

        This helper method is intentionally lightweight to allow future
        expansion of query building, retries, and response handling.
        """
        url = self._build_url(path)
        timeout = kwargs.pop("timeout", self.DEFAULT_TIMEOUT)
        raise_for_status = kwargs.pop("raise_for_status", True)
        logger.debug("Sending %s request to %s with kwargs=%s timeout=%s", method, url, kwargs, timeout)
        response = self.session.request(method, url, timeout=timeout, **kwargs)
        if raise_for_status:
            try:
                response.raise_for_status()
            except requests.HTTPError:
                logger.error(
                    "Semantic Scholar request failed: %s %s; status=%s; body=%s",
                    method,
                    url,
                    response.status_code,
                    response.text,
                )
                raise

        logger.debug("Received response %s from %s", response.status_code, url)
        return response

    def _should_retry(self, status_code: int) -> bool:
        """Return True when a request should be retried."""
        return status_code in {429, 500, 502, 503, 504}

    def _perform_request_with_retries(self, method: str, path: str, **kwargs) -> requests.Response:
        """Perform a request with retries for transient failures."""

        for attempt in range(1, self.MAX_RETRIES + 1):
            response = self.request(method, path, raise_for_status=False, **kwargs)
            if response.status_code < 400:
                return response

            if not self._should_retry(response.status_code):
                response.raise_for_status()
                return response

            if attempt == self.MAX_RETRIES:
                logger.warning(
                    "Exceeded retry attempts for %s %s with status %s",
                    method,
                    path,
                    response.status_code,
                )
                response.raise_for_status()
                return response

            retry_after = response.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                delay = float(retry_after)
            else:
                delay = self.BACKOFF_BASE * (2 ** (attempt - 1))

            logger.warning(
                "Retrying %s %s after status %s; attempt=%s delay=%.1fs",
                method,
                path,
                response.status_code,
                attempt,
                delay,
            )
            time.sleep(delay)

        raise RuntimeError("Unexpected retry loop exit")

    def _map_paper_to_schema(self, raw_paper: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a raw Semantic Scholar paper into the project's paper schema.

        Args:
            raw_paper: The raw paper payload returned by the Semantic Scholar API.

        Returns:
            A dictionary containing the mapped paper schema fields.
        """
        if not isinstance(raw_paper, dict):
            raw_paper = {}

        external_ids = raw_paper.get("externalIds", {})
        if not isinstance(external_ids, dict):
            external_ids = {}

        authors = raw_paper.get("authors", [])
        if not isinstance(authors, list):
            authors = []

        mapped_authors: List[str] = []
        for author in authors:
            if isinstance(author, dict):
                name = author.get("name")
                if name:
                    mapped_authors.append(name)
            elif isinstance(author, str):
                mapped_authors.append(author)
            else:
                continue

        fields_of_study = raw_paper.get("fieldsOfStudy", [])
        if not isinstance(fields_of_study, list):
            fields_of_study = []

        citations = raw_paper.get("citations", [])
        if not isinstance(citations, list):
            citations = []

        mapped_citations: List[str] = []
        for citation in citations:
            if isinstance(citation, dict):
                paper_id = citation.get("paperId")
                if paper_id:
                    mapped_citations.append(paper_id)
            elif isinstance(citation, str):
                mapped_citations.append(citation)
            else:
                continue

        references = raw_paper.get("references", [])
        if not isinstance(references, list):
            references = []

        mapped_references: List[str] = []
        for reference in references:
            if isinstance(reference, dict):
                paper_id = reference.get("paperId")
                if paper_id:
                    mapped_references.append(paper_id)
            elif isinstance(reference, str):
                mapped_references.append(reference)
            else:
                continue

        return {
            "paper_id": "semantic_" + raw_paper.get("paperId", ""),
            "title": raw_paper.get("title", ""),
            "abstract": raw_paper.get("abstract", ""),
            "authors": mapped_authors,
            "year": raw_paper.get("year", 0) or 0,
            "venue": raw_paper.get("venue", ""),
            "doi": external_ids.get("DOI", ""),
            "arxiv_id": external_ids.get("ArXiv", ""),
            "semantic_scholar_id": raw_paper.get("paperId", ""),
            "keywords": [],
            "fields_of_study": fields_of_study,
            "citations": mapped_citations,
            "references": mapped_references,
            "url": raw_paper.get("url", ""),
            "pdf_url": (
                raw_paper.get("openAccessPdf", {}).get("url", "")
                if isinstance(raw_paper.get("openAccessPdf"), dict)
                else ""
            ),
            "source": ["semantic_scholar"],
            "metadata": {
                "citation_count": raw_paper.get("citationCount", 0) or 0,
                "reference_count": raw_paper.get("referenceCount", 0) or 0,
                "publication_date": raw_paper.get("publicationDate", ""),
                "updated_at": datetime.now().isoformat(),
            },
        }

    def save_papers(self, papers: List[Dict[str, Any]], query: str) -> Path:
        """Save mapped papers to disk in JSON format.

        Args:
            papers: A list of mapped paper dictionaries.
            query: The search query used to retrieve the papers.

        Returns:
            The path to the saved JSON file.
        """

        if not isinstance(papers, list):
            raise TypeError("papers must be a list")

        if not query.strip():
            raise ValueError("query must not be empty")

        sanitized_query = re.sub(r"\s+", "_", query.strip())
        sanitized_query = re.sub(r"[^A-Za-z0-9_\-]", "", sanitized_query)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"semantic_scholar_{sanitized_query}_{timestamp}.json"

        project_root = Path(__file__).resolve().parents[2]
        destination_dir = project_root / "data" / "raw" / "semantic_scholar"
        destination_dir.mkdir(parents=True, exist_ok=True)

        destination_path = destination_dir / filename
        with destination_path.open("w", encoding="utf-8") as file_handle:
            json.dump(papers, file_handle, indent=4, ensure_ascii=False,  sort_keys=False,)

        logger.info("Saved %s papers to %s", len(papers), destination_path)
        return destination_path

    def get_paper_by_identifier(
        self,
        doi: Optional[str] = None,
        arxiv_id: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve the best matching paper using DOI, ArXiv ID, or an exact title.

        Lookup order is DOI, ArXiv ID, and finally an exact title search. The
        method returns the mapped paper schema object for the first successful
        match, or None if no matching paper is found.
        """
        candidates: List[tuple[str, str]] = []

        if doi and doi.strip():
            candidates.append((doi.strip(), "DOI"))
        if arxiv_id and arxiv_id.strip():
            candidates.append((arxiv_id.strip(), "ArXiv"))
        if title and title.strip():
            candidates.append((title.strip(), "title"))

        if not candidates:
            logger.warning("No lookup values provided for Semantic Scholar paper lookup")
            return None

        fields = (
            "paperId,"
            "title,"
            "abstract,"
            "authors,"
            "year,"
            "venue,"
            "externalIds,"
            "citationCount,"
            "referenceCount,"
            "references,"
            "citations,"
            "fieldsOfStudy,"
            "publicationDate,"
            "openAccessPdf,"
            "url"
        )

        for value, lookup_type in candidates:
            if lookup_type == "DOI":
                query = f"DOI:{value}"
            elif lookup_type == "ArXiv":
                query = f"arXiv:{value}"
            else:
                query = f'"{value}"'

            logger.info(
                "Looking up Semantic Scholar paper by %s using query=%s",
                lookup_type,
                query,
            )

            try:
                response = self._perform_request_with_retries(
                    "GET",
                    self.SEARCH_ENDPOINT,
                    params={
                        "query": query,
                        "limit": 1,
                        "fields": fields,
                    },
                    timeout=self.DEFAULT_TIMEOUT,
                )
            except requests.HTTPError as exc:
                status_code = exc.response.status_code if exc.response else None
                if status_code == 404:
                    logger.warning(
                        "No Semantic Scholar paper found for %s lookup using %s",
                        lookup_type,
                        value,
                    )
                    continue

                logger.warning(
                    "Semantic Scholar paper lookup by %s failed for %s: %s",
                    lookup_type,
                    value,
                    exc,
                )
                continue
            except requests.RequestException as exc:
                logger.warning(
                    "Semantic Scholar paper lookup by %s failed for %s: %s",
                    lookup_type,
                    value,
                    exc,
                )
                continue

            data = response.json()
            if not isinstance(data, dict):
                logger.warning(
                    "Unexpected response format from Semantic Scholar API during %s lookup",
                    lookup_type,
                )
                continue

            batch: List[Dict[str, Any]] = []
            if isinstance(data.get("data"), list):
                batch = [paper for paper in data.get("data", []) if isinstance(paper, dict)]
            elif isinstance(data, dict) and data.get("paperId"):
                batch = [data]

            if not batch:
                logger.warning(
                    "No Semantic Scholar paper found for %s lookup using %s",
                    lookup_type,
                    value,
                )
                continue

            mapped_paper = self._map_paper_to_schema(batch[0])
            logger.info(
                "Resolved Semantic Scholar paper for %s lookup: %s",
                lookup_type,
                mapped_paper.get("title", ""),
            )
            return mapped_paper

        logger.warning("No Semantic Scholar paper matched the provided identifiers")
        return None

    def search_papers(self, query: str, limit: int = 100) -> Dict[str, Any]:
        """Search for papers using the Semantic Scholar Graph API.

        Args:
            query: Search query string.
            limit: Maximum number of papers to retrieve.

        Returns:
            A dictionary containing the search metadata and the retrieved papers.
        Raises:
            ValueError: If the query is empty or limit is not positive.
            requests.HTTPError: If the request fails with a non-retryable HTTP error.
        """
        if not query.strip():
            raise ValueError("Search query must not be empty")

        if limit <= 0:
            raise ValueError("Limit must be a positive integer")

        params = {
            "query": query,
            "limit": min(limit, 100),
            "offset": 0,
            "fields": (
                "paperId,"
                "title,"
                "abstract,"
                "authors,"
                "year,"
                "venue,"
                "externalIds,"
                "citationCount,"
                "referenceCount,"
                "references,"
                "citations,"
                "fieldsOfStudy,"
                "publicationDate,"
                "openAccessPdf,"
                "url"
            ),
        }

        results: List[Dict[str, Any]] = []
        total = None

        while len(results) < limit:
            response = self._perform_request_with_retries(
                "GET",
                self.SEARCH_ENDPOINT,
                params=params,
                timeout=self.DEFAULT_TIMEOUT,
            )
            data = response.json()

            if not isinstance(data, dict):
                raise ValueError("Unexpected response format from Semantic Scholar API")

            batch = data.get("data", [])
            if not batch:
                break
            if not isinstance(batch, list):
                raise ValueError("Unexpected search result format from Semantic Scholar API")

            mapped_batch = [
                self._map_paper_to_schema(paper)
                for paper in batch
            ]

            results.extend(mapped_batch)
            if total is None:
                total = data.get("total")

            if len(results) >= limit:
                break

            if total is not None and len(results) >= total:
                break

            params["offset"] += len(batch)
            params["limit"] = min(limit - len(results), 100)

        return {
            "query": query,
            "limit": limit,
            "returned": len(results),
            "total": total,
            "data": results,
        }

def main() -> None:
    """Run a simple test of the Semantic Scholar ingestion pipeline."""

    client = SemanticScholarClient()

    try:
        query = input("Enter research topic: ").strip()

        if not query:
            print("Error: Research topic cannot be empty.")
            return

        limit_input = input("Maximum number of papers (default 100): ").strip()

        if limit_input:
            limit = int(limit_input)
        else:
            limit = 100

        result = client.search_papers(query=query, limit=limit)

        output_path = client.save_papers(
            papers=result["data"],
            query=query,
        )

        print("\n" + "=" * 50)
        print("Semantic Scholar Search Completed")
        print("=" * 50)
        print(f"Research Topic      : {query}")
        print(f"Papers Retrieved    : {result['returned']}")
        print(f"Total Papers Found  : {result['total']}")
        print(f"Output File         : {output_path}")
        print("=" * 50)

    except ValueError as error:
        print(f"Input Error: {error}")

    except requests.HTTPError as error:
        print(f"HTTP Error: {error}")

    except Exception as error:
        print(f"Unexpected Error: {error}")

    finally:
        client.close()


if __name__ == "__main__":
    main()