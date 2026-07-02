"""OpenAlex API client for the Research Knowledge Graph Navigator."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, List, Optional
from typing import Final

import requests
from dotenv import load_dotenv


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class OpenAlexClient:
    """Client for interacting with the OpenAlex API.

    Implements a small compatibility layer so that downstream code which
    expects the existing merged-paper schema remains functional.
    """

    API_BASE_URL: Final = "https://api.openalex.org"
    DEFAULT_TIMEOUT: int = 10
    MAX_RETRIES: Final = 5
    BACKOFF_BASE: Final = 1.0
    SEARCH_ENDPOINT: Final = "works"

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._load_environment()
        self.api_key = api_key or os.getenv("OPENALEX_API_KEY")
        self.session = requests.Session()
        self.session.headers.update(self._default_headers())
        logger.debug("Initialized OpenAlexClient: authenticated=%s", bool(self.api_key))

    @staticmethod
    def _load_environment() -> None:
        project_root = Path(__file__).resolve().parents[2]
        dotenv_path = project_root / ".env"

        if dotenv_path.exists():
            load_dotenv(dotenv_path)
            logger.debug("Loaded environment variables from %s", dotenv_path)
        else:
            load_dotenv()
            logger.debug("Loaded environment variables from default locations")

    def _default_headers(self) -> Dict[str, str]:
        return {
            "Accept": "application/json",
            "User-Agent": "ResearchKnowledgeGraphNavigator/1.0",
        }

    def close(self) -> None:
        self.session.close()
        logger.debug("Closed OpenAlexClient session")

    def _build_url(self, path: str) -> str:
        return f"{self.API_BASE_URL.rstrip('/')}/{path.lstrip('/') }"

    def request(self, method: str, path: str, **kwargs) -> requests.Response:
        url = self._build_url(path)
        timeout = kwargs.pop("timeout", self.DEFAULT_TIMEOUT)
        raise_for_status = kwargs.pop("raise_for_status", True)
        logger.debug("Sending %s request to %s with kwargs=%s timeout=%s", method, url, kwargs, timeout)
        try:
            print("URL:", url)
            print("HEADERS:", self.session.headers)
            print("PARAMS:", kwargs.get("params"))
            params = kwargs.setdefault("params", {})

            if self.api_key:
                params["api_key"] = self.api_key

            email = os.getenv("OPENALEX_EMAIL")
            if email:
                params["mailto"] = email
            response = self.session.request(
                method,
                url,
                timeout=timeout,
                **kwargs,
            )
        except requests.RequestException:
            logger.exception("OpenAlex request failed")
            raise
        if raise_for_status:
            try:
                response.raise_for_status()
            except requests.HTTPError:
                logger.error(
                    "OpenAlex request failed: %s %s; status=%s; body=%s",
                    method,
                    url,
                    response.status_code,
                    response.text,
                )
                raise

        logger.debug("Received response %s from %s", response.status_code, url)
        return response

    def _should_retry(self, status_code: int) -> bool:
        return status_code in {429, 500, 502, 503, 504}

    def _perform_request_with_retries(self, method: str, path: str, **kwargs) -> requests.Response:
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

    def _reconstruct_abstract(self, inverted_index: Dict[str, List[int]]) -> str:
        if not isinstance(inverted_index, dict):
            return ""

        # abstract_inverted_index maps token -> [positions]
        positions = {}
        max_pos = -1
        for token, locs in inverted_index.items():
            if not isinstance(locs, list):
                continue
            for pos in locs:
                try:
                    idx = int(pos)
                except Exception:
                    continue
                positions[idx] = token
                if idx > max_pos:
                    max_pos = idx

        if max_pos < 0:
            return ""

        words = [positions.get(i, "") for i in range(max_pos + 1)]
        # join and clean up extra spaces
        return re.sub(r"\s+", " ", " ".join(words)).strip()

    def _extract_arxiv_id_from_urls(self, payload: Dict[str, Any]) -> str:
        candidates: List[str] = []
        # search common fields for arXiv URLs
        possible_fields = [payload.get("id"), payload.get("url"), payload.get("primary_location", {}).get("landing_page_url"), payload.get("primary_location", {}).get("url")]
        for field in possible_fields:
            if not field:
                continue
            if isinstance(field, str):
                m = re.search(r"arxiv\.org/(abs|pdf)/([\w\.\-v]+)", field)
                if m:
                    return m.group(2)
        return ""

    def _map_paper_to_schema(self, raw_paper: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(raw_paper, dict):
            raw_paper = {}

        # OpenAlex id is a URL like https://openalex.org/W123
        oa_id_raw = raw_paper.get("id", "") or ""
        oa_id = oa_id_raw.rstrip("/").split("/")[-1] if isinstance(oa_id_raw, str) and oa_id_raw else ""

        # authors
        authors_list = []
        for auth in raw_paper.get("authorships", []) or []:
            if not isinstance(auth, dict):
                continue
            author = auth.get("author") or {}
            name = author.get("display_name") if isinstance(author, dict) else None
            if name:
                authors_list.append(name)

        # abstract
        abstract = raw_paper.get("abstract") or ""
        if not abstract and isinstance(raw_paper.get("abstract_inverted_index"), dict):
            abstract = self._reconstruct_abstract(raw_paper.get("abstract_inverted_index", {}))

        # concepts -> keywords / fields
        keywords: List[str] = []
        fields_of_study: List[str] = []
        for concept in raw_paper.get("concepts", []) or []:
            if not isinstance(concept, dict):
                continue
            name = concept.get("display_name")
            if name:
                keywords.append(name)
                fields_of_study.append(name)

        keywords = list(dict.fromkeys(keywords))
        fields_of_study = list(dict.fromkeys(fields_of_study))

        # references
        references: List[str] = []
        refs = raw_paper.get("referenced_works") or []
        if isinstance(refs, list):
            for ref in refs:
                if isinstance(ref, str):
                    references.append(ref)

        # citations list is not part of a single work response in OpenAlex
        citations: List[str] = []

        # primary location and URLs
        primary = raw_paper.get("primary_location") or {}
        landing_url = primary.get("landing_page_url") or raw_paper.get("id") or ""
        pdf_url = primary.get("pdf_url") or ""
        if not pdf_url and isinstance(landing_url, str) and landing_url.endswith(".pdf"):
            pdf_url = landing_url

        doi = raw_paper.get("doi") or ""
        arxiv_id = self._extract_arxiv_id_from_urls(raw_paper) or ""

        source_list = ["openalex"]
        if arxiv_id:
            source_list.append("arxiv")

        publication_date = raw_paper.get("publication_date") or ""

        return {
            "paper_id": oa_id,
            "title": raw_paper.get("title", "") or "",
            "abstract": abstract,
            "authors": authors_list,
            "year": raw_paper.get("publication_year", 0) or 0,
            "venue": (
                ((raw_paper.get("primary_location") or {}).get("source") or {}).get("display_name", "")
                if isinstance((raw_paper.get("primary_location") or {}).get("source"), dict)
                else ""
            ),
            "doi": doi,
            "arxiv_id": arxiv_id,
            "openalex_id": oa_id,
            # keep compatibility key expected by downstream code
            "keywords": keywords,
            "fields_of_study": fields_of_study,
            "citations": citations,
            "references": references,
            "url": landing_url or "",
            "pdf_url": pdf_url or "",
            "source": source_list,
            "metadata": {
                "citation_count": raw_paper.get("cited_by_count", 0) or 0,
                "reference_count": len(references),
                "publication_date": publication_date,
                "updated_at": datetime.now(UTC).isoformat(),
            },
        }

    def save_papers(self, papers: List[Dict[str, Any]], query: str) -> Path:
        if not isinstance(papers, list):
            raise TypeError("papers must be a list")

        if not query.strip():
            raise ValueError("query must not be empty")

        sanitized_query = re.sub(r"\s+", "_", query.strip())
        sanitized_query = re.sub(r"[^A-Za-z0-9_\-]", "", sanitized_query)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"openalex_{sanitized_query}_{timestamp}.json"

        project_root = Path(__file__).resolve().parents[2]
        destination_dir = project_root / "data" / "raw" / "openalex"
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
        candidates: List[tuple[str, str]] = []

        if doi and doi.strip():
            candidates.append((doi.strip(), "DOI"))
        if arxiv_id and arxiv_id.strip():
            clean_arxiv = arxiv_id.strip().split("v")[0]
            candidates.append((clean_arxiv, "ArXiv"))
        if title and title.strip():
            candidates.append((title.strip(), "title"))

        if not candidates:
            logger.warning("No lookup values provided for OpenAlex paper lookup")
            return None

        for value, lookup_type in candidates:
            print(f"\nTrying {lookup_type}: {value}")
            if lookup_type == "DOI":
                path = f"{self.SEARCH_ENDPOINT}/doi:{value}"
                params = {}
            else:
                path = self.SEARCH_ENDPOINT
                if lookup_type == "ArXiv":
                    # try searching by arXiv in the landing page / urls
                    params = {
                        "search": value,
                        "per-page": 1,
                        "mailto": os.getenv("OPENALEX_EMAIL", ""),
                    }
                else:
                    params = {
                        "search": value,
                        "per-page": 1,
                        "mailto": os.getenv("OPENALEX_EMAIL", ""),
                    }

            logger.info("Looking up OpenAlex paper by %s using value=%s", lookup_type, value)
            try:
                print("Sending request to OpenAlex...")
                response = self._perform_request_with_retries(
                    "GET",
                    path,
                    params=params,
                    timeout=self.DEFAULT_TIMEOUT,
                )
                print("Received response from OpenAlex")
            except requests.HTTPError as exc:
                status_code = exc.response.status_code if exc.response else None
                if status_code == 404:
                    logger.warning("No OpenAlex paper found for %s lookup using %s", lookup_type, value)
                    continue

                logger.warning("OpenAlex paper lookup by %s failed for %s: %s", lookup_type, value, exc)
                continue
            except requests.RequestException as exc:
                logger.warning("OpenAlex paper lookup by %s failed for %s: %s", lookup_type, value, exc)
                continue

            data = response.json()
            print("JSON parsed successfully")
            if not isinstance(data, dict):
                logger.warning("Unexpected response format from OpenAlex API during %s lookup", lookup_type)
                continue

            # OpenAlex returns a single work object for direct doi: path,
            # or a paginated 'results' list for search endpoint.
            items: List[Dict[str, Any]] = []
            if isinstance(data.get("results"), list):
                items = [item for item in data.get("results", []) if isinstance(item, dict)]
            elif data.get("id"):
                items = [data]

            if not items:
                logger.warning("No OpenAlex paper found for %s lookup using %s", lookup_type, value)
                continue
            
            print("Mapping paper...")
            mapped = self._map_paper_to_schema(items[0])
            print("Paper mapped successfully")
            logger.info("Resolved OpenAlex paper for %s lookup: %s", lookup_type, mapped.get("title", ""))
            return mapped

        logger.warning("No OpenAlex paper matched the provided identifiers")
        return None

    def search_papers(self, query: str, limit: int = 100) -> Dict[str, Any]:
        if not query.strip():
            raise ValueError("Search query must not be empty")

        if limit <= 0:
            raise ValueError("limit must be positive")

        params = {"search": query, "per-page": min(limit, 100), "page": 1}
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
                raise ValueError("Unexpected response format from OpenAlex API")

            batch = data.get("results") or []
            if not batch:
                break

            mapped_batch = [self._map_paper_to_schema(p) for p in batch if isinstance(p, dict)]
            results.extend(mapped_batch)

            if total is None:
                total = data.get("meta", {}).get("count")

            if len(results) >= limit:
                break

            if len(batch) < params["per-page"]:
                break

            params["page"] = params.get("page", 1) + 1
        return {"query": query, "limit": limit, "returned": len(results), "total": total, "data": results}


def main() -> None:
    client = OpenAlexClient()

    try:
        query = input("Enter research topic: ").strip()
        if not query:
            print("Error: Research topic cannot be empty.")
            return

        limit_input = input("Maximum number of papers (default 100): ").strip()
        limit = int(limit_input) if limit_input else 100

        result = client.search_papers(query=query, limit=limit)
        output_path = client.save_papers(papers=result["data"], query=query)

        print("\n" + "=" * 50)
        print("OpenAlex Search Completed")
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
