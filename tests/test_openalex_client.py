import unittest
from unittest.mock import Mock, patch

import requests

from src.ingestion.openalex_client import OpenAlexClient


class OpenAlexClientTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = OpenAlexClient(api_key="test-key")

    @patch("src.ingestion.openalex_client.requests.Session.request")
    def test_get_paper_by_identifier_returns_mapped_paper_for_doi(self, mock_request: Mock) -> None:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.text = ""
        mock_response.json.return_value = {
            "id": "https://openalex.org/W123",
            "title": "A Test Paper",
            "abstract": "An abstract",
            "authorships": [{"author": {"display_name": "Ada Lovelace"}}],
            "publication_year": 2024,
            "host_venue": {"display_name": "NeurIPS"},
            "doi": "10.1000/test",
            "cited_by_count": 7,
            "referenced_works": [],
            "concepts": [{"display_name": "Computer Science"}],
            "publication_date": "2024-01-01",
            "primary_location": {"landing_page_url": "https://example.com/paper"},
        }
        mock_request.return_value = mock_response

        result = self.client.get_paper_by_identifier(doi="10.1000/test")

        self.assertIsNotNone(result)
        self.assertEqual(result["title"], "A Test Paper")
        self.assertEqual(result["doi"], "10.1000/test")
        self.assertEqual(result["arxiv_id"], "")
        self.assertEqual(result["metadata"]["citation_count"], 7)

    @patch("src.ingestion.openalex_client.requests.Session.request")
    def test_get_paper_by_identifier_returns_none_when_paper_not_found(self, mock_request: Mock) -> None:
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.headers = {}
        mock_response.text = "Not Found"
        mock_response.raise_for_status.side_effect = requests.HTTPError(response=mock_response)
        mock_request.return_value = mock_response

        result = self.client.get_paper_by_identifier(arxiv_id="9999.99999")

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
