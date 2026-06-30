import unittest
from unittest.mock import Mock, patch

import requests

from src.ingestion.semantic_scholar_client import SemanticScholarClient


class SemanticScholarClientTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = SemanticScholarClient(api_key="test-key")

    @patch("src.ingestion.semantic_scholar_client.requests.Session.request")
    def test_get_paper_by_identifier_returns_mapped_paper_for_doi(self, mock_request: Mock) -> None:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.text = ""
        mock_response.json.return_value = {
            "paperId": "paper-123",
            "title": "A Test Paper",
            "abstract": "An abstract",
            "authors": [{"name": "Ada Lovelace"}],
            "year": 2024,
            "venue": "NeurIPS",
            "externalIds": {"DOI": "10.1000/test", "ArXiv": "2401.00001"},
            "citationCount": 7,
            "referenceCount": 2,
            "references": [],
            "citations": [],
            "fieldsOfStudy": ["Computer Science"],
            "publicationDate": "2024-01-01",
            "url": "https://example.com/paper",
        }
        mock_request.return_value = mock_response

        result = self.client.get_paper_by_identifier(doi="10.1000/test")

        self.assertIsNotNone(result)
        self.assertEqual(result["title"], "A Test Paper")
        self.assertEqual(result["doi"], "10.1000/test")
        self.assertEqual(result["arxiv_id"], "2401.00001")
        self.assertEqual(result["metadata"]["citation_count"], 7)

    @patch("src.ingestion.semantic_scholar_client.requests.Session.request")
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
