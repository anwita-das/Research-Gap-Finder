import unittest
from unittest.mock import Mock, patch

from src.ingestion.ingestion_pipeline import enrich_paper, merge_paper_metadata, process_topic


class IngestionPipelineTest(unittest.TestCase):
    def test_merge_paper_metadata_prefers_semantic_scholar_for_citations_and_arxiv_for_title(self) -> None:
        arxiv_paper = {
            "paper_id": "arxiv_123",
            "title": "ArXiv title",
            "abstract": "ArXiv abstract",
            "authors": ["A"],
            "year": 2024,
            "venue": "arXiv",
            "doi": "10.1000/test",
            "arxiv_id": "1234.56789",
            "semantic_scholar_id": "",
            "keywords": [],
            "fields_of_study": ["cs"],
            "citations": [],
            "references": [],
            "url": "https://arxiv.org/abs/1234.56789",
            "pdf_url": "https://arxiv.org/pdf/1234.56789.pdf",
            "source": ["arxiv"],
            "metadata": {
                "citation_count": 0,
                "reference_count": 0,
                "publication_date": "2024-01-01",
                "updated_at": "",
            },
        }
        semantic_paper = {
            "paper_id": "semantic_456",
            "title": "Semantic title",
            "abstract": "Semantic abstract",
            "authors": ["B"],
            "year": 2024,
            "venue": "ACL",
            "doi": "10.1000/test",
            "arxiv_id": "1234.56789",
            "semantic_scholar_id": "ss-456",
            "keywords": [],
            "fields_of_study": ["Machine Learning"],
            "citations": ["c1"],
            "references": ["r1"],
            "url": "https://semanticscholar.org/paper",
            "pdf_url": "https://example.com/paper.pdf",
            "source": ["semantic_scholar"],
            "metadata": {
                "citation_count": 10,
                "reference_count": 4,
                "publication_date": "2024-02-01",
                "updated_at": "",
            },
        }

        merged = merge_paper_metadata(arxiv_paper, semantic_paper)

        self.assertEqual(merged["title"], "ArXiv title")
        self.assertEqual(merged["abstract"], "ArXiv abstract")
        self.assertEqual(merged["pdf_url"], "https://arxiv.org/pdf/1234.56789.pdf")
        self.assertEqual(merged["semantic_scholar_id"], "ss-456")
        self.assertEqual(merged["fields_of_study"], ["Machine Learning"])
        self.assertEqual(merged["citations"], ["c1"])
        self.assertEqual(merged["references"], ["r1"])
        self.assertEqual(merged["metadata"]["citation_count"], 10)

    @patch("src.ingestion.ingestion_pipeline.fetch_papers")
    @patch("src.ingestion.ingestion_pipeline.OpenAlexClient")
    def test_process_topic_returns_enriched_papers(self, mock_client_cls: Mock, mock_fetch_papers: Mock) -> None:
        mock_fetch_papers.return_value = [
            {
                "paper_id": "arxiv_123",
                "title": "ArXiv title",
                "abstract": "ArXiv abstract",
                "authors": ["A"],
                "year": 2024,
                "venue": "arXiv",
                "doi": "10.1000/test",
                "arxiv_id": "1234.56789",
                "semantic_scholar_id": "",
                "keywords": [],
                "fields_of_study": ["cs"],
                "citations": [],
                "references": [],
                "url": "https://arxiv.org/abs/1234.56789",
                "pdf_url": "https://arxiv.org/pdf/1234.56789.pdf",
                "source": ["arxiv"],
                "metadata": {
                    "citation_count": 0,
                    "reference_count": 0,
                    "publication_date": "2024-01-01",
                    "updated_at": "",
                },
            }
        ]

        mock_client = mock_client_cls.return_value
        mock_client.get_paper_by_identifier.return_value = {
            "paper_id": "semantic_456",
            "title": "Semantic title",
            "abstract": "Semantic abstract",
            "authors": ["B"],
            "year": 2024,
            "venue": "ACL",
            "doi": "10.1000/test",
            "arxiv_id": "1234.56789",
            "semantic_scholar_id": "ss-456",
            "keywords": [],
            "fields_of_study": ["Machine Learning"],
            "citations": ["c1"],
            "references": ["r1"],
            "url": "https://semanticscholar.org/paper",
            "pdf_url": "https://example.com/paper.pdf",
            "source": ["semantic_scholar"],
            "metadata": {
                "citation_count": 10,
                "reference_count": 4,
                "publication_date": "2024-02-01",
                "updated_at": "",
            },
        }

        papers = process_topic("machine learning", 1)

        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0]["title"], "ArXiv title")
        self.assertEqual(papers[0]["semantic_scholar_id"], "ss-456")


if __name__ == "__main__":
    unittest.main()
