import unittest
from unittest.mock import Mock, patch

from src.Extraction.extraction_pipeline import extract_paper_entities
from src.entity_extraction.entity_extractor import EntityExtractor


class EntityExtractorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.paper = {
            "paper_id": "paper-1",
            "title": "BERT on GLUE",
            "abstract": "We evaluate BERT on the GLUE benchmark for natural language inference.",
            "keywords": ["bert", "glue"],
        }

    @patch("src.entity_extraction.entity_extractor.GroqClient")
    def test_extract_entities_returns_validated_schema(self, mock_groq_cls: Mock) -> None:
        mock_client = Mock()
        mock_client.generate.return_value = '{"datasets": ["GLUE"], "tasks": ["natural language inference"], "metrics": [], "claims": ["BERT improves performance"]}'
        mock_groq_cls.return_value = mock_client

        extractor = EntityExtractor(groq_client=mock_client)
        result = extractor.extract_entities(self.paper)

        self.assertEqual(result["paper_id"], "paper-1")
        self.assertEqual(result["datasets"], ["GLUE"])
        self.assertEqual(result["tasks"], ["natural language inference"])
        self.assertEqual(result["metrics"], [])
        self.assertEqual(result["claims"], ["BERT improves performance"])
        self.assertEqual(result["methods"], [])
        self.assertEqual(result["models"], [])
        self.assertEqual(result["algorithms"], [])
        self.assertEqual(result["keywords"], [])
        self.assertEqual(result["summary"], "")

    @patch("src.Extraction.extraction_pipeline.EntityExtractor")
    def test_extract_paper_entities_uses_unified_extractor(self, mock_extractor_cls: Mock) -> None:
        mock_extractor = Mock()
        mock_extractor.extract_entities.return_value = {
            "paper_id": "paper-1",
            "methods": [],
            "models": [],
            "algorithms": [],
            "datasets": ["GLUE"],
            "tasks": [],
            "metrics": [],
            "claims": [],
            "keywords": [],
            "summary": "",
        }
        mock_extractor_cls.return_value = mock_extractor

        result = extract_paper_entities(self.paper, groq_client=Mock())

        self.assertEqual(result["datasets"], ["GLUE"])
        mock_extractor.extract_entities.assert_called_once_with(self.paper)


if __name__ == "__main__":
    unittest.main()
