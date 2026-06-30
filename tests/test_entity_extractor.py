import unittest
from unittest.mock import Mock, patch

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
        mock_client.generate.return_value = '{"datasets": [{"name": "GLUE", "confidence": 0.99}], "tasks": [{"name": "natural language inference", "confidence": 0.95}], "metrics": [], "claims": [{"name": "BERT improves performance", "confidence": 0.90}]}'
        mock_groq_cls.return_value = mock_client

        extractor = EntityExtractor(groq_client=mock_client)
        result = extractor.extract_entities(self.paper)

        self.assertEqual(result["paper_id"], "paper-1")
        self.assertEqual(result["datasets"], [{"name": "GLUE", "confidence": 0.99}])
        self.assertEqual(result["tasks"], [{"name": "natural language inference", "confidence": 0.95}])
        self.assertEqual(result["metrics"], [])
        self.assertEqual(result["claims"], [{"name": "BERT improves performance", "confidence": 0.9}])
        self.assertEqual(result["methods"], [])
        self.assertEqual(result["models"], [])
        self.assertEqual(result["algorithms"], [])
        self.assertEqual(result["keywords"], ["bert", "glue"])
        self.assertEqual(result["summary"], "")


if __name__ == "__main__":
    unittest.main()
