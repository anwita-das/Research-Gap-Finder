import unittest
from unittest.mock import Mock, patch

from src.llm.groq_client import GroqClient


class GroqClientTest(unittest.TestCase):
    @patch("src.llm.groq_client.Groq")
    def test_generate_returns_text_response(self, mock_groq_cls: Mock) -> None:
        mock_client = Mock()
        mock_completion = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Hello from Groq"))]
        mock_completion.create.return_value = mock_response
        mock_client.chat.completions = mock_completion
        mock_groq_cls.return_value = mock_client

        client = GroqClient(api_key="test-key", model_name="llama-3.1-8b-instant")
        result = client.generate("Say hi")

        self.assertEqual(result, "Hello from Groq")
        mock_groq_cls.assert_called_once()


if __name__ == "__main__":
    unittest.main()
