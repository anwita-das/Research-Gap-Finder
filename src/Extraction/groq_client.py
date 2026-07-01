from __future__ import annotations

import os
import logging
from typing import Optional

from dotenv import load_dotenv
from groq import Groq


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class GroqClient:
    """Client wrapper for interacting with the Groq chat completion API."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        """Initialize the Groq client using the configured API key.

        Args:
            api_key: Optional Groq API key. If not provided, the key is loaded
                from the environment or a .env file.
            model: The Groq chat model to use for generation.

        Raises:
            ValueError: If no API key is available.
        """
        self._load_environment()
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv(       
            "GROQ_MODEL",
            "llama-3.1-8b-instant"
        )

        if not self.api_key:
            raise ValueError("GROQ_API_KEY is required")

        self.client = Groq(
            api_key=self.api_key,
            timeout=60
        )
        logger.debug("Initialized GroqClient with model=%s", self.model)

    @staticmethod
    def _load_environment() -> None:
        """Load environment variables from a .env file if present."""
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        dotenv_path = os.path.join(project_root, ".env")

        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path)
            logger.debug("Loaded environment variables from %s", dotenv_path)
        else:
            load_dotenv()
            logger.debug("Loaded environment variables from default locations")

    def generate(self, prompt: str) -> str:
        """Generate a completion for the provided prompt.

        Args:
            prompt: The prompt text to send to the Groq model.

        Returns:
            The generated text response.

        Raises:
            ValueError: If the prompt is empty.
            RuntimeError: If the API response is empty.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        try:
           response = self.client.chat.completions.create(
               
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0,
                response_format={
                    "type": "json_object"
                }
            )
        except Exception as exc:  # pragma: no cover - runtime integration
            logger.exception("Groq API request failed: %s", exc)
            raise RuntimeError(f"Groq API request failed: {exc}") from exc

        if not response or not getattr(response, "choices", None):
            raise RuntimeError("Groq API returned an empty response")

        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Groq API returned an empty message")

        return content.strip()
