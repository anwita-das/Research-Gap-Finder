"""Reusable Groq client for generating text responses from Llama 3."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Final, Optional

from dotenv import load_dotenv
from groq import Groq

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class GroqClient:
    """Thin wrapper around the Groq SDK for prompt generation.

    The client loads its API key from a local ``.env`` file when available,
    reuses a single SDK client instance, retries transient failures, and
    returns only the generated text payload from the model.
    """

    DEFAULT_MODEL: Final = "llama-3.3-70b-versatile"
    DEFAULT_TIMEOUT: Final = 30
    MAX_RETRIES: Final = 3
    BACKOFF_BASE: Final = 1.0

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None) -> None:
        """Initialize the Groq client.

        Args:
            api_key: Optional API key. If omitted, the client loads it from the
                project's ``.env`` file or the environment.
            model_name: Optional model identifier. If omitted, the client uses
                a configured value if present, otherwise a sensible default.
        """
        self._load_environment()
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model_name = model_name or self._read_model_name_from_config() or self.DEFAULT_MODEL
        self.client = self._build_client()
        logger.debug("Initialized GroqClient with model=%s", self.model_name)

    def _project_root(self) -> Path:
        """Return the repository root based on the location of this module."""
        return Path(__file__).resolve().parents[2]

    @staticmethod
    def _load_environment() -> None:
        """Load environment variables from a project-level .env file if present."""
        project_root = Path(__file__).resolve().parents[2]
        dotenv_path = project_root / ".env"

        if dotenv_path.exists():
            load_dotenv(dotenv_path)
            logger.debug("Loaded environment variables from %s", dotenv_path)
        else:
            load_dotenv()
            logger.debug("Loaded environment variables from default locations")

    def _read_model_name_from_config(self) -> Optional[str]:
        """Read a configured model name from common config files if available."""
        project_root = self._project_root()
        candidates = [
            project_root / "configs" / "config.json",
        ]

        for path in candidates:
            if not path.exists():
                continue
            try:
                import json

                with path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                if isinstance(data, dict):
                    value = data.get("llm")
                    if isinstance(value, str) and value.strip():
                        return value.strip()
            except (OSError, json.JSONDecodeError, TypeError) as exc:
                logger.warning("Could not read model config from %s: %s", path, exc)

        return None

    def _build_client(self) -> Groq:
        """Create and return a single Groq SDK client instance."""
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Please add it to your .env file."
            )

        return Groq(
            api_key=self.api_key,
            timeout=self.DEFAULT_TIMEOUT,
        )

    def generate(self, prompt: str) -> str:
        """Send a prompt to Groq and return only the generated text.

        Args:
            prompt: The prompt text to send to the model.

        Returns:
            The model's text response as a string.

        Raises:
            RuntimeError: If the request fails after retries or the response is
                malformed.
        """
        if not prompt or not prompt.strip():
            raise ValueError("prompt must not be empty")

        logger.info("Generating response for prompt length=%s", len(prompt))

        last_error: Optional[Exception] = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                content = self._extract_text(response)
                logger.info("Groq generation succeeded on attempt %s", attempt)
                return content
            except Exception as exc:  # pragma: no cover - defensive branch for SDK failures
                last_error = exc
                logger.warning("Groq generation attempt %s failed: %s", attempt, exc)
                if attempt == self.MAX_RETRIES:
                    break
                time.sleep(self.BACKOFF_BASE * attempt)

        logger.error("Groq generation failed after %s attempts", self.MAX_RETRIES)
        raise RuntimeError("Groq generation failed") from last_error

    def _extract_text(self, response: object) -> str:
        """Extract the text content from a Groq SDK response object."""
        try:
            choices = getattr(response, "choices", None)
            if not choices:
                raise ValueError("Response contains no choices")

            message = choices[0].message
            content = getattr(message, "content", None)
            if isinstance(content, str):
                return content

            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        parts.append(item["text"])
                if parts:
                    return "".join(parts)

            raise ValueError("Response content was not a string")
        except Exception as exc:  # pragma: no cover - defensive branch
            logger.error("Failed to parse Groq response: %s", exc)
            raise RuntimeError("Unable to parse Groq response") from exc
