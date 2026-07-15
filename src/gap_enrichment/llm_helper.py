"""
llm_helper.py

Phase 6 LLM wrapper.

Used for:
- Temporal Analysis Agent
- Comparison Agent
- Gap Detector

Unlike Phase 2's Groq client, this allows
normal text responses.
"""

from __future__ import annotations

import os
import logging
from typing import Optional

from dotenv import load_dotenv
from groq import Groq


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class LLMHelper:
    """
    Groq wrapper for Phase 6 reasoning agents.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):

        load_dotenv()

        self.api_key = (
            api_key
            or os.getenv("GROQ_API_KEY")
        )

        self.model = (
            model
            or os.getenv(
                "GROQ_MODEL",
                "llama-3.1-8b-instant"
            )
        )

        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY is required"
            )

        self.client = Groq(
            api_key=self.api_key,
            timeout=60
        )


    def generate(
        self,
        prompt: str
    ) -> str:
        """
        Generate a normal text response.
        """

        if not prompt.strip():
            raise ValueError(
                "Prompt cannot be empty"
            )

        try:

            response = self.client.chat.completions.create(

                model=self.model,

                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                temperature=0
            )


        except Exception as exc:

            logger.exception(
                "Groq request failed"
            )

            raise RuntimeError(
                f"Groq request failed: {exc}"
            )


        if not response.choices:
            raise RuntimeError(
                "Empty Groq response"
            )


        content = (
            response
            .choices[0]
            .message
            .content
        )


        if not content:
            raise RuntimeError(
                "Empty model output"
            )


        return content.strip()