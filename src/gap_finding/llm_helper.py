from __future__ import annotations

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from groq import Groq

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class LLMHelper:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:

        load_dotenv()

        self.api_key = api_key or os.getenv("GROQ_API_KEY")

        # Default Groq model
        self.model = model or os.getenv(
            "GROQ_MODEL",
            "llama-3.3-70b-versatile",
        )

        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found.")

        print("Model:", self.model)

        self.client = Groq(api_key=self.api_key)

    def generate(self, prompt: str) -> str:

        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        try:
            print("Sending prompt...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0,
            )

        except Exception as exc:
            logger.exception("Groq request failed")
            raise RuntimeError(
                f"Groq request failed: {exc}"
            ) from exc

        if not response.choices:
            raise RuntimeError(
                "Groq returned an empty response."
            )

        print("Received response")
        print(repr(response.choices[0].message.content))
        return response.choices[0].message.content.strip()