"""
runner.py

Runs the complete Google ADK Phase 6 workflow.

Root Agent
 ├── Section Extraction Agent
 ├── Enrichment Agent
 ├── Temporal Agent
 ├── Comparison Agent
 ├── Gap Detector Agent
 └── Result Agent
"""

from __future__ import annotations

import asyncio
import uuid

from google.genai.types import Content, Part
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from ..agents.root_agent import root_agent


APP_NAME = "research-gap-finder"


async def run_gap_pipeline(user_input: str):
    """
    Execute the complete Phase 6 workflow.
    """

    session_service = InMemorySessionService()

    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id="local-user",
        session_id=str(uuid.uuid4()),
    )

    runner = Runner(
        agent=root_agent,
        session_service=session_service,
        app_name=APP_NAME,
    )

    message = Content(
        role="user",
        parts=[
            Part(text=user_input)
        ],
    )

    async for event in runner.run_async(
        session_id=session.id,
        user_id=session.user_id,
        new_message=message,
    ):
        print(event)


if __name__ == "__main__":

    asyncio.run(
        run_gap_pipeline(
            "Run research gap detection."
        )
    )