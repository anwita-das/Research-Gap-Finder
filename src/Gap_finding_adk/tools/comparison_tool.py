"""
comparison_tool.py

Google ADK Comparison Tool.

Responsibilities
----------------
Perform deterministic comparison between two
enriched paper summaries.

This tool performs NO LLM reasoning.

It computes:

- shared methods
- shared datasets
- shared metrics
- different methods
- different datasets
- different metrics

The Comparison Agent uses these results for
higher-level reasoning.
"""

from __future__ import annotations

from typing import Any


class ComparisonTool:
    """
    Tool used by the Comparison Agent.

    Performs deterministic comparison between
    two enriched papers.
    """

    # ---------------------------------------------------------

    def prepare_comparison(
        self,
        paper1: dict[str, Any],
        paper2: dict[str, Any],
    ) -> dict[str, Any]:

        shared_methods = self._intersection(
            paper1.get("methods", []),
            paper2.get("methods", []),
        )

        shared_datasets = self._intersection(
            paper1.get("datasets", []),
            paper2.get("datasets", []),
        )

        shared_metrics = self._intersection(
            paper1.get("metrics", []),
            paper2.get("metrics", []),
        )

        different_methods = (

            self._difference(
                paper1.get("methods", []),
                paper2.get("methods", []),
            )

            +

            self._difference(
                paper2.get("methods", []),
                paper1.get("methods", []),
            )

        )

        different_datasets = (

            self._difference(
                paper1.get("datasets", []),
                paper2.get("datasets", []),
            )

            +

            self._difference(
                paper2.get("datasets", []),
                paper1.get("datasets", []),
            )

        )

        different_metrics = (

            self._difference(
                paper1.get("metrics", []),
                paper2.get("metrics", []),
            )

            +

            self._difference(
                paper2.get("metrics", []),
                paper1.get("metrics", []),
            )

        )

        return {

            "shared_methods": shared_methods,

            "shared_datasets": shared_datasets,

            "shared_metrics": shared_metrics,

            "different_methods": different_methods,

            "different_datasets": different_datasets,

            "different_metrics": different_metrics,

        }

    # ---------------------------------------------------------

    def _normalize_entity(
        self,
        text: str,
    ) -> str:

        return (

            text.lower()

            .replace("-", " ")

            .replace("_", " ")

            .strip()

        )

    # ---------------------------------------------------------

    def _intersection(
        self,
        list1: list[str],
        list2: list[str],
    ) -> list[str]:

        normalized1 = {

            self._normalize_entity(x): x

            for x in list1

        }

        normalized2 = {

            self._normalize_entity(x): x

            for x in list2

        }

        return [

            normalized1[key]

            for key in normalized1

            if key in normalized2

        ]

    # ---------------------------------------------------------

    def _difference(
        self,
        list1: list[str],
        list2: list[str],
    ) -> list[str]:

        normalized2 = {

            self._normalize_entity(x)

            for x in list2

        }

        return [

            value

            for value in list1

            if self._normalize_entity(value)
            not in normalized2

        ]


# ---------------------------------------------------------
# ADK Tool Instance
# ---------------------------------------------------------

comparison_tool = ComparisonTool()


# ---------------------------------------------------------
# ADK Callable Function
# ---------------------------------------------------------

def prepare_comparison(
    paper1: dict[str, Any],
    paper2: dict[str, Any],
) -> dict[str, Any]:
    """
    ADK callable tool.

    Called by comparison_agent.
    """

    return comparison_tool.prepare_comparison(
        paper1,
        paper2,
    )