"""
pipeline.py

Phase 6 Gap Enrichment Pipeline

Flow:

Candidate Loader
        |
        v
Paper Loader
        |
        +----------------+
        |                |
        v                v
    Paper 1          Paper 2
        |
        v
Temporal Analysis Agent
        |
        v
Comparison Agent
        |
        v
Gap Detector
        |
        v
Research Gap JSON
"""


from __future__ import annotations

import json
from pathlib import Path


from src.gap_finding.candidate_loader import CandidateLoader
from src.gap_finding.paper_loader import PaperLoader
from src.gap_finding.graph_loader import GraphLoader

from src.gap_finding.temporal_agent import TemporalAnalysisAgent
from src.gap_finding.comparison_agent import ComparisonAgent
from src.gap_finding.gap_detector_agent import GapDetector
from src.gap_finding.enrichment_pipeline import GapEnrichmentPipeline as PaperEnrichmentPipeline


class GapFindingPipeline:


    def __init__(
        self,
        candidate_file: str,
        graph_file: str
    ):


        # -----------------------------
        # Loaders
        # -----------------------------

        self.candidate_loader = CandidateLoader(
            candidate_file
        )


        self.paper_loader = PaperLoader(
            "data/processed/enriched_papers"
        )


        self.graph_loader = GraphLoader(
            graph_file
        )


        # -----------------------------
        # Agents
        # -----------------------------

        self.temporal_agent = TemporalAnalysisAgent(
            self.graph_loader,
            self.paper_loader
        )


        self.comparison_agent = ComparisonAgent()


        self.gap_detector = GapDetector()



    # ==================================================
    # Candidate conversion
    # ==================================================

    def _candidate_to_dict(
        self,
        candidate
    ):
        """
        Convert CandidateEdge object from
        Phase 5 into dictionary format.
        """

        if isinstance(candidate, dict):
            return candidate


        return {

            "gap_id":
                candidate.gap_id,

            "source_node":
                candidate.source_node,

            "target_node":
                candidate.target_node,

            "relation":
                candidate.relation,

            "shared_entity":
                candidate.shared_entity,

            "motif_score":
                candidate.motif_score,

            "graphsage_score":
                candidate.graphsage_score,

            "confidence":
                candidate.confidence,

            "status":
                candidate.status
        }



    # ==================================================
    # PROCESS ONE CANDIDATE
    # ==================================================

    def process_candidate(
        self,
        candidate
    ):


        # Convert CandidateEdge -> dict

        candidate = self._candidate_to_dict(
            candidate
        )


        print(
            "\nProcessing:",
            candidate["gap_id"]
        )



        # ----------------------------------
        # Load candidate papers
        # ----------------------------------

        source = candidate["source_node"]

        target = candidate["target_node"]



        paper1 = self.paper_loader.load(
            source
        )


        paper2 = self.paper_loader.load(
            target
        )



        print(
            "Loaded papers:",
            paper1["paper_id"],
            paper2["paper_id"]
        )



        # ----------------------------------
        # Temporal Analysis
        # ----------------------------------

        temporal_summary = (
            self.temporal_agent.analyze(
                candidate
            )
        )


        print(
            "Temporal analysis complete"
        )



        # ----------------------------------
        # Comparison
        # ----------------------------------

        comparison_summary = (
            self.comparison_agent.compare(
                candidate,
                paper1,
                paper2,
                temporal_summary
            )
        )


        print(
            "Comparison complete"
        )



        # ----------------------------------
        # Gap Detection
        # ----------------------------------

        gap = (
            self.gap_detector.detect_gap(
                candidate,
                paper1,
                paper2,
                temporal_summary,
                comparison_summary
            )
        )


        print(
            "Gap detection complete"
        )


        return gap



    # ==================================================
    # PROCESS ALL CANDIDATES
    # ==================================================

    def run(self, limit=None):

        candidates = (
            self.candidate_loader.load()
        )

        if limit:
            candidates = candidates[:limit]


        results = []


        for candidate in candidates:

            try:

                gap = self.process_candidate(
                    candidate
                )

                results.append(
                    gap
                )

            except Exception as e:

                candidate_dict = (
                    self._candidate_to_dict(candidate)
                )

                print(
                    "Failed candidate:",
                    candidate_dict.get(
                        "gap_id",
                        "unknown"
                    ),
                    e
                )


        output_file = (
            "data/processed/research_gaps.json"
        )


        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                results,
                f,
                indent=4,
                ensure_ascii=False
            )


        print(
            f"Saved {len(results)} gaps"
        )


        return results



# ======================================================
# MAIN EXECUTION
# ======================================================


if __name__ == "__main__":


    # -------------------------------------
    # Step 1: Build enriched papers
    # -------------------------------------

    # print("\nRunning Paper Enrichment Pipeline...\n")

    # enrichment = PaperEnrichmentPipeline()

    # enrichment.run()

    print("\nPaper enrichment completed.\n")

    pipeline = GapFindingPipeline(

        candidate_file=
        "data/processed/final_gap_candidates.json",

        graph_file=
        "data/processed/knowledge_graph.json"

    )



    gaps = pipeline.run()



    output = Path(
        "data/processed/research_gaps.json"
    )



    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )



    with open(
        output,
        "w",
        encoding="utf-8"
    ) as f:


        json.dump(
            gaps,
            f,
            indent=4
        )



    print(
        "\nSaved research gaps to:",
        output
    )