"""
Test Temporal Analysis Agent
"""

from src.gap_enrichment.graph_loader import GraphLoader
from src.gap_enrichment.paper_loader import PaperLoader
from src.gap_enrichment.temporal_agent import TemporalAnalysisAgent


def main():

    # -------------------------------
    # Load graph
    # -------------------------------

    graph_loader = GraphLoader(
        "data/processed/knowledge_graph.json"
    )


    # -------------------------------
    # Load papers + entities
    # -------------------------------

    paper_loader = PaperLoader(

        merged_file=
        "data/processed/merged/merged_RAG.json",

        entity_folder=
        "data/processed/entities"

    )


    # -------------------------------
    # Create agent
    # -------------------------------

    agent = TemporalAnalysisAgent(
        graph_loader,
        paper_loader
    )


    # -------------------------------
    # Dummy Phase 5 candidate
    # -------------------------------

    candidate = {

        "gap_id": "gap_0059",

        "source_node":
        "paper:arxiv_2409.03708v2",

        "target_node":
        "paper:arxiv_2607.10151v1",

        "relation":
        "SHARES_TASK",

        "shared_entity":
        "task:question_answering",

        "motif_score": 1.0,

        "graphsage_score": 0.7003,

        "confidence": 0.8501,

        "status":
        "candidate"
    }


    # -------------------------------
    # Run temporal analysis
    # -------------------------------

    result = agent.analyze(candidate)


    print("\n========== TEMPORAL SUMMARY ==========")

    for key, value in result.items():

        print(f"\n{key}:")
        print(value)



if __name__ == "__main__":
    main()