"""
test_comparison_agent.py

Test Phase 6 Comparison Agent
"""

from src.gap_enrichment.paper_loader import PaperLoader
from src.gap_enrichment.comparison_agent import ComparisonAgent


def main():

    # -------------------------------------------------
    # Load papers + entities
    # -------------------------------------------------

    paper_loader = PaperLoader(

        merged_file=
        "data/processed/merged/merged_RAG.json",

        entity_folder=
        "data/processed/entities"

    )


    # -------------------------------------------------
    # Create comparison agent
    # -------------------------------------------------

    agent = ComparisonAgent()



    # -------------------------------------------------
    # Dummy Phase 5 candidate
    # -------------------------------------------------

    candidate = {

        "gap_id":
        "gap_0059",

        "source_node":
        "paper:arxiv_2409.03708v2",

        "target_node":
        "paper:arxiv_2607.10151v1",

        "relation":
        "SHARES_TASK",

        "shared_entity":
        "task:question_answering",

        "motif_score":
        1.0,

        "graphsage_score":
        0.7003,

        "confidence":
        0.8501,

        "status":
        "candidate"
    }



    # -------------------------------------------------
    # Load the two connected papers
    # -------------------------------------------------

    paper1 = paper_loader.load(
        candidate["source_node"]
    )


    paper2 = paper_loader.load(
        candidate["target_node"]
    )



    # -------------------------------------------------
    # Run comparison
    # -------------------------------------------------

    result = agent.compare(

        candidate,

        paper1,

        paper2

    )



    # -------------------------------------------------
    # Display result
    # -------------------------------------------------

    print("\n========== COMPARISON SUMMARY ==========")


    for key, value in result.items():

        print(f"\n{key}:")
        print(value)



if __name__ == "__main__":

    main()