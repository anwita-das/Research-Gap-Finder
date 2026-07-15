"""
Test Gap Detector Agent
"""


from src.gap_enrichment.paper_loader import PaperLoader
from src.gap_enrichment.gap_detector_agent import GapDetector



def main():


    # ------------------------------------------
    # Load papers
    # ------------------------------------------

    loader = PaperLoader(

        merged_file=
        "data/processed/merged/merged_RAG.json",

        entity_folder=
        "data/processed/entities"

    )



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
        0.8501

    }



    paper1 = loader.load(
        candidate["source_node"]
    )


    paper2 = loader.load(
        candidate["target_node"]
    )



    # ------------------------------------------
    # Dummy temporal output
    # (replace with temporal agent output)
    # ------------------------------------------

    temporal_summary = {


        "entity":
        "task:question_answering",


        "trend":
        "Stable",


        "publication_counts":
        {
            2021:1,
            2024:4,
            2025:3,
            2026:2
        },


        "trend_summary":
        "Question answering research shows stable growth."
    }



    # ------------------------------------------
    # Dummy comparison output
    # (replace with comparison agent output)
    # ------------------------------------------

    comparison_summary = {


        "shared_methods":
        [
            "Retrieval Augmented Generation"
        ],


        "different_methods":
        [
            "end-to-end framework",
            "Structure-Driven RAG System"
        ],


        "different_metrics":
        [
            "accuracy",
            "relevance"
        ],


        "comparison_summary":
        "Both papers use RAG but differ in architecture and evaluation."

    }



    # ------------------------------------------
    # Run detector
    # ------------------------------------------

    detector = GapDetector()


    result = detector.detect_gap(

        candidate,

        paper1,

        paper2,

        temporal_summary,

        comparison_summary

    )



    print("\n========== RESEARCH GAP ==========")


    for key,value in result.items():

        print(f"\n{key}:")
        print(value)




if __name__ == "__main__":

    main()