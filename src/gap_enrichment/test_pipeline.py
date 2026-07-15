"""
Small integration test for Phase 6 pipeline.

Runs only selected candidates to avoid
calling Groq API for every candidate.
"""


from src.gap_enrichment.pipeline import GapEnrichmentPipeline
from src.gap_enrichment.candidate_loader import CandidateLoader



def main():


    # --------------------------------
    # Initialize pipeline
    # Pipeline creates its own loaders
    # --------------------------------

    pipeline = GapEnrichmentPipeline(

        candidate_file=
            "data/processed/final_gap_candidates.json",

        merged_file=
            "data/processed/merged/merged_RAG.json",

        entity_folder=
            "data/processed/entities",

        graph_file=
            "data/processed/knowledge_graph.json"
    )



    # --------------------------------
    # Load candidates
    # --------------------------------

    candidate_loader = CandidateLoader(
        "data/processed/final_gap_candidates.json"
    )


    candidates = (
        candidate_loader.load()
    )



    # --------------------------------
    # Test only first 3 candidates
    # --------------------------------

    test_candidates = candidates[:3]


    print(
        f"Testing {len(test_candidates)} candidates"
    )



    results = []



    # --------------------------------
    # Run pipeline step by step
    # --------------------------------

    for candidate in test_candidates:


        print("\n==============================")

        print(
            "Testing:",
            candidate.gap_id
        )

        print("==============================")



        try:


            result = pipeline.process_candidate(
                candidate
            )


            results.append(
                result
            )


            print(
                "SUCCESS:",
                candidate.gap_id
            )



        except Exception as e:


            print(
                "FAILED:",
                candidate.gap_id
            )


            print(
                "ERROR:",
                e
            )



    # --------------------------------
    # Display results
    # --------------------------------

    print(
        "\n\n========== TEST RESULTS =========="
    )



    for result in results:


        print(
            "\nGap ID:"
        )

        print(
            result.get("gap_id")
        )


        print(
            "\nType:"
        )

        print(
            result.get("gap_type")
        )


        print(
            "\nConfidence:"
        )

        print(
            result.get("confidence")
        )



if __name__ == "__main__":

    main()