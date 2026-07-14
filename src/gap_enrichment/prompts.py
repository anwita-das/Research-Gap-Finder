import json
from .schemas import PaperSections

def build_enrichment_prompt(
    paper: dict,
    entities: dict,
    sections: PaperSections,
) -> str:
    """
    Builds the prompt for enriching a research paper.

    The LLM should NOT repeat already extracted entities.
    Instead, it should focus on extracting higher-level
    semantic information that will be useful for gap detection.
    """

    return f"""
You are an expert AI researcher specializing in analyzing scientific papers.

You are given:

1. Paper metadata
2. Paper abstract
3. Information that has ALREADY been extracted from this paper.

Your task is NOT to repeat or rephrase the existing extracted information.

Instead, infer and extract ONLY the following:

1. Key Contributions
   - What are the paper's main contributions?
   - Return 2-5 concise bullet points.

2. Experimental Setup
   Describe ONLY what is explicitly mentioned about:

    - datasets
    - baselines
    - evaluation procedure
    - implementation
    - experimental protocol

    If insufficient information is available,
    return an empty string.

3. Experimental Results
   - Summarize the important findings.
   - Mention improvements over baselines if stated.

4. Limitations
   - Prefer limitations explicitly discussed by the authors.
   - If none are explicitly stated, you may infer at most 1-2 reasonable limitations, and clearly indicate they are inferred.
   - Keep them concise.

5. Future Work
   - Prefer future work explicitly mentioned by the authors.
   - If none is mentioned, suggest at most 1-2 plausible future directions based on the paper.

6. Novelty Points
   - What makes this paper different from previous work?
   - Mention only the genuinely novel ideas.

7. Implementation Details
   - Mention implementation details that could help reproduce the work.
   - Examples:
       - architecture
       - retrieval strategy
       - indexing
       - training procedure
       - optimization tricks

Return ONLY valid JSON.

The JSON schema MUST be:

{{
    "experimental_results": [],
    "limitations": [],
    "future_work": [],
    "key_contributions": [],
    "novelty_points": [],
    "experimental_setup": "",
    "implementation_details": []
}}

--------------------------------------------------

PAPER METADATA

Title:
{paper["title"]}

Year:
{paper["year"]}

Authors:
{", ".join(paper["authors"])}

--------------------------------------------------

ABSTRACT

{paper["abstract"]}

--------------------------------------------------

EXTRACTED PAPER SECTIONS

METHODOLOGY

{"\n".join(sections.methodology)}

--------------------------------------------------

EXPERIMENTAL SETUP

{"\n".join(sections.experimental_setup)}

--------------------------------------------------

EXPERIMENTAL RESULTS

{"\n".join(sections.experimental_results)}

--------------------------------------------------

DISCUSSION

{"\n".join(sections.discussion)}

--------------------------------------------------

LIMITATIONS

{"\n".join(sections.limitations)}

--------------------------------------------------

FUTURE WORK

{"\n".join(sections.future_work)}

--------------------------------------------------

CONCLUSION

{"\n".join(sections.conclusion)}

--------------------------------------------------

ALREADY EXTRACTED INFORMATION

{json.dumps(entities, indent=2)}

--------------------------------------------------

Remember:

- DO NOT repeat the existing entity extraction.
- Focus ONLY on the missing semantic information.
- Return ONLY valid JSON.
- Do not invent experimental results, datasets, limitations or future work that are not supported by the abstract or provided information.
- The extracted paper sections originate from the full paper.
- Prefer them over the abstract whenever they contain additional information.

"""

if __name__ == "__main__":

    from .context_retriever import ContextRetriever

    retriever = ContextRetriever()

    candidate = retriever.load_candidates()[0]

    context = retriever.build_context(candidate)

    prompt = build_enrichment_prompt(
        context.source_paper,
        context.source_entities,
    )

    print(prompt)