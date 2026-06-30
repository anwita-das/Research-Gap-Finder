import arxiv
import json
import os
import time
from datetime import datetime
from tqdm import tqdm


# SETTINGS

OUTPUT_FILE = "data/raw/arxiv"

PAGE_SIZE = 100

RATE_LIMIT = 3


# CREATE PAPER FORMAT

def create_paper_schema():

    return {
        "paper_id": "",
        "title": "",
        "abstract": "",
        "authors": [],
        "year": 0,
        "venue": "arXiv",
        "doi": "",
        "arxiv_id": "",
        "semantic_scholar_id": "",
        "keywords": [],
        "fields_of_study": [],
        "citations": [],
        "references": [],
        "url": "",
        "pdf_url": "",
        "source": [
            "arxiv"
        ],
        "metadata": {
            "citation_count": 0,
            "reference_count": 0,
            "publication_date": "",
            "updated_at": ""
        }
    }


# CONVERT ARXIV PAPER


def convert_arxiv_result(result):

    paper = create_paper_schema()


    arxiv_id = (
        result.entry_id
        .split("/")[-1]
    )


    paper["paper_id"] =  "arxiv_" + arxiv_id

    paper["title"] = (
        result.title
        .replace("\n", " ")
    )

    paper["abstract"] = (
        result.summary
        .replace("\n", " ")
    )


    paper["authors"] = [
        author.name
        for author in result.authors
    ]


    paper["year"] = (
        result.published.year
    )


    paper["arxiv_id"] = arxiv_id


    paper["url"] = result.entry_id


    paper["pdf_url"] = result.pdf_url


    paper["fields_of_study"] = (
        result.categories
    )


    paper["metadata"]["publication_date"] = (
        result.published.isoformat()
    )


    paper["metadata"]["updated_at"] = (
        datetime.now().isoformat()
    )


    return paper

# FETCH FROM ARXIV

def fetch_papers(query, total):


    client = arxiv.Client(

        page_size=PAGE_SIZE,

        delay_seconds=RATE_LIMIT,

        num_retries=5
    )


    search = arxiv.Search(

        query=query,

        max_results=total,

        sort_by=arxiv.SortCriterion.Relevance

    )


    papers = []


    print("Fetching papers...")


    for result in tqdm(
        client.results(search),
        total=total
    ):


        paper = convert_arxiv_result(
            result
        )


        papers.append(
            paper
        )


        time.sleep(
            RATE_LIMIT
        )


    return papers

# SAVE JSON
def save_results(papers):


    os.makedirs(
        OUTPUT_FILE,
        exist_ok=True
    )


    filename = (
        f"arxiv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )


    output_file = os.path.join(
        OUTPUT_FILE,
        filename
    )


    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:


        json.dump(

            papers,

            file,

            indent=4,

            ensure_ascii=False

        )


    print(
        "Saved",
        len(papers),
        "papers to",
        output_file
    )



# -----------------------------
# RUN PROGRAM
# -----------------------------

if __name__ == "__main__":


    query = (
        "machine learning"
    )


    papers = fetch_papers(

        query,

        total=50

    )


    save_results(
        papers
    )