from src.Extraction.extraction_pipeline import extract_paper_entities
from src.Extraction.groq_client import GroqClient

paper = {
     "paper_id": "test456",
    "title": "Optimization Algorithms for Neural Networks",
    "abstract": """
    We propose a neural network training method using the Adam optimization
    algorithm. The model is trained using stochastic gradient descent and
    backpropagation algorithms.
    """
}


def main():

    groq_client = GroqClient()

    result = extract_paper_entities(
        paper,
        groq_client
    )

    print(result)


if __name__ == "__main__":
    main()