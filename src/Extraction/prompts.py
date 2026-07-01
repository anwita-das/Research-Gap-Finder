from __future__ import annotations

from typing import Any, Dict


def create_extraction_prompt(paper: Dict[str, Any]) -> str:
    """Create a prompt that instructs Groq to extract methods, models, and algorithms.

    Args:
        paper: A research paper dictionary containing at least paper_id, title,
            and abstract fields.

    Returns:
        A formatted prompt string for the Groq chat completion API.
    """
    paper_id = paper.get("paper_id", "")
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")

    return f"""You are an information extraction assistant for research papers.

Extract only the information explicitly mentioned in the paper.
Do not infer missing methods, models, or algorithms.
Do not assume information that is not present.
Do not hallucinate.
Return valid JSON only.
Do not include markdown.
Do not include explanations outside JSON.

The response MUST follow exactly this JSON schema:
{{
  "paper_id": "",
  "methods": [],
  "models": [],
  "algorithms": [],
  "datasets": [],
  "tasks": [],
  "metrics": [],
  "claims": [],
  "keywords": [],
  "summary": ""
}}

Extraction rules:

Methods:
- Extract high-level approaches, techniques, frameworks, or proposed approaches.
- Examples:
  "residual learning"
  "transfer learning"
  "data augmentation"

Models:
- Extract machine learning models, architectures, and model families used in the paper.
- Include both specific models and general model types if they represent the learning system.
- Examples:
  "ResNet"
  "CNN"
  "Convolutional Neural Network"
  "Transformer"
  "neural network"
  "Random Forest"
  "Support Vector Machine"
- Do not place optimization algorithms in models.

Algorithms:
- Extract named algorithms, optimization algorithms, mathematical procedures, or training algorithms.
- Examples:
  "Adam"
  "Adam optimizer"
  "SGD"
  "stochastic gradient descent"
  "backpropagation"
  "beam search"

Classification rules:
- If something contains the word "algorithm", "optimizer", "descent", or is a named computational procedure, put it in algorithms.
- Do not put algorithms inside methods.
- Do not duplicate the same item across categories.

Datasets:
- Always return [].

Tasks:
- Always return [].

Metrics:
- Always return [].

Claims:
- Always return [].
Paper ID:
{paper_id}

Title:
{title}

Abstract:
{abstract}
"""
