import json
from pathlib import Path

from src.knowledge_graph.build_graph import build_knowledge_graph


def test_build_knowledge_graph_writes_export(tmp_path: Path) -> None:
    merged_dir = tmp_path / "merged"
    entities_dir = tmp_path / "entities"
    output_path = tmp_path / "knowledge_graph.json"

    merged_dir.mkdir()
    entities_dir.mkdir()

    paper = {
        "paper_id": "paper-1",
        "title": "Test Paper",
        "abstract": "A test abstract",
        "authors": ["Alice Smith"],
        "year": 2024,
        "venue": "Test Venue",
        "doi": "10.1000/test",
        "arxiv_id": "1234.5678",
        "openalex_id": "W123456",
        "keywords": ["test"],
        "fields_of_study": ["computer science"],
        "citations": [],
        "references": [],
        "url": "https://example.com/paper",
        "pdf_url": "https://example.com/paper.pdf",
        "source": ["arxiv"],
        "metadata": {
            "citation_count": 1,
            "reference_count": 0,
            "publication_date": "2024-01-01",
            "updated_at": "2024-01-01",
        },
    }

    entities = {
        "paper_id": "paper-1",
        "methods": ["Cross Validation"],
        "models": ["Transformer"],
        "algorithms": ["Gradient Descent"],
        "datasets": ["ImageNet"],
        "tasks": ["Classification"],
        "metrics": ["Accuracy"],
        "claims": ["The method improves accuracy"],
        "keywords": ["transformers"],
        "summary": "A summary",
    }

    (merged_dir / "papers.json").write_text(json.dumps([paper]), encoding="utf-8")
    (entities_dir / "paper-1.json").write_text(json.dumps(entities), encoding="utf-8")

    result_path = build_knowledge_graph(
        merged_dir=merged_dir,
        entities_dir=entities_dir,
        output_path=output_path,
    )

    assert result_path == output_path
    assert output_path.exists()

    exported = json.loads(output_path.read_text(encoding="utf-8"))
    assert "nodes" in exported
    assert "links" in exported
    assert any(node.get("label") == "Paper" for node in exported["nodes"])
