import json
import tempfile
from pathlib import Path

from src.knowledge_graph.build_graph import build_knowledge_graph


tmp = Path(tempfile.mkdtemp())
merged = tmp / "merged"
entities = tmp / "entities"
merged.mkdir()
entities.mkdir()

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

entities_payload = {
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

(merged / "papers.json").write_text(json.dumps([paper]), encoding="utf-8")
(entities / "paper-1.json").write_text(json.dumps(entities_payload), encoding="utf-8")
out_path = tmp / "knowledge_graph.json"
result = build_knowledge_graph(merged_dir=merged, entities_dir=entities, output_path=out_path)
print(result)
print(out_path.exists())
data = json.loads(out_path.read_text(encoding="utf-8"))
print(sorted(data.keys()))
print(len(data.get("nodes", [])), len(data.get("links", [])))
