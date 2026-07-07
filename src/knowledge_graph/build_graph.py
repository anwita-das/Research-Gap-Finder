"""Orchestrate knowledge graph construction from merged papers and extracted entities."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, Union

from src.knowledge_graph.builder import GraphBuilder
from src.knowledge_graph.exporter import GraphExporter
from src.knowledge_graph.validator import GraphValidationError, GraphValidator

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def build_knowledge_graph(
    merged_dir: Optional[Union[str, Path]] = None,
    entities_dir: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Build and export the knowledge graph from processed paper and entity data."""
    project_root = Path(__file__).resolve().parents[2]

    merged_path = Path(merged_dir) if merged_dir is not None else project_root / "data" / "processed" / "merged"
    entities_path = Path(entities_dir) if entities_dir is not None else project_root / "data" / "processed" / "entities"
    export_path = Path(output_path) if output_path is not None else project_root / "data" / "processed" / "knowledge_graph.json"

    builder = GraphBuilder()
    validator = GraphValidator()
    exporter = GraphExporter()

    papers = _load_papers(merged_path)
    for paper in papers:
        _add_paper_to_builder(builder, paper)

    entity_files = _load_entity_files(entities_path)
    if not entity_files:
        logger.warning("No entity files found in %s", entities_path)

    for entity_file in entity_files:
        _add_entities_to_builder(builder, entity_file)

    graph = builder.build_graph()

    try:
        validator.validate_graph(graph)
        logger.info("Knowledge graph validation succeeded")
    except GraphValidationError as exc:
        logger.error("Knowledge graph validation failed: %s", exc)

    export_path.parent.mkdir(parents=True, exist_ok=True)
    exporter.export_json(graph, export_path)
    logger.info("Knowledge graph exported to %s", export_path)
    return export_path


def _load_papers(merged_dir: Path) -> list[dict[str, Any]]:
    """Load paper metadata from merged JSON files."""
    if not merged_dir.exists():
        logger.warning("Merged papers directory does not exist: %s", merged_dir)
        return []

    if not merged_dir.is_dir():
        logger.error("Merged papers path is not a directory: %s", merged_dir)
        return []

    paper_files = sorted(merged_dir.glob("*.json"))
    if not paper_files:
        logger.warning("No merged paper files found in %s", merged_dir)
        return []

    papers: list[dict[str, Any]] = []
    seen_paper_ids: set[str] = set()

    for file_path in paper_files:
        try:
            with file_path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
        except json.JSONDecodeError as exc:
            logger.warning("Skipping malformed paper JSON %s: %s", file_path, exc)
            continue
        except OSError as exc:
            logger.warning("Could not read paper JSON %s: %s", file_path, exc)
            continue

        if not isinstance(loaded, list):
            logger.warning("Skipping paper JSON with unexpected structure: %s", file_path)
            continue

        for paper in loaded:
            if not isinstance(paper, dict):
                logger.warning("Skipping non-object paper entry in %s", file_path)
                continue

            paper_id = str(paper.get("paper_id", "")).strip()
            if not paper_id:
                logger.warning("Skipping paper entry without paper_id in %s", file_path)
                continue

            if paper_id in seen_paper_ids:
                logger.warning("Skipping duplicate paper_id %s from %s", paper_id, file_path)
                continue

            seen_paper_ids.add(paper_id)
            papers.append(paper)

    return papers


def _load_entity_files(entities_dir: Path) -> list[Path]:
    """Return entity JSON files from the entities directory."""
    if not entities_dir.exists():
        logger.warning("Entity directory does not exist: %s", entities_dir)
        return []

    if not entities_dir.is_dir():
        logger.error("Entity path is not a directory: %s", entities_dir)
        return []

    entity_files = sorted(entities_dir.glob("*.json"))
    if not entity_files:
        logger.warning("No entity files found in %s", entities_dir)
    return entity_files


def _add_paper_to_builder(builder: GraphBuilder, paper: dict[str, Any]) -> None:
    """Add a single paper to the builder using the existing API."""
    paper_id = str(paper.get("paper_id", "")).strip()
    if not paper_id:
        logger.warning("Skipping paper with empty paper_id")
        return

    try:
        builder.add_paper(paper)
        logger.info("Added paper %s", paper_id)
    except ValueError as exc:
        logger.warning("Could not add paper %s: %s", paper_id, exc)


def _add_entities_to_builder(builder: GraphBuilder, entity_file: Path) -> None:
    """Load semantic entities from a JSON file and attach them to the paper."""
    try:
        with entity_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        logger.warning("Skipping malformed entity JSON %s: %s", entity_file, exc)
        return
    except OSError as exc:
        logger.warning("Could not read entity JSON %s: %s", entity_file, exc)
        return

    if not isinstance(payload, dict):
        logger.warning("Skipping entity payload with unexpected structure: %s", entity_file)
        return

    paper_id = str(payload.get("paper_id", "")).strip()
    if not paper_id:
        logger.warning("Skipping entity payload without paper_id: %s", entity_file)
        return

    try:
        builder.add_semantic_entities_to_paper(paper_id, payload)
        logger.info("Added semantic entities for paper %s", paper_id)
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning("Could not attach semantic entities for paper %s: %s", paper_id, exc)


if __name__ == "__main__":
    build_knowledge_graph()
