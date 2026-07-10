"""Top-level package for the research knowledge graph navigator."""

from __future__ import annotations

# import importlib
# import sys

# # Provide an import alias for the semantic_search package so both the requested
# # package path and the existing implementation location resolve correctly.
# try:
#     semantic_search_module = importlib.import_module("src.Semantic_search")
# except ModuleNotFoundError:  # pragma: no cover - fallback for unusual layouts
#     semantic_search_module = None

# if semantic_search_module is not None:
#     sys.modules.setdefault("src.semantic_search", semantic_search_module)
#     for module_name in [
#         "src.Semantic_search.config",
#         "src.Semantic_search.types",
#         "src.Semantic_search.graph_loader",
#         "src.Semantic_search.embedding_service",
#         "src.Semantic_search.faiss_index",
#         "src.Semantic_search.retrieval",
#         "src.Semantic_search.ranking",
#         "src.Semantic_search.graph_connector",
#         "src.Semantic_search.search_service",
#         "src.Semantic_search.cli",
#     ]:
#         try:
#             module = importlib.import_module(module_name)
#         except ModuleNotFoundError:
#             continue
#         sys.modules.setdefault(module_name.replace("src.Semantic_search", "src.semantic_search"), module)
