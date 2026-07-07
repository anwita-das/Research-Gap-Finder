import re
from .schema import PAPER_PREFIX, AUTHOR_PREFIX, VENUE_PREFIX
from typing import Any, Mapping, Optional, TypeVar, cast
T = TypeVar("T")
__all__ = [
    "normalize_name",
    "generate_paper_id",
    "generate_author_id",
    "generate_venue_id",
    "get_optional_field",
]
def normalize_name(text: str) -> str:
    """Normalize a name for use in identifier generation.
    The normalization process lowercases the input, trims surrounding
    whitespace,
    collapses repeated spaces, replaces spaces with underscores, and
    strips
    unnecessary punctuation.
    """
    if not text or not text.strip():
        raise ValueError("Name cannot be empty.")
    normalized = text.strip().lower()

    # Remove punctuation by replacing it with spaces
    normalized = re.sub(r"[^\w\s]", " ", normalized)

    # Collapse multiple spaces
    normalized = re.sub(r"\s+", " ", normalized).strip()

    # Replace spaces with underscores
    normalized = normalized.replace(" ", "_")

    return normalized

def generate_paper_id(paper_id: str) -> str:
    """Return a canonical paper node identifier."""
    return f"{PAPER_PREFIX}:{paper_id}"

def generate_author_id(name: str) -> str:
    """Return a canonical author node identifier."""
    return f"{AUTHOR_PREFIX}:{normalize_name(name)}"

def generate_venue_id(name: str) -> str:
    """Return a canonical venue node identifier."""
    return f"{VENUE_PREFIX}:{normalize_name(name)}"

def get_optional_field(
    data: Mapping[str, Any], field_name: str, default: Optional[T] =
    None
    ) -> Optional[T]:
    """Get a value from a mapping and return a fallback when absent or
    None."""
    if field_name not in data:
        return default
    value = data[field_name]
    return default if value is None else cast(Optional[T], value)