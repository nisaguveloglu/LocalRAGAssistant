"""Domain model representing a vector similarity search result."""

from dataclasses import dataclass

from domain.chunk import Chunk


@dataclass(slots=True)
class SearchResult:
    """Represents a retrieved chunk with its similarity score."""

    chunk: Chunk
    score: float