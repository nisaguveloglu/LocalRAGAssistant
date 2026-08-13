"""Domain model representing a text chunk extracted from a document."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class Chunk:
    """Represents a text chunk extracted from a document with optional vector embedding."""

    filename: str
    chunk_index: int
    start_char: int
    end_char: int
    content: str
    section_title: str | None = field(default=None)
    embedding: list[float] | None = field(default=None)