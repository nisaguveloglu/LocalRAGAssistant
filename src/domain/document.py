"""Domain model representing a source document."""

from dataclasses import dataclass


@dataclass(slots=True)
class Document:
    """Represents a source document loaded from disk."""

    filename: str
    content: str