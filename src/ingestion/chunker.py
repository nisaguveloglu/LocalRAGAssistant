"""Semantic and Header-Aware text chunking strategies."""

import re

from domain.chunk import Chunk
from domain.document import Document


class SemanticChunker:
    """Splits document text using sentence boundaries and PDF header/section detection."""

    # Patterns for detecting section headers (e.g., "1. Introduction", "Section 2.1", "CHAPTER I", ALL CAPS TITLE)
    HEADER_PATTERNS = [
        re.compile(r"^(?:(?:bölüm|chapter|section|\d+(?:\.\d+)*)\s*[:.-]?\s*[\w\s]{2,60})$", re.IGNORECASE),
        re.compile(r"^\d+(?:\.\d+)*\s*[:.-]?\s*[A-ZÇĞİÖŞÜ0-9\s]{2,60}$"),
        re.compile(r"^[A-ZÇĞİÖŞÜ0-9\s:-]{3,60}$"),
    ]


    def __init__(
        self,
        chunk_size: int = 600,
        overlap: int = 100,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")
        if overlap < 0:
            raise ValueError("overlap cannot be negative.")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size.")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, document: Document) -> list[Chunk]:
        """Splits document into semantically bounded, header-aware Chunk instances."""
        text = document.content.strip()
        if not text:
            return []

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        chunks: list[Chunk] = []

        current_section: str | None = None
        current_sentences: list[str] = []
        current_length = 0
        chunk_index = 0
        char_offset = 0

        for line in lines:
            if self._is_header(line):
                if current_sentences:
                    chunk_text = " ".join(current_sentences).strip()
                    if chunk_text:
                        end_offset = char_offset + len(chunk_text)
                        chunks.append(
                            Chunk(
                                filename=document.filename,
                                chunk_index=chunk_index,
                                start_char=char_offset,
                                end_char=end_offset,
                                content=chunk_text,
                                section_title=current_section,
                            )
                        )
                        chunk_index += 1
                        char_offset = end_offset
                    current_sentences = []
                    current_length = 0
                current_section = line
                continue


            sentences = self._split_sentences(line)
            for sentence in sentences:
                sentence_len = len(sentence)

                if current_length + sentence_len > self.chunk_size and current_sentences:
                    chunk_text = " ".join(current_sentences).strip()
                    end_offset = char_offset + len(chunk_text)

                    chunks.append(
                        Chunk(
                            filename=document.filename,
                            chunk_index=chunk_index,
                            start_char=char_offset,
                            end_char=end_offset,
                            content=chunk_text,
                            section_title=current_section,
                        )
                    )
                    chunk_index += 1

                    # Overlap handling: retain tail sentences up to overlap length
                    overlap_sentences: list[str] = []
                    overlap_len = 0
                    for s in reversed(current_sentences):
                        if overlap_len + len(s) <= self.overlap:
                            overlap_sentences.insert(0, s)
                            overlap_len += len(s)
                        else:
                            break

                    char_offset = max(char_offset, end_offset - overlap_len)
                    current_sentences = overlap_sentences
                    current_length = overlap_len

                current_sentences.append(sentence)
                current_length += sentence_len

        if current_sentences:
            chunk_text = " ".join(current_sentences).strip()
            if chunk_text:
                chunks.append(
                    Chunk(
                        filename=document.filename,
                        chunk_index=chunk_index,
                        start_char=char_offset,
                        end_char=char_offset + len(chunk_text),
                        content=chunk_text,
                        section_title=current_section,
                    )
                )

        return chunks

    def _is_header(self, line: str) -> bool:
        """Determines if a given line is a section heading."""
        if len(line) > 80:
            return False
        for pattern in self.HEADER_PATTERNS:
            if pattern.match(line):
                return True
        return False

    def _split_sentences(self, text: str) -> list[str]:
        """Splits text string into sentence units preserving punctuation."""
        raw_sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in raw_sentences if s.strip()]


class TextChunker(SemanticChunker):
    """Backwards-compatible alias for SemanticChunker."""
    pass