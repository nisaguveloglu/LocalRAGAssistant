"""SQLite database manager for document chunks and vector persistence."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import TracebackType

from domain.chunk import Chunk
from utils.logger import logger


class DatabaseManager:
    """Manages SQLite database connections, schema creation, and chunk storage."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.connection: sqlite3.Connection | None = None
        self.cursor: sqlite3.Cursor | None = None

    def __enter__(self) -> DatabaseManager:
        """Context manager entry point."""
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Context manager exit point with automatic commit or rollback."""
        if exc_type is not None:
            self.rollback()
            logger.error("Transaction rolled back due to error: %s", exc_val)
        else:
            self.commit()
        self.disconnect()

    def connect(self) -> None:
        """Connects to the SQLite database."""
        if self.connection is not None:
            return

        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        self.cursor = self.connection.cursor()
        logger.debug("Connected to SQLite database at %s", self.db_path)

    def disconnect(self) -> None:
        """Closes the active database connection."""
        if self.cursor is not None:
            self.cursor.close()
            self.cursor = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None
            logger.debug("Disconnected from SQLite database at %s", self.db_path)

    def _ensure_connected(self) -> tuple[sqlite3.Connection, sqlite3.Cursor]:
        """Ensures active database connection and cursor exist."""
        if self.connection is None or self.cursor is None:
            self.connect()
        if self.connection is None or self.cursor is None:
            raise RuntimeError("Failed to establish SQLite database connection.")
        return self.connection, self.cursor

    def initialize(self) -> None:
        """Connects to the database and initializes required table schemas."""
        self.connect()
        self.create_tables()

    def create_tables(self) -> None:
        """Creates table schemas if they do not already exist."""
        conn, cursor = self._ensure_connected()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                start_char INTEGER NOT NULL,
                end_char INTEGER NOT NULL,
                content TEXT NOT NULL,
                section_title TEXT,
                embedding TEXT
            );
            """
        )
        try:
            cursor.execute("ALTER TABLE chunks ADD COLUMN section_title TEXT")
        except sqlite3.OperationalError:
            pass

        conn.commit()

    def insert_chunk(self, chunk: Chunk) -> None:
        """Inserts a single Chunk instance into the database."""
        _, cursor = self._ensure_connected()

        embedding_json = (
            json.dumps(chunk.embedding) if chunk.embedding is not None else None
        )

        cursor.execute(
            """
            INSERT INTO chunks(
                filename,
                chunk_index,
                start_char,
                end_char,
                content,
                section_title,
                embedding
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk.filename,
                chunk.chunk_index,
                chunk.start_char,
                chunk.end_char,
                chunk.content,
                chunk.section_title,
                embedding_json,
            ),
        )

    def insert_chunks(self, chunks: list[Chunk]) -> None:
        """Bulk inserts multiple Chunk instances into the database."""
        if not chunks:
            return

        _, cursor = self._ensure_connected()

        payload = [
            (
                chunk.filename,
                chunk.chunk_index,
                chunk.start_char,
                chunk.end_char,
                chunk.content,
                chunk.section_title,
                json.dumps(chunk.embedding) if chunk.embedding is not None else None,
            )
            for chunk in chunks
        ]

        cursor.executemany(
            """
            INSERT INTO chunks(
                filename,
                chunk_index,
                start_char,
                end_char,
                content,
                section_title,
                embedding
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )

    def get_chunks(self) -> list[Chunk]:
        """Retrieves all stored Chunk instances from the database."""
        _, cursor = self._ensure_connected()

        cursor.execute(
            """
            SELECT
                filename,
                chunk_index,
                start_char,
                end_char,
                content,
                section_title,
                embedding
            FROM chunks
            ORDER BY filename, chunk_index
            """
        )
        rows = cursor.fetchall()
        chunks: list[Chunk] = []

        for row in rows:
            embedding = (
                json.loads(row["embedding"]) if row["embedding"] is not None else None
            )
            section_title = (
                row["section_title"] if "section_title" in row.keys() else None
            )

            chunks.append(
                Chunk(
                    filename=row["filename"],
                    chunk_index=row["chunk_index"],
                    start_char=row["start_char"],
                    end_char=row["end_char"],
                    content=row["content"],
                    section_title=section_title,
                    embedding=embedding,
                )
            )

        return chunks


    def update_embedding(
        self,
        filename: str,
        chunk_index: int,
        embedding: list[float],
    ) -> None:
        """Updates the vector embedding for a specific chunk."""
        _, cursor = self._ensure_connected()

        cursor.execute(
            """
            UPDATE chunks
            SET embedding = ?
            WHERE filename = ?
            AND chunk_index = ?
            """,
            (
                json.dumps(embedding),
                filename,
                chunk_index,
            ),
        )

    def clear_chunks(self) -> None:
        """Deletes all stored chunks from the database table."""
        _, cursor = self._ensure_connected()
        cursor.execute("DELETE FROM chunks")

    def commit(self) -> None:
        """Commits open transactions."""
        if self.connection is not None:
            self.connection.commit()

    def rollback(self) -> None:
        """Rolls back open transactions."""
        if self.connection is not None:
            self.connection.rollback()