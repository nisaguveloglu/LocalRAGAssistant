from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from domain.chunk import Chunk


class DatabaseManager:
    """
    SQLite veritabanını yöneten sınıf.
    """

    def __init__(self, db_path: Path) -> None:

        self.db_path = Path(db_path)

        self.connection: sqlite3.Connection | None = None
        self.cursor: sqlite3.Cursor | None = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """
        Veritabanına bağlanır.
        """

        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.connection = sqlite3.connect(self.db_path)

        self.connection.row_factory = sqlite3.Row

        self.cursor = self.connection.cursor()

    def disconnect(self) -> None:
        """
        Veritabanı bağlantısını kapatır.
        """

        if self.connection is not None:

            self.connection.close()

            self.connection = None
            self.cursor = None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """
        Veritabanını başlatır.
        """

        self.connect()

        self.create_tables()

    def create_tables(self) -> None:
        """
        Gerekli tabloları oluşturur.
        """

        if self.cursor is None:
            raise RuntimeError("Database is not connected.")

        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks(

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                filename TEXT NOT NULL,

                chunk_index INTEGER NOT NULL,

                start_char INTEGER NOT NULL,

                end_char INTEGER NOT NULL,

                content TEXT NOT NULL,

                embedding TEXT

            );
            """
        )

        self.connection.commit()

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert_chunk(self, chunk: Chunk) -> None:
        """
        Veritabanına tek bir chunk ekler.
        """

        if self.cursor is None:
            raise RuntimeError("Database is not connected.")

        embedding = None

        if chunk.embedding is not None:

            embedding = json.dumps(chunk.embedding)

        self.cursor.execute(
            """
            INSERT INTO chunks(

                filename,
                chunk_index,
                start_char,
                end_char,
                content,
                embedding

            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                chunk.filename,
                chunk.chunk_index,
                chunk.start_char,
                chunk.end_char,
                chunk.content,
                embedding,
            ),
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_chunks(self) -> list[Chunk]:
        """
        Veritabanındaki bütün chunk'ları döndürür.
        """

        if self.cursor is None:
            raise RuntimeError("Database is not connected.")

        self.cursor.execute(
            """
            SELECT
                filename,
                chunk_index,
                start_char,
                end_char,
                content,
                embedding
            FROM chunks
            ORDER BY filename, chunk_index
            """
        )

        rows = self.cursor.fetchall()

        chunks: list[Chunk] = []

        for row in rows:

            embedding = None

            if row["embedding"] is not None:

                embedding = json.loads(row["embedding"])

            chunks.append(
                Chunk(
                    filename=row["filename"],
                    chunk_index=row["chunk_index"],
                    start_char=row["start_char"],
                    end_char=row["end_char"],
                    content=row["content"],
                    embedding=embedding,
                )
            )

        return chunks

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update_embedding(
        self,
        filename: str,
        chunk_index: int,
        embedding: list[float],
    ) -> None:
        """
        Belirli bir chunk'ın embedding bilgisini günceller.
        """

        if self.cursor is None:
            raise RuntimeError("Database is not connected.")

        self.cursor.execute(
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

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def clear_chunks(self) -> None:
        """
        Tablodaki tüm chunk'ları siler.
        """

        if self.cursor is None:
            raise RuntimeError("Database is not connected.")

        self.cursor.execute(
            "DELETE FROM chunks"
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def commit(self) -> None:
        """
        Yapılan değişiklikleri kaydeder.
        """

        if self.connection is not None:

            self.connection.commit()