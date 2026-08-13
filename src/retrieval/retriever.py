"""Semantic vector retrieval engine."""

from config import DATABASE_PATH
from domain.search_result import SearchResult
from retrieval.similarity import cosine_similarity
from storage.database import DatabaseManager


class Retriever:
    """Retrieves top-k relevant text chunks based on vector cosine similarity."""

    def __init__(self) -> None:
        self.database = DatabaseManager(DATABASE_PATH)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Performs vector similarity search and returns top-k SearchResult items."""
        self.database.connect()
        chunks = self.database.get_chunks()
        self.database.disconnect()

        results: list[SearchResult] = []

        for chunk in chunks:
            if chunk.embedding is None:
                continue

            score = cosine_similarity(
                query_embedding,
                chunk.embedding,
            )

            results.append(
                SearchResult(
                    chunk=chunk,
                    score=score,
                )
            )

        results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return results[:top_k]