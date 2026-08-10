from storage.database import DatabaseManager
from retrieval.similarity import cosine_similarity

from domain.chunk import Chunk
from domain.search_result import SearchResult

from config import DATABASE_PATH


class Retriever:
    """
    Veritabanındaki chunk'lar arasında semantik arama gerçekleştirir.
    """

    def __init__(self) -> None:
        self.database = DatabaseManager(DATABASE_PATH)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        Query embedding'i ile en benzer chunk'ları döndürür.

        Parameters
        ----------
        query_embedding : list[float]
            Kullanıcının sorusuna ait embedding.

        top_k : int
            Döndürülecek maksimum sonuç sayısı.

        Returns
        -------
        list[SearchResult]
        """

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