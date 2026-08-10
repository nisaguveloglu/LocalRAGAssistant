from dataclasses import dataclass

from domain.chunk import Chunk


@dataclass(slots=True)
class SearchResult:
    """
    Retrieval aşamasında elde edilen arama sonucunu temsil eder.

    Attributes
    ----------
    chunk : Chunk
        Eşleşen metin parçası.

    score : float
        Sorgu ile chunk arasındaki benzerlik skoru
        (örneğin cosine similarity).
    """

    chunk: Chunk
    score: float