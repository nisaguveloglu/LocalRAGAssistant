from abc import ABC, abstractmethod

from sentence_transformers import SentenceTransformer

from domain.chunk import Chunk


class BaseEmbedder(ABC):
    """
    Tüm embedding sınıfları için temel arayüz.
    """

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """
        Verilen metin için embedding üretir.
        """
        raise NotImplementedError

    def embed_chunk(self, chunk: Chunk) -> Chunk:
        """
        Chunk nesnesinin embedding alanını doldurur.
        """

        chunk.embedding = self.embed(chunk.content)

        return chunk

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Birden fazla chunk için embedding üretir.
        """

        for chunk in chunks:
            self.embed_chunk(chunk)

        return chunks


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    SentenceTransformer tabanlı embedding üreticisi.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
    ) -> None:

        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        """
        Metni embedding vektörüne dönüştürür.
        """

        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embedding.tolist()