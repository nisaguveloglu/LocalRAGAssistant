"""Vector embedding generation interface and implementation."""

from abc import ABC, abstractmethod

from sentence_transformers import SentenceTransformer

from domain.chunk import Chunk


class BaseEmbedder(ABC):
    """Abstract base class for embedding generators."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Generates a vector embedding for a text string."""
        raise NotImplementedError

    def embed_chunk(self, chunk: Chunk) -> Chunk:
        """Populates embedding vector for a single Chunk."""
        chunk.embedding = self.embed(chunk.content)
        return chunk

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Populates embedding vectors for a list of Chunks."""
        for chunk in chunks:
            self.embed_chunk(chunk)
        return chunks


class SentenceTransformerEmbedder(BaseEmbedder):
    """Embedding generator utilizing SentenceTransformers."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
    ) -> None:
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        """Encodes text into a normalized float embedding vector."""
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding.tolist()