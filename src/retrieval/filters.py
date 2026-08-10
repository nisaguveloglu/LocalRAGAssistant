from domain.chunk import Chunk


class ChunkFilter:
    """
    Filters low quality chunks before they are sent to the LLM.
    """

    INVALID_PREFIXES = (
        "şekil",
        "figure",
        "tablo",
        "table",
        "içindekiler",
        "contents",
        "kaynakça",
        "references",
    )

    MIN_LENGTH = 100

    @classmethod
    def is_valid(cls, chunk: Chunk) -> bool:

        text = chunk.content.strip().lower()

        if len(text) < cls.MIN_LENGTH:
            return False

        if text.startswith(cls.INVALID_PREFIXES):
            return False

        return True