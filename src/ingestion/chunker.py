from domain.chunk import Chunk
from domain.document import Document


class TextChunker:
    """
    Belgeleri sabit boyutlu ve overlap destekli Chunk nesnelerine dönüştürür.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 100,
    ) -> None:
        """
        Parameters
        ----------
        chunk_size : int
            Her chunk'ın maksimum karakter sayısı.

        overlap : int
            Ardışık chunk'lar arasında ortak bulunacak karakter sayısı.
        """

        if chunk_size <= 0:
            raise ValueError("chunk_size sıfırdan büyük olmalıdır.")

        if overlap < 0:
            raise ValueError("overlap negatif olamaz.")

        if overlap >= chunk_size:
            raise ValueError("overlap değeri chunk_size'dan küçük olmalıdır.")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, document: Document) -> list[Chunk]:
        """
        Bir belgeyi Chunk listesine dönüştürür.

        Parameters
        ----------
        document : Document
            Parçalanacak belge.

        Returns
        -------
        list[Chunk]
            Oluşturulan chunk listesi.
        """

        chunks: list[Chunk] = []

        text = document.content.strip()

        if not text:
            return chunks

        step = self.chunk_size - self.overlap
        chunk_index = 0

        for start in range(0, len(text), step):

            end = min(start + self.chunk_size, len(text))

            chunk_text = text[start:end].strip()

            if not chunk_text:
                continue

            chunks.append(
                Chunk(
                    filename=document.filename,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end,
                    content=chunk_text,
                )
            )

            chunk_index += 1

            # Son chunk'a ulaştıysak döngüyü bitir.
            if end >= len(text):
                break

        return chunks