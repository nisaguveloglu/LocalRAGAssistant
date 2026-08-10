from dataclasses import dataclass, field


@dataclass(slots=True)
class Chunk:
    """
    Bir belgeden üretilen metin parçasını temsil eder.

    Attributes
    ----------
    filename : str
        Chunk'ın ait olduğu dosya adı.

    chunk_index : int
        Aynı belge içerisindeki sıra numarası.

    start_char : int
        Belgedeki başlangıç karakter indeksi.

    end_char : int
        Belgedeki bitiş karakter indeksi.

    content : str
        Chunk'ın metin içeriği.

    embedding : list[float] | None
        Embedding oluşturulduktan sonra doldurulacak vektör.
        İlk oluşturulduğunda None'dır.
    """

    filename: str
    chunk_index: int

    start_char: int
    end_char: int

    content: str

    embedding: list[float] | None = field(default=None)