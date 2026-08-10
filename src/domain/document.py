from dataclasses import dataclass


@dataclass(slots=True)
class Document:
    """
    Sisteme yüklenen bir belgeyi temsil eder.

    Attributes
    ----------
    filename : str
        Belgenin dosya adı.

    content : str
        Belgenin tamamının metin içeriği.
    """

    filename: str
    content: str