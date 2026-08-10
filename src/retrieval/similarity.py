import math


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    """
    İki embedding vektörü arasındaki cosine similarity değerini hesaplar.

    Parameters
    ----------
    vector_a : list[float]
        İlk embedding vektörü.

    vector_b : list[float]
        İkinci embedding vektörü.

    Returns
    -------
    float
        Cosine similarity değeri.
        Sonuç -1 ile 1 arasındadır.

        1.0  -> Tamamen aynı yön
        0.0  -> İlişkisiz
        -1.0 -> Tamamen zıt yön
    """

    if len(vector_a) != len(vector_b):
        raise ValueError(
            "Embedding boyutları aynı olmalıdır."
        )

    dot_product = sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    magnitude_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)