import logging
from pathlib import Path


def setup_logger(
    name: str = "LocalRAGAssistant",
    log_level: int = logging.INFO,
) -> logging.Logger:
    """
    Proje genelinde kullanılacak logger'ı oluşturur.

    Parameters
    ----------
    name : str
        Logger adı.

    log_level : int
        Logging seviyesi.

    Returns
    -------
    logging.Logger
        Yapılandırılmış logger nesnesi.
    """

    logger = logging.getLogger(name)

    # Aynı logger'ın tekrar tekrar handler eklemesini önle
    if logger.handlers:
        return logger

    logger.setLevel(log_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Konsol çıktısı
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    # logs klasörünü oluştur
    log_directory = Path("logs")
    log_directory.mkdir(exist_ok=True)

    # Dosya çıktısı
    file_handler = logging.FileHandler(
        log_directory / "local_rag_assistant.log",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    logger.propagate = False

    return logger


# Proje genelinde kullanılacak ortak logger
logger = setup_logger()