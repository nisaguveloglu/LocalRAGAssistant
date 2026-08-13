"""Logging configuration for Local RAG Assistant."""

import logging
from pathlib import Path


def setup_logger(
    name: str = "LocalRAGAssistant",
    log_level: int = logging.INFO,
) -> logging.Logger:
    """Configures and returns a thread-safe logger with console and file handlers."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(log_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_directory = Path("logs")
    log_directory.mkdir(exist_ok=True)

    file_handler = logging.FileHandler(
        log_directory / "local_rag_assistant.log",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


logger = setup_logger()