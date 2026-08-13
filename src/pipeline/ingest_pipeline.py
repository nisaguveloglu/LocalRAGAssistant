"""Document ingestion pipeline orchestrator."""

from config import CHUNK_OVERLAP, CHUNK_SIZE, DATABASE_PATH, DOCUMENTS_PATH
from ingestion.chunker import TextChunker
from ingestion.embedder import SentenceTransformerEmbedder
from ingestion.loader import DocumentLoader
from storage.database import DatabaseManager
from utils.logger import logger


class IngestionPipeline:
    """Document ingestion pipeline orchestrates PDF extraction, chunking, embedding, and storage."""

    def __init__(self) -> None:
        self.loader = DocumentLoader(DOCUMENTS_PATH)
        self.chunker = TextChunker(
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        )
        self.embedder = SentenceTransformerEmbedder()
        self.database = DatabaseManager(DATABASE_PATH)

    def run(self) -> None:
        """Executes the complete document ingestion pipeline."""
        logger.info("Starting Document Ingestion Pipeline")

        self.database.initialize()
        self.database.clear_chunks()

        documents = self.loader.load_documents()
        total_documents = len(documents)
        total_chunks = 0

        for document in documents:
            logger.info("Processing document: %s", document.filename)

            chunks = self.chunker.split(document)
            chunks = self.embedder.embed_chunks(chunks)

            self.database.insert_chunks(chunks)

            total_chunks += len(chunks)
            logger.info("Created %d chunks for %s", len(chunks), document.filename)

        self.database.commit()
        self.database.disconnect()


        logger.info(
            "Ingestion completed. Processed %d documents into %d chunks.",
            total_documents,
            total_chunks,
        )