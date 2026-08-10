from config import DATABASE_PATH
from config import DOCUMENTS_PATH

from ingestion.loader import DocumentLoader
from ingestion.chunker import TextChunker
from ingestion.embedder import SentenceTransformerEmbedder

from storage.database import DatabaseManager


class IngestionPipeline:
    """
    Belgeleri sisteme kazandıran (ingestion) pipeline.

    İş Akışı
    --------
    PDF
        ↓
    Document
        ↓
    Chunk
        ↓
    Embedding
        ↓
    SQLite
    """

    def __init__(self) -> None:

        self.loader = DocumentLoader(DOCUMENTS_PATH)

        self.chunker = TextChunker(
            chunk_size=500,
            overlap=100,
        )

        self.embedder = SentenceTransformerEmbedder()

        self.database = DatabaseManager(DATABASE_PATH)

    def run(self) -> None:
        """
        Ingestion işlemini başlatır.
        """

        print("=" * 60)
        print("Starting Ingestion Pipeline")
        print("=" * 60)
        print()

        self.database.initialize()
        self.database.clear_chunks()

        documents = self.loader.load_documents()

        total_documents = len(documents)
        total_chunks = 0

        for document in documents:

            print(f"Processing : {document.filename}")

            chunks = self.chunker.split(document)

            chunks = self.embedder.embed_chunks(chunks)

            for chunk in chunks:
                self.database.insert_chunk(chunk)

            total_chunks += len(chunks)

            print(f"  -> {len(chunks)} chunk created")

        self.database.commit()
        self.database.disconnect()

        print()
        print("=" * 60)
        print("Ingestion Completed")
        print("=" * 60)
        print(f"Documents : {total_documents}")
        print(f"Chunks    : {total_chunks}")