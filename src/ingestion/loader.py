"""PDF document loader for text extraction."""

from pathlib import Path

from pypdf import PdfReader

from domain.document import Document
from utils.logger import logger


class DocumentLoader:
    """Loads PDF files from a directory and extracts their text contents."""

    def __init__(self, documents_path: Path) -> None:
        self.documents_path = Path(documents_path)

    def load_documents(self) -> list[Document]:
        """Loads all non-empty PDF documents from the configured directory."""
        documents: list[Document] = []

        if not self.documents_path.exists():
            logger.error("Documents directory not found: %s", self.documents_path)
            raise FileNotFoundError(
                f"Documents directory not found: {self.documents_path}"
            )

        pdf_files = sorted(self.documents_path.glob("*.pdf"))
        if not pdf_files:
            logger.warning("No PDF files found in %s", self.documents_path)

        for pdf_file in pdf_files:
            try:
                document = self._load_pdf(pdf_file)
                if document.content.strip():
                    documents.append(document)
                else:
                    logger.warning("Skipping empty PDF document: %s", pdf_file.name)
            except Exception as err:
                logger.error("Failed to parse PDF document %s: %s", pdf_file.name, err)

        return documents

    def _load_pdf(self, pdf_path: Path) -> Document:
        """Extracts text content from a single PDF file."""
        reader = PdfReader(pdf_path)
        pages: list[str] = []

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                pages.append(page_text.strip())

        content = "\n\n".join(pages)

        return Document(
            filename=pdf_path.name,
            content=content,
        )

