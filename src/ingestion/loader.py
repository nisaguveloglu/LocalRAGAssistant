from pathlib import Path

from pypdf import PdfReader

from domain.document import Document


class DocumentLoader:
    """
    documents klasöründeki PDF belgelerini okuyarak
    Document nesnelerine dönüştürür.
    """

    def __init__(self, documents_path: Path) -> None:
        """
        Parameters
        ----------
        documents_path : Path
            PDF dosyalarının bulunduğu klasör.
        """

        self.documents_path = Path(documents_path)

    def load_documents(self) -> list[Document]:
        """
        documents klasöründeki tüm PDF dosyalarını yükler.

        Returns
        -------
        list[Document]
            Okunan belgelerin listesi.
        """

        documents: list[Document] = []

        if not self.documents_path.exists():
            raise FileNotFoundError(
                f"Documents directory not found: {self.documents_path}"
            )

        pdf_files = sorted(self.documents_path.glob("*.pdf"))

        for pdf_file in pdf_files:

            document = self._load_pdf(pdf_file)

            if document.content.strip():
                documents.append(document)

        return documents

    def _load_pdf(self, pdf_path: Path) -> Document:
        """
        Tek bir PDF dosyasını okuyarak Document nesnesi oluşturur.

        Parameters
        ----------
        pdf_path : Path
            Okunacak PDF dosyası.

        Returns
        -------
        Document
            Okunan belge.
        """

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
