from pathlib import Path
import uuid

import pymupdf

from src.fdi_ml.ingestion.models import DocumentPage, FinancialDocument


class PDFParser:

    def parse(self, file_path: str | Path) -> FinancialDocument:
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"PDF file not found: {file_path}"
            )

        if file_path.suffix.lower() != ".pdf":
            raise ValueError(
                f"Expected a PDF file, got: {file_path.suffix}"
            )

        document_id = str(uuid.uuid4())

        pages: list[DocumentPage] = []

        with pymupdf.open(file_path) as pdf:

            for page_number, page in enumerate(pdf, start=1):

                text = page.get_text("text").strip()

                pages.append(
                    DocumentPage(
                        page_number=page_number,
                        text=text,
                    )
                )

            metadata = {
                "page_count": len(pdf),
                "file_size_bytes": file_path.stat().st_size,
            }

        return FinancialDocument(
            document_id=document_id,
            filename=file_path.name,
            pages=pages,
            metadata=metadata,
        )


if __name__ == "__main__":
    parser = PDFParser()

    pdf_path = (
        Path("data")
        / "documents"
        / "benchmark"
        / "financebench"
        / "APPLE_2022_10K.pdf"
    )

    document = parser.parse(pdf_path)

    print("=" * 60)
    print("DOCUMENT")
    print("=" * 60)

    print(f"Document ID : {document.document_id}")
    print(f"Filename    : {document.filename}")
    print(f"Page count  : {document.page_count}")
    print(f"Metadata    : {document.metadata}")

    print("\n" + "=" * 60)
    print("FIRST PAGE")
    print("=" * 60)

    print(document.pages[0].text[:2000])
