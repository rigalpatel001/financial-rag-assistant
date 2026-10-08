from uuid import uuid4

from src.fdi_ml.ingestion.models import DocumentChunk, FinancialDocument


class FixedSizeChunker:

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(
        self,
        document: FinancialDocument,
    ) -> list[DocumentChunk]:

        chunks: list[DocumentChunk] = []
        chunk_index = 0

        for page in document.pages:

            text = page.text.strip()

            if not text:
                continue

            start = 0

            while start < len(text):

                end = start + self.chunk_size

                chunk_text = text[start:end].strip()

                if chunk_text:
                    chunks.append(
                        DocumentChunk(
                            chunk_id=str(uuid4()),
                            document_id=document.document_id,
                            filename=document.filename,
                            page_number=page.page_number,
                            text=chunk_text,
                            chunk_index=chunk_index,
                        )
                    )

                    chunk_index += 1

                if end >= len(text):
                    break

                start = end - self.chunk_overlap

        return chunks


if __name__ == "__main__":
    from pathlib import Path

    from src.fdi_ml.ingestion.pdf_parser import PDFParser

    pdf_path = (
        Path("data")
        / "documents"
        / "benchmark"
        / "financebench"
        / "APPLE_2022_10K.pdf"
    )

    # Parse PDF
    parser = PDFParser()
    document = parser.parse(pdf_path)

    # Create chunks
    chunker = FixedSizeChunker(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = chunker.chunk(document)

    # Inspect results
    print("=" * 60)
    print("CHUNKING RESULTS")
    print("=" * 60)

    print(f"Document       : {document.filename}")
    print(f"Pages          : {document.page_count}")
    print(f"Total chunks   : {len(chunks)}")

    print("\n" + "=" * 60)
    print("FIRST CHUNK")
    print("=" * 60)

    print(f"Chunk ID       : {chunks[0].chunk_id}")
    print(f"Page           : {chunks[0].page_number}")
    print(f"Chunk index    : {chunks[0].chunk_index}")
    print(f"Character count: {len(chunks[0].text)}")
    print()
    print(chunks[0].text)

    print("\n" + "=" * 60)
    print("SECOND CHUNK")
    print("=" * 60)

    print(chunks[1].text)

    print("\n" + "=" * 60)
    print("LAST CHUNK")
    print("=" * 60)

    print(f"Page           : {chunks[-1].page_number}")
    print(f"Chunk index    : {chunks[-1].chunk_index}")
    print(f"Character count: {len(chunks[-1].text)}")
    print()
    print(chunks[-1].text)
