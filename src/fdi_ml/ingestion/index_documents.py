from pathlib import Path

from src.fdi_ml.embeddings.encoder import EmbeddingEncoder
from src.fdi_ml.ingestion.chunking import FixedSizeChunker
from src.fdi_ml.ingestion.pdf_parser import PDFParser
from src.fdi_ml.retrieval.qdrant_store import QdrantVectorStore


def index_documents() -> None:
    documents_dir = (
        Path("data")
        / "documents"
        / "benchmark"
        / "financebench"
    )

    pdf_files = sorted(documents_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {documents_dir.resolve()}"
        )

    parser = PDFParser()
    chunker = FixedSizeChunker(
        chunk_size=1000,
        chunk_overlap=200,
    )
    encoder = EmbeddingEncoder()

    vector_store = QdrantVectorStore(
        collection_name="financial_documents",
        storage_path="data/vector_store",
        embedding_dimension=384,
    )

    total_chunks = 0

    try:
        print("=" * 65)
        print("FINANCIAL DOCUMENT BATCH INDEXING")
        print("=" * 65)
        print(f"PDF files found: {len(pdf_files)}")

        for pdf_file in pdf_files:
            print(f"\nProcessing: {pdf_file.name}")

            document = parser.parse(pdf_file)
            chunks = chunker.chunk(document)

            if not chunks:
                print("WARNING: No text chunks extracted.")
                continue

            texts = [chunk.text for chunk in chunks]
            embeddings = encoder.encode(texts)

            vector_store.add(
                chunks=chunks,
                embeddings=embeddings,
            )

            total_chunks += len(chunks)

            print(f"  Pages: {document.page_count}")
            print(f"  Chunks indexed: {len(chunks)}")
            print("  Status: SUCCESS")

        print("\n" + "=" * 65)
        print("INDEXING SUMMARY")
        print("=" * 65)
        print(f"PDF files processed: {len(pdf_files)}")
        print(f"Chunks submitted: {total_chunks}")
        print("Qdrant collection: financial_documents")
        print("Indexing complete.")

    finally:
        vector_store.client.close()


if __name__ == "__main__":
    index_documents()
