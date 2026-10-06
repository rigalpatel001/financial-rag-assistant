import numpy as np

from src.fdi_ml.embeddings.encoder import EmbeddingEncoder
from src.fdi_ml.ingestion.models import DocumentChunk


class SemanticRetriever:

    def __init__(
        self,
        embedding_encoder: EmbeddingEncoder,
    ):
        self.embedding_encoder = embedding_encoder

    def retrieve(
        self,
        query: str,
        chunks: list[DocumentChunk],
        chunk_embeddings: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[DocumentChunk, float]]:

        if not chunks:
            return []

        if len(chunks) != len(chunk_embeddings):
            raise ValueError(
                "Number of chunks must match number of embeddings."
            )

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        query_embedding = self.embedding_encoder.encode([query])[0]

        scores = chunk_embeddings @ query_embedding

        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = [
            (chunks[index], float(scores[index]))
            for index in ranked_indices
        ]

        return results


if __name__ == "__main__":
    from pathlib import Path

    from src.fdi_ml.ingestion.chunking import FixedSizeChunker
    from src.fdi_ml.ingestion.pdf_parser import PDFParser

    pdf_path = (
        Path("data")
        / "documents"
        / "benchmark"
        / "financebench"
        / "APPLE_2022_10K.pdf"
    )

    # 1. Parse document
    parser = PDFParser()
    document = parser.parse(pdf_path)

    # 2. Create chunks
    chunker = FixedSizeChunker(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = chunker.chunk(document)

    # 3. Create embeddings for all chunks
    encoder = EmbeddingEncoder()

    chunk_texts = [chunk.text for chunk in chunks]

    chunk_embeddings = encoder.encode(chunk_texts)

    # 4. Create retriever
    retriever = SemanticRetriever(encoder)

    # 5. Ask a question
    query = "What was Apple's total net sales in 2022?"

    results = retriever.retrieve(
        query=query,
        chunks=chunks,
        chunk_embeddings=chunk_embeddings,
        top_k=5,
    )

    # 6. Display results
    print("=" * 60)
    print("SEMANTIC RETRIEVAL")
    print("=" * 60)

    print(f"Query: {query}")
    print(f"Retrieved chunks: {len(results)}")

    for rank, (chunk, score) in enumerate(results, start=1):

        print("\n" + "-" * 60)

        print(f"Rank       : {rank}")
        print(f"Score      : {score:.4f}")
        print(f"Page       : {chunk.page_number}")
        print(f"Chunk index: {chunk.chunk_index}")

        print("\nText:")
        print(chunk.text)
