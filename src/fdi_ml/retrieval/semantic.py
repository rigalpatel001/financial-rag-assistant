import numpy as np

from src.fdi_ml.embeddings.encoder import EmbeddingEncoder
from src.fdi_ml.ingestion.models import DocumentChunk
# from src.fdi_ml.retrieval.vector_store import InMemoryVectorStore
from src.fdi_ml.retrieval.qdrant_store import QdrantVectorStore


class SemanticRetriever:

    def __init__(
        self,
        embedding_encoder: EmbeddingEncoder,
        vector_store: InMemoryVectorStore,
    ):
        self.embedding_encoder = embedding_encoder
        self.vector_store = vector_store

    def index(
        self,
        chunks: list[DocumentChunk],
    ) -> None:

        chunk_texts = [chunk.text for chunk in chunks]

        chunk_embeddings = self.embedding_encoder.encode(
            chunk_texts
        )

        self.vector_store.add(
            chunks=chunks,
            embeddings=chunk_embeddings,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[DocumentChunk, float]]:

        query_embedding = self.embedding_encoder.encode(
            [query]
        )[0]

        return self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )


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

    # 3. Create embedding model
    encoder = EmbeddingEncoder()

    # 4. Create vector store
    # vector_store = InMemoryVectorStore()
    vector_store = QdrantVectorStore(
        collection_name="financial_documents",
    )

    # 5. Create retriever
    retriever = SemanticRetriever(
        embedding_encoder=encoder,
        vector_store=vector_store,
    )

    # 6. Index chunks
    retriever.index(chunks)

    # 7. Search
    query = "What was Apple's total net sales in 2022?"

    results = retriever.retrieve(
        query=query,
        top_k=5,
    )

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
