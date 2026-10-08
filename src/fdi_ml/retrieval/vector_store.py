import numpy as np

from src.fdi_ml.ingestion.models import DocumentChunk


class InMemoryVectorStore:

    def __init__(self):
        self.chunks: list[DocumentChunk] = []
        self.embeddings: np.ndarray | None = None

    def add(
        self,
        chunks: list[DocumentChunk],
        embeddings: np.ndarray,
    ) -> None:

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks must match number of embeddings."
            )

        self.chunks = chunks
        self.embeddings = embeddings

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[DocumentChunk, float]]:

        if self.embeddings is None:
            raise ValueError("Vector store is empty.")

        scores = self.embeddings @ query_embedding

        indices = np.argsort(scores)[::-1][:top_k]

        return [
            (self.chunks[index], float(scores[index]))
            for index in indices
        ]
