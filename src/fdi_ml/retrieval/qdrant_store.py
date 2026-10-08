from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.fdi_ml.ingestion.models import DocumentChunk


class QdrantVectorStore:

    def __init__(
        self,
        collection_name: str = "financial_documents",
        storage_path: str = "data/vector_store",
        embedding_dimension: int = 384,
    ):
        self.collection_name = collection_name

        Path(storage_path).mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client = QdrantClient(
            path=storage_path
        )

        self.embedding_dimension = embedding_dimension

        self._create_collection()

    def _create_collection(self) -> None:

        collections = self.client.get_collections()

        collection_names = [
            collection.name
            for collection in collections.collections
        ]

        if self.collection_name not in collection_names:

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dimension,
                    distance=Distance.COSINE,
                ),
            )

    def add(
        self,
        chunks: list[DocumentChunk],
        embeddings,
    ) -> None:

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks must match number of embeddings."
            )

        points = []

        for chunk, embedding in zip(
            chunks,
            embeddings,
        ):

            point = PointStruct(
                id=chunk.chunk_id,
                vector=embedding.tolist(),
                payload={
                    "document_id": chunk.document_id,
                    "filename": chunk.filename,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                },
            )

            points.append(point)

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def search(
        self,
        query_embedding,
        top_k: int = 5,
    ):

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            limit=top_k,
        )

        return [
            (
                DocumentChunk(
                    chunk_id=str(result.id),
                    document_id=result.payload["document_id"],
                    filename=result.payload["filename"],
                    page_number=result.payload["page_number"],
                    text=result.payload["text"],
                    chunk_index=result.payload["chunk_index"],
                    metadata=result.payload.get("metadata", {}),
                ),
                float(result.score),
            )
            for result in results.points
        ]
