
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList


def clean_qdrant() -> None:
    storage_path = str(Path("data/vector_store").resolve())
    collection_name = "financial_documents"

    client = QdrantClient(path=storage_path)

    try:
        # First scan the collection without deleting anything.
        malformed_ids = []
        offset = None

        while True:
            points, next_offset = client.scroll(
                collection_name=collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in points:
                payload = point.payload or {}

                if not payload.get("filename"):
                    malformed_ids.append(point.id)

            if next_offset is None:
                break

            offset = next_offset

        print(f"Storage path: {storage_path}")
        print(f"Malformed points found: {len(malformed_ids)}")

        if not malformed_ids:
            print("Nothing to clean.")
            return

        # Delete only the identified malformed point IDs.
        client.delete(
            collection_name=collection_name,
            points_selector=PointIdsList(points=malformed_ids),
            wait=True,
        )

        # Verify the result.
        info = client.get_collection(collection_name)
        print(f"Remaining points: {info.points_count}")

        remaining_bad = 0
        offset = None

        while True:
            points, next_offset = client.scroll(
                collection_name=collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            remaining_bad += sum(
                1
                for point in points
                if not (point.payload or {}).get("filename")
            )

            if next_offset is None:
                break

            offset = next_offset

        print(f"Points still missing filename: {remaining_bad}")

    finally:
        client.close()


if __name__ == "__main__":
    clean_qdrant()
