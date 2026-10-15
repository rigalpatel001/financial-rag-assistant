
from pathlib import Path

from qdrant_client import QdrantClient


def inspect_qdrant() -> None:
    storage_path = Path("data/vector_store").resolve()
    collection_name = "financial_documents"

    print(f"Storage path: {storage_path}")
    print(f"Path exists: {storage_path.exists()}")

    client = QdrantClient(path=str(storage_path))

    try:
        info = client.get_collection(collection_name)
        print(f"Collection: {collection_name}")
        print(f"Reported point count: {info.points_count}")

        total = 0
        missing_filename = 0
        samples = []

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
                total += 1
                payload = point.payload or {}

                if not payload.get("filename"):
                    missing_filename += 1

                    if len(samples) < 5:
                        samples.append({
                            "id": str(point.id),
                            "payload_keys": list(payload.keys()),
                            "payload": payload,
                        })

            if next_offset is None:
                break

            offset = next_offset

        print(f"\nPoints inspected: {total}")
        print(f"Points missing filename: {missing_filename}")

        for index, sample in enumerate(samples, start=1):
            print(f"\nMalformed point {index}")
            print(f"ID: {sample['id']}")
            print(f"Payload keys: {sample['payload_keys']}")
            print(f"Payload: {sample['payload']}")

    finally:
        client.close()


if __name__ == "__main__":
    inspect_qdrant()
