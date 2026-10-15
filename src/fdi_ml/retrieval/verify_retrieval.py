
from src.fdi_ml.embeddings.encoder import EmbeddingEncoder
from src.fdi_ml.retrieval.qdrant_store import QdrantVectorStore


def verify_retrieval() -> None:
    encoder = EmbeddingEncoder()

    vector_store = QdrantVectorStore(
        collection_name="financial_documents",
        storage_path="data/vector_store",
        embedding_dimension=384,
    )

    test_queries = [
        {
            "question": "What were Apple's total net sales in fiscal year 2022?",
            "expected_file": "APPLE_2022_10K.pdf",
        },
        {
            "question": "What were Amazon's net sales in fiscal year 2022?",
            "expected_file": "AMAZON_2022_10K.pdf",
        },
        {
            "question": "What was Adobe's total revenue in fiscal year 2022?",
            "expected_file": "ADOBE_2022_10K.pdf",
        },
        {
            "question": "What were 3M's net sales in fiscal year 2022?",
            "expected_file": "3M_2022_10K.pdf",
        },
        {
            "question": (
                "What revenue and financial results did "
                "Activision Blizzard report for 2022?"
            ),
            "expected_file": "ACTIVISIONBLIZZARD_2022_10K.pdf",
        },
    ]

    try:
        collection = vector_store.client.get_collection(
            "financial_documents"
        )

        print("=" * 70)
        print("MULTI-DOCUMENT RETRIEVAL VERIFICATION")
        print("=" * 70)
        print(f"Qdrant points: {collection.points_count}")
        print(f"Test queries: {len(test_queries)}")

        successful_queries = 0

        for test in test_queries:
            question = test["question"]
            expected_file = test["expected_file"]

            query_embedding = encoder.encode([question])[0]
            results = vector_store.search(
                query_embedding=query_embedding,
                top_k=5,
            )

            found_expected_file = any(
                chunk.filename == expected_file
                for chunk, score in results
            )

            if found_expected_file:
                successful_queries += 1

            print("\n" + "-" * 70)
            print(f"QUESTION: {question}")
            print(f"EXPECTED SOURCE: {expected_file}")
            print(
                "EXPECTED SOURCE IN TOP 5: "
                f"{'PASS' if found_expected_file else 'CHECK'}"
            )

            for rank, (chunk, score) in enumerate(results, start=1):
                snippet = " ".join(chunk.text.split())[:220]

                print(
                    f"\n{rank}. {chunk.filename} | "
                    f"Page {chunk.page_number} | "
                    f"Score: {score:.4f}"
                )
                print(f"   {snippet}...")

        print("\n" + "=" * 70)
        print("VERIFICATION SUMMARY")
        print("=" * 70)
        print(
            f"Expected source found in top 5: "
            f"{successful_queries}/{len(test_queries)}"
        )

        if collection.points_count != 8240:
            print(
                "NOTE: The collection point count differs from "
                "8,240. Check for points left over from earlier "
                "indexing runs before drawing final conclusions."
            )

    finally:
        vector_store.client.close()


if __name__ == "__main__":
    verify_retrieval()
