from sentence_transformers import SentenceTransformer


class EmbeddingEncoder:

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]):
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )


if __name__ == "__main__":
    encoder = EmbeddingEncoder()

    texts = [
        "Apple reported strong revenue growth.",
        "Apple's sales increased significantly.",
        "The weather is sunny today.",
    ]

    embeddings = encoder.encode(texts)

    print("=" * 60)
    print("EMBEDDING RESULTS")
    print("=" * 60)

    print(f"Number of texts : {len(texts)}")
    print(f"Embedding shape : {embeddings.shape}")

    print("\nFirst embedding:")
    print(embeddings[0])
