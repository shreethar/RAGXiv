from abc import ABC, abstractmethod

from openai import OpenAI


EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 3072


class EmbeddingService(ABC):
    @abstractmethod
    def embed(
        self,
        *,
        texts: list[str],
    ) -> list[list[float]]:
        ...


class OpenAIEmbeddingService(EmbeddingService):
    def __init__(
        self,
        *,
        client: OpenAI,
        model: str = EMBEDDING_MODEL,
    ):
        self.client = client
        self.model = model

    def embed(
        self,
        *,
        texts: list[str],
    ) -> list[list[float]]:
        if not texts:
            return []

        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )

        embeddings = [
            item.embedding
            for item in response.data
        ]

        if len(embeddings) != len(texts):
            raise RuntimeError(
                "Embedding API returned a different number "
                "of embeddings than input texts"
            )

        if any(
            len(embedding) != EMBEDDING_DIMENSIONS
            for embedding in embeddings
        ):
            raise RuntimeError(
                f"Embedding API returned an unexpected "
                f"embedding dimension; expected "
                f"{EMBEDDING_DIMENSIONS}"
            )

        return embeddings