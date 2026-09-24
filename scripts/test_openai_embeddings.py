import os

import pytest
from dotenv import load_dotenv
from openai import OpenAI

from app.infrastructure.embeddings import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    OpenAIEmbeddingService,
)


load_dotenv()


@pytest.mark.integration
def test_openai_embedding():
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not configured")

    client = OpenAI()
    service = OpenAIEmbeddingService(client=client)

    texts = [
        "This is a RAGxiv embedding integration test."
    ]

    embeddings = service.embed(texts=texts)

    assert len(embeddings) == 1
    assert len(embeddings[0]) == EMBEDDING_DIMENSIONS