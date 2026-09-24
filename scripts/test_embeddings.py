from unittest.mock import Mock

import pytest

from app.infrastructure.embeddings import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    OpenAIEmbeddingService,
)


def make_embedding(dimension: int = EMBEDDING_DIMENSIONS) -> list[float]:
    return [0.0] * dimension


def test_embed_empty_input_returns_empty_list():
    client = Mock()
    service = OpenAIEmbeddingService(client=client)

    result = service.embed(texts=[])

    assert result == []
    client.embeddings.create.assert_not_called()


def test_embed_returns_embeddings():
    client = Mock()

    client.embeddings.create.return_value.data = [
        Mock(embedding=make_embedding()),
        Mock(embedding=make_embedding()),
    ]

    service = OpenAIEmbeddingService(client=client)

    texts = [
        "First paper chunk",
        "Second paper chunk",
    ]

    result = service.embed(texts=texts)

    assert len(result) == 2
    assert len(result[0]) == EMBEDDING_DIMENSIONS
    assert len(result[1]) == EMBEDDING_DIMENSIONS


def test_embed_sends_correct_model_and_texts():
    client = Mock()

    client.embeddings.create.return_value.data = [
        Mock(embedding=make_embedding()),
        Mock(embedding=make_embedding()),
    ]

    service = OpenAIEmbeddingService(client=client)

    texts = [
        "First paper chunk",
        "Second paper chunk",
    ]

    service.embed(texts=texts)

    client.embeddings.create.assert_called_once_with(
        model=EMBEDDING_MODEL,
        input=texts,
    )


def test_embed_rejects_mismatched_embedding_count():
    client = Mock()

    client.embeddings.create.return_value.data = [
        Mock(embedding=make_embedding()),
    ]

    service = OpenAIEmbeddingService(client=client)

    texts = [
        "First paper chunk",
        "Second paper chunk",
    ]

    with pytest.raises(RuntimeError, match="different number"):
        service.embed(texts=texts)


def test_embed_rejects_wrong_embedding_dimension():
    client = Mock()

    client.embeddings.create.return_value.data = [
        Mock(embedding=make_embedding(100)),
    ]

    service = OpenAIEmbeddingService(client=client)

    with pytest.raises(RuntimeError, match="unexpected embedding dimension"):
        service.embed(texts=["Some text"])