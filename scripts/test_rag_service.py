from uuid import uuid4
from unittest.mock import Mock

from app.infrastructure.vector_store import RetrievedChunk
from app.services.rag_service import RAGService


def make_service():
    embedding_service = Mock()
    vector_store = Mock()
    llm_service = Mock()

    service = RAGService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        llm_service=llm_service,
    )

    return service, embedding_service, vector_store, llm_service


def test_query_paper_runs_retrieval_and_generation():
    service, embedding_service, vector_store, llm_service = (
        make_service()
    )

    paper_id = uuid4()

    embedding_service.embed.return_value = [
        [0.1, 0.2, 0.3]
    ]

    chunks = [
        RetrievedChunk(
            chunk_id="chunk-1",
            text="Self-attention allows tokens to attend to other tokens.",
            score=0.12,
            chunk_index=3,
        ),
        RetrievedChunk(
            chunk_id="chunk-2",
            text="The Transformer uses multi-head attention.",
            score=0.18,
            chunk_index=4,
        ),
    ]

    vector_store.search.return_value = chunks

    llm_service.generate.return_value = (
        "Self-attention allows tokens to attend to other tokens."
    )

    result = service.query_paper(
        question="How does self-attention work?",
        paper_id=paper_id,
        limit=2,
    )

    assert result.query == "How does self-attention work?"
    assert (
        result.answer
        == "Self-attention allows tokens to attend to other tokens."
    )
    assert result.chunks == chunks

    embedding_service.embed.assert_called_once_with(
        texts=["How does self-attention work?"]
    )

    vector_store.search.assert_called_once_with(
        paper_id=paper_id,
        query_embedding=[0.1, 0.2, 0.3],
        limit=2,
    )

    llm_service.generate.assert_called_once()

def test_query_paper_includes_retrieved_chunks_in_llm_prompt():
    service, embedding_service, vector_store, llm_service = (
        make_service()
    )

    paper_id = uuid4()

    embedding_service.embed.return_value = [
        [0.1, 0.2, 0.3]
    ]

    vector_store.search.return_value = [
        RetrievedChunk(
            chunk_id="chunk-1",
            text="Attention uses queries, keys, and values.",
            score=0.10,
            chunk_index=5,
        ),
        RetrievedChunk(
            chunk_id="chunk-2",
            text="Multi-head attention uses multiple attention heads.",
            score=0.20,
            chunk_index=6,
        ),
    ]

    llm_service.generate.return_value = "Answer"

    service.query_paper(
        question="What does attention use?",
        paper_id=paper_id,
        limit=2,
    )

    call = llm_service.generate.call_args

    user_prompt = call.kwargs["user_prompt"]

    assert "What does attention use?" in user_prompt
    assert "Attention uses queries, keys, and values." in user_prompt
    assert "Multi-head attention uses multiple attention heads." in user_prompt
    assert "[Chunk 5]" in user_prompt
    assert "[Chunk 6]" in user_prompt

def test_query_paper_handles_no_retrieved_chunks():
    service, embedding_service, vector_store, llm_service = (
        make_service()
    )

    paper_id = uuid4()

    embedding_service.embed.return_value = [
        [0.1, 0.2, 0.3]
    ]

    vector_store.search.return_value = []

    llm_service.generate.return_value = (
        "The provided context does not contain enough information."
    )

    result = service.query_paper(
        question="What is something unrelated?",
        paper_id=paper_id,
        limit=10,
    )

    assert result.query == "What is something unrelated?"
    assert result.chunks == []
    assert (
        result.answer
        == "The provided context does not contain enough information."
    )

    llm_service.generate.assert_called_once()