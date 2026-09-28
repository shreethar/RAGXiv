from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest

from app.infrastructure.vector_store import RetrievedChunk
from app.services.rag_service import (
    RAGService,
    RagAnswerOutput,
    RagCitationSelection,
)


def make_service():
    embedding_service = Mock()
    vector_store = Mock()
    llm_service = Mock()
    paper_repository = Mock()
    paper_repository.get_by_id.return_value = SimpleNamespace(
        title="Attention Is All You Need",
        arxiv_id="1706.03762",
    )
    service = RAGService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        llm_service=llm_service,
        paper_repository=paper_repository,
    )
    return (
        service,
        embedding_service,
        vector_store,
        llm_service,
        paper_repository,
    )


def _chunk(*, index: int, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid4(),
        text=text,
        score=0.1,
        chunk_index=index,
    )


def test_query_paper_assigns_sources_and_maps_selected_citations():
    service, embeddings, vector_store, llm, _ = make_service()
    paper_id = uuid4()
    chunks = [
        _chunk(index=3, text="Self-attention relates tokens."),
        _chunk(index=4, text="Transformers use multiple heads."),
    ]
    embeddings.embed.return_value = [[0.1, 0.2]]
    vector_store.search.return_value = chunks
    llm.generate_structured.return_value = RagAnswerOutput(
        answer="Self-attention relates tokens.",
        citations=[RagCitationSelection(source_id="source_1")],
    )

    result = service.query_paper(
        question="How does self-attention work?",
        paper_id=paper_id,
        limit=2,
    )

    assert result.chunks == chunks
    assert result.answer == "Self-attention relates tokens."
    assert len(result.citations) == 1
    assert result.citations[0].source_id == "source_1"
    assert result.citations[0].reference.paper_id == paper_id
    assert result.citations[0].reference.chunk_id == chunks[0].chunk_id
    embeddings.embed.assert_called_once_with(
        texts=["How does self-attention work?"]
    )
    vector_store.search.assert_called_once_with(
        paper_id=paper_id,
        query_embedding=[0.1, 0.2],
        limit=2,
    )


def test_source_ids_and_paper_metadata_are_in_structured_llm_prompt():
    service, embeddings, vector_store, llm, _ = make_service()
    embeddings.embed.return_value = [[0.1]]
    vector_store.search.return_value = [
        _chunk(index=5, text="Attention uses queries, keys, and values."),
        _chunk(index=8, text="Multiple heads learn different relations."),
    ]
    llm.generate_structured.return_value = RagAnswerOutput(
        answer="Answer",
        citations=[],
    )

    service.query_paper(
        question="What does attention use?",
        paper_id=uuid4(),
    )

    call = llm.generate_structured.call_args
    prompt = call.kwargs["user_prompt"]
    assert "[Source ID: source_1]" in prompt
    assert "[Source ID: source_2]" in prompt
    assert "Paper: Attention Is All You Need" in prompt
    assert "arXiv ID: 1706.03762" in prompt
    assert "Chunk index: 5" in prompt
    assert call.kwargs["output_type"] is RagAnswerOutput


def test_zero_citations_is_valid_even_when_chunks_were_retrieved():
    service, embeddings, vector_store, llm, _ = make_service()
    embeddings.embed.return_value = [[0.1]]
    vector_store.search.return_value = [
        _chunk(index=1, text="Unrelated evidence")
    ]
    llm.generate_structured.return_value = RagAnswerOutput(
        answer="The sources do not support an answer.",
        citations=[],
    )

    result = service.query_paper(
        question="Unsupported question",
        paper_id=uuid4(),
    )

    assert result.chunks
    assert result.citations == []


def test_multiple_selected_sources_preserve_model_order():
    service, embeddings, vector_store, llm, _ = make_service()
    paper_id = uuid4()
    chunks = [
        _chunk(index=1, text="First"),
        _chunk(index=2, text="Second"),
        _chunk(index=3, text="Third"),
    ]
    embeddings.embed.return_value = [[0.1]]
    vector_store.search.return_value = chunks
    llm.generate_structured.return_value = RagAnswerOutput(
        answer="Combined answer",
        citations=[
            RagCitationSelection(source_id="source_3"),
            RagCitationSelection(source_id="source_1"),
        ],
    )

    result = service.query_paper(
        question="Combine evidence",
        paper_id=paper_id,
    )

    assert [item.source_id for item in result.citations] == [
        "source_3",
        "source_1",
    ]
    assert [item.reference.chunk_id for item in result.citations] == [
        chunks[2].chunk_id,
        chunks[0].chunk_id,
    ]


def test_invalid_model_source_id_is_rejected():
    service, embeddings, vector_store, llm, _ = make_service()
    embeddings.embed.return_value = [[0.1]]
    vector_store.search.return_value = [
        _chunk(index=1, text="Only valid source")
    ]
    llm.generate_structured.return_value = RagAnswerOutput(
        answer="Answer",
        citations=[RagCitationSelection(source_id="source_99")],
    )

    with pytest.raises(
        ValueError,
        match="unknown citation source: source_99",
    ):
        service.query_paper(question="Question", paper_id=uuid4())


def test_missing_paper_is_rejected_before_retrieval():
    service, embeddings, vector_store, _, paper_repository = make_service()
    paper_repository.get_by_id.return_value = None

    with pytest.raises(ValueError, match="Paper not found"):
        service.query_paper(question="Question", paper_id=uuid4())

    embeddings.embed.assert_not_called()
    vector_store.search.assert_not_called()
