from datetime import datetime, timezone
from unittest.mock import Mock
from uuid import uuid4

from app.db.models.paper import Paper
from app.infrastructure.arxiv import ArxivPaperMetadata
from app.infrastructure.vector_store import DocumentChunk
from app.services.paper_service import PaperService
import pytest


def make_service():
    paper_repository = Mock()
    paper_repository.get_by_arxiv_id.return_value = None

    return PaperService(
        paper_repository=paper_repository,
        arxiv_service=Mock(),
        document_processor=Mock(),
        chunker=Mock(),
        embedding_service=Mock(),
        vector_store=Mock(),
    )


def test_ingest_returns_existing_paper():
    service = make_service()

    existing_paper = Mock(spec=Paper)

    service.paper_repository.get_by_arxiv_id.return_value = (
        existing_paper
    )

    result = service.ingest(
        arxiv_id="2401.12345"
    )

    assert result is existing_paper

    service.arxiv_service.get_metadata.assert_not_called()
    service.arxiv_service.download_pdf.assert_not_called()
    service.document_processor.extract_text.assert_not_called()
    service.chunker.chunk.assert_not_called()
    service.embedding_service.embed.assert_not_called()
    service.vector_store.add_chunks.assert_not_called()
    service.paper_repository.create.assert_not_called()


def test_ingest_processes_and_persists_new_paper():
    service = make_service()

    paper_id = uuid4()

    metadata = ArxivPaperMetadata(
        arxiv_id="2401.12345",
        title="Test Paper",
        abstract="Test abstract",
        authors=["Alice", "Bob"],
        version="v1",
        published_at=datetime(
            2024,
            1,
            1,
            tzinfo=timezone.utc,
        ),
        pdf_url="https://arxiv.org/pdf/2401.12345.pdf",
    )

    chunks = [
        DocumentChunk(
            text="First chunk",
            chunk_index=0,
        ),
        DocumentChunk(
            text="Second chunk",
            chunk_index=1,
        ),
    ]

    embeddings = [
        [0.1, 0.2],
        [0.3, 0.4],
    ]

    paper = Mock(spec=Paper)
    paper.paper_id = paper_id

    service.arxiv_service.get_metadata.return_value = metadata
    service.arxiv_service.download_pdf.return_value = b"pdf"
    service.document_processor.extract_text.return_value = (
        "extracted text"
    )
    service.chunker.chunk.return_value = chunks
    service.embedding_service.embed.return_value = embeddings
    service.paper_repository.create.return_value = paper

    result = service.ingest(
        arxiv_id="2401.12345"
    )

    assert result is paper

    service.arxiv_service.get_metadata.assert_called_once_with(
        arxiv_id="2401.12345"
    )

    service.arxiv_service.download_pdf.assert_called_once_with(
        arxiv_id="2401.12345"
    )

    service.document_processor.extract_text.assert_called_once_with(
        document=b"pdf"
    )

    service.chunker.chunk.assert_called_once_with(
        text="extracted text"
    )

    service.embedding_service.embed.assert_called_once_with(
        texts=[
            "First chunk",
            "Second chunk",
        ]
    )

    service.paper_repository.create.assert_called_once_with(
        arxiv_id="2401.12345",
        title="Test Paper",
        abstract="Test abstract",
        authors="Alice, Bob",
        version="v1",
        published_at=metadata.published_at,
        pdf_url="https://arxiv.org/pdf/2401.12345.pdf",
        status="ready",
    )

    service.vector_store.add_chunks.assert_called_once_with(
        paper_id=paper_id,
        chunks=chunks,
        embeddings=embeddings,
    )

def test_ingest_rejects_paper_with_no_chunks():
    service = make_service()

    metadata = ArxivPaperMetadata(
        arxiv_id="2401.12345",
        title="Test Paper",
        abstract="Test abstract",
        authors=["Alice"],
        version="v1",
        published_at=datetime(
            2024,
            1,
            1,
            tzinfo=timezone.utc,
        ),
        pdf_url="https://arxiv.org/pdf/2401.12345.pdf",
    )

    service.arxiv_service.get_metadata.return_value = metadata
    service.arxiv_service.download_pdf.return_value = b"pdf"
    service.document_processor.extract_text.return_value = (
        ""
    )
    service.chunker.chunk.return_value = []

    with pytest.raises(
        ValueError,
        match="No text could be extracted",
    ):
        service.ingest(
            arxiv_id="2401.12345"
        )

    service.embedding_service.embed.assert_not_called()
    service.paper_repository.create.assert_not_called()
    service.vector_store.add_chunks.assert_not_called()


def test_ingest_rejects_embedding_count_mismatch():
    service = make_service()

    metadata = ArxivPaperMetadata(
        arxiv_id="2401.12345",
        title="Test Paper",
        abstract="Test abstract",
        authors=["Alice"],
        version="v1",
        published_at=datetime(
            2024,
            1,
            1,
            tzinfo=timezone.utc,
        ),
        pdf_url="https://arxiv.org/pdf/2401.12345.pdf",
    )

    chunks = [
        DocumentChunk(
            text="First chunk",
            chunk_index=0,
        ),
        DocumentChunk(
            text="Second chunk",
            chunk_index=1,
        ),
    ]

    service.arxiv_service.get_metadata.return_value = metadata
    service.arxiv_service.download_pdf.return_value = b"pdf"
    service.document_processor.extract_text.return_value = (
        "text"
    )
    service.chunker.chunk.return_value = chunks

    service.embedding_service.embed.return_value = [
        [0.1, 0.2],
    ]

    with pytest.raises(
        RuntimeError,
        match="Embedding count",
    ):
        service.ingest(
            arxiv_id="2401.12345"
        )

    service.paper_repository.create.assert_not_called()
    service.vector_store.add_chunks.assert_not_called()