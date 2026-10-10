from datetime import datetime, timezone
from hashlib import sha256
from types import SimpleNamespace
from unittest.mock import Mock, call
from uuid import uuid4

import pytest

from app.infrastructure.arxiv import ArxivPaperMetadata
from app.infrastructure.vector_store import DocumentChunk
from app.services.paper_service import PaperService


def make_service():
    paper_repository = Mock()
    paper_repository.get_by_arxiv_id.return_value = None
    paper_repository.get_by_content_hash.return_value = None

    service = PaperService(
        paper_repository=paper_repository,
        arxiv_service=Mock(),
        document_processor=Mock(),
        chunker=Mock(),
        embedding_service=Mock(),
        vector_store=Mock(),
    )
    return service


def make_paper(*, source_type="arxiv", status="pending"):
    return SimpleNamespace(
        paper_id=uuid4(),
        source_type=source_type,
        status=status,
        ingestion_error=None,
    )


def configure_successful_processing(service, paper):
    chunks = [
        DocumentChunk(text="First chunk", chunk_index=0),
        DocumentChunk(text="Second chunk", chunk_index=1),
    ]
    embeddings = [[0.1, 0.2], [0.3, 0.4]]
    service.paper_repository.create.return_value = paper
    service.document_processor.extract_text.return_value = "extracted text"
    service.chunker.chunk.return_value = chunks
    service.embedding_service.embed.return_value = embeddings
    return chunks, embeddings


def test_arxiv_ingest_returns_existing_paper_without_processing():
    service = make_service()
    existing_paper = make_paper(status="ready")
    service.paper_repository.get_by_arxiv_id.return_value = existing_paper

    result = service.ingest(arxiv_id=" 2401.12345 ")

    assert result is existing_paper
    service.paper_repository.get_by_arxiv_id.assert_called_once_with(
        "2401.12345"
    )
    service.arxiv_service.get_metadata.assert_not_called()
    service.paper_repository.create.assert_not_called()


def test_arxiv_ingest_tracks_states_and_persists_processed_paper():
    service = make_service()
    paper = make_paper()
    chunks, embeddings = configure_successful_processing(service, paper)
    metadata = ArxivPaperMetadata(
        arxiv_id="2401.12345",
        title="Test Paper",
        abstract="Test abstract",
        authors=["Alice", "Bob"],
        version="v1",
        published_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        pdf_url="https://arxiv.org/pdf/2401.12345.pdf",
    )
    service.arxiv_service.get_metadata.return_value = metadata
    service.arxiv_service.download_pdf.return_value = b"pdf"

    result = service.ingest(arxiv_id="2401.12345")

    assert result is paper
    service.paper_repository.create.assert_called_once_with(
        source_type="arxiv",
        arxiv_id="2401.12345",
        title="2401.12345",
        status="pending",
    )
    service.paper_repository.update_metadata.assert_called_once_with(
        paper,
        title="Test Paper",
        abstract="Test abstract",
        authors="Alice, Bob",
        version="v1",
        published_at=metadata.published_at,
        pdf_url=metadata.pdf_url,
    )
    assert service.paper_repository.set_ingestion_status.call_args_list == [
        call(paper, status="processing"),
        call(paper, status="ready"),
    ]
    service.vector_store.add_chunks.assert_called_once_with(
        paper_id=paper.paper_id,
        chunks=chunks,
        embeddings=embeddings,
    )


@pytest.mark.parametrize(
    "failure, expected_stage",
    [
        (ValueError("metadata unavailable"), "metadata_lookup"),
        (None, "document_processing"),
    ],
)
def test_arxiv_ingest_persists_failed_state(failure, expected_stage):
    service = make_service()
    paper = make_paper()
    service.paper_repository.create.return_value = paper

    if failure is not None:
        service.arxiv_service.get_metadata.side_effect = failure
    else:
        metadata = ArxivPaperMetadata(
            arxiv_id="2401.12345",
            title="Test Paper",
            abstract="Test abstract",
            authors=["Alice"],
            version="v1",
            published_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            pdf_url="https://arxiv.org/pdf/2401.12345.pdf",
        )
        service.arxiv_service.get_metadata.return_value = metadata
        service.arxiv_service.download_pdf.return_value = b"pdf"
        service.document_processor.extract_text.return_value = ""
        service.chunker.chunk.return_value = []

    result = service.ingest(arxiv_id="2401.12345")

    assert result is paper
    failed_call = service.paper_repository.set_ingestion_status.call_args_list[-1]
    assert failed_call.args == (paper,)
    assert failed_call.kwargs["status"] == "failed"
    assert expected_stage in failed_call.kwargs["ingestion_error"]
    service.vector_store.add_chunks.assert_not_called()


def test_direct_pdf_upload_processes_and_deduplicates_by_content_hash():
    service = make_service()
    document = b"%PDF-1.7 test document"
    paper = make_paper(source_type="upload")
    chunks, embeddings = configure_successful_processing(service, paper)

    result = service.ingest_pdf(
        document=document,
        filename="../papers/example.pdf",
        title="  Uploaded Paper  ",
        abstract="  Abstract  ",
        authors="  Alice, Bob  ",
    )

    assert result is paper
    content_hash = sha256(document).hexdigest()
    service.paper_repository.get_by_content_hash.assert_called_once_with(
        content_hash
    )
    service.paper_repository.create.assert_called_once_with(
        source_type="upload",
        arxiv_id=None,
        content_hash=content_hash,
        original_filename="example.pdf",
        title="Uploaded Paper",
        abstract="Abstract",
        authors="Alice, Bob",
        status="pending",
    )
    service.vector_store.add_chunks.assert_called_once_with(
        paper_id=paper.paper_id,
        chunks=chunks,
        embeddings=embeddings,
    )

    duplicate_service = make_service()
    duplicate_service.paper_repository.get_by_content_hash.return_value = paper
    assert duplicate_service.ingest_pdf(
        document=document,
        filename="renamed.pdf",
    ) is paper
    duplicate_service.paper_repository.create.assert_not_called()
    duplicate_service.document_processor.extract_text.assert_not_called()


def test_direct_pdf_upload_uses_filename_as_default_title():
    service = make_service()
    paper = make_paper(source_type="upload")
    configure_successful_processing(service, paper)

    service.ingest_pdf(
        document=b"%PDF-1.7 document",
        filename="research-paper.pdf",
    )

    assert service.paper_repository.create.call_args.kwargs["title"] == (
        "research-paper"
    )


def test_failed_direct_upload_is_retried_using_same_paper():
    service = make_service()
    failed_paper = make_paper(source_type="upload", status="failed")
    service.paper_repository.get_by_content_hash.return_value = failed_paper
    service.document_processor.extract_text.return_value = "text"
    service.chunker.chunk.return_value = [
        DocumentChunk(text="chunk", chunk_index=0)
    ]
    service.embedding_service.embed.return_value = [[0.1]]

    result = service.ingest_pdf(
        document=b"%PDF-1.7 document",
        filename="paper.pdf",
    )

    assert result is failed_paper
    service.paper_repository.create.assert_not_called()
    service.vector_store.delete_paper.assert_called_once_with(
        paper_id=failed_paper.paper_id
    )
    assert service.paper_repository.set_ingestion_status.call_args_list == [
        call(failed_paper, status="processing"),
        call(failed_paper, status="ready"),
    ]


@pytest.mark.parametrize(
    "document, filename, message",
    [
        (b"", "paper.pdf", "must not be empty"),
        (b"not a pdf", "paper.pdf", "not a valid PDF"),
        (b"%PDF-1.7", "", "must have a filename"),
    ],
)
def test_direct_pdf_upload_validates_document(document, filename, message):
    service = make_service()

    with pytest.raises(ValueError, match=message):
        service.ingest_pdf(document=document, filename=filename)

    service.paper_repository.create.assert_not_called()


def test_embedding_count_mismatch_is_persisted_as_failed():
    service = make_service()
    paper = make_paper(source_type="upload")
    service.paper_repository.create.return_value = paper
    service.document_processor.extract_text.return_value = "text"
    service.chunker.chunk.return_value = [
        DocumentChunk(text="one", chunk_index=0),
        DocumentChunk(text="two", chunk_index=1),
    ]
    service.embedding_service.embed.return_value = [[0.1]]

    result = service.ingest_pdf(
        document=b"%PDF-1.7 document",
        filename="paper.pdf",
    )

    assert result is paper
    failed_call = service.paper_repository.set_ingestion_status.call_args_list[-1]
    assert failed_call.kwargs["status"] == "failed"
    assert "RuntimeError" in failed_call.kwargs["ingestion_error"]
    service.vector_store.add_chunks.assert_not_called()
