from unittest.mock import Mock
from hashlib import sha256

import pymupdf as fitz
import pytest
from sqlalchemy import func, select

from app.db.database import SessionLocal
from app.db.models.paper import Paper
from app.db.models.paper_chunk import PaperChunk
from app.infrastructure.embeddings import EMBEDDING_DIMENSIONS
from app.infrastructure.pdf_document_processor import PdfDocumentProcessor
from app.infrastructure.pg_vector_store import PgVectorStore
from app.infrastructure.text_chunker import TextChunker
from app.repositories.paper import PaperRepository
from app.services.paper_service import PaperService


class FakeEmbeddingService:
    def embed(self, *, texts: list[str]) -> list[list[float]]:
        return [[0.01] * EMBEDDING_DIMENSIONS for _ in texts]


class FailingEmbeddingService:
    def embed(self, *, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("provider details must remain private")


def make_pdf() -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "Direct PDF upload integration test with retrievable research text.",
    )
    try:
        return document.tobytes()
    finally:
        document.close()


def make_service(session, embedding_service):
    return PaperService(
        paper_repository=PaperRepository(session),
        arxiv_service=Mock(),
        document_processor=PdfDocumentProcessor(),
        chunker=TextChunker(chunk_size=100, overlap=20),
        embedding_service=embedding_service,
        vector_store=PgVectorStore(session=session),
    )


@pytest.mark.integration
def test_direct_pdf_upload_persists_ready_paper_chunks_and_deduplicates():
    session = SessionLocal()
    pdf = make_pdf()
    content_hash = sha256(pdf).hexdigest()

    try:
        service = make_service(session, FakeEmbeddingService())
        paper = service.ingest_pdf(
            document=pdf,
            filename="uploaded-paper.pdf",
            title="Uploaded Paper",
            authors="Alice, Bob",
        )
        session.commit()

        assert paper.source_type == "upload"
        assert paper.arxiv_id is None
        assert paper.original_filename == "uploaded-paper.pdf"
        assert paper.status == "ready"
        assert paper.ingestion_error is None

        chunks = list(
            session.scalars(
                select(PaperChunk)
                .where(PaperChunk.paper_id == paper.paper_id)
                .order_by(PaperChunk.chunk_index)
            )
        )
        assert chunks
        assert [chunk.chunk_index for chunk in chunks] == list(
            range(len(chunks))
        )

        duplicate = service.ingest_pdf(
            document=pdf,
            filename="renamed.pdf",
        )
        assert duplicate.paper_id == paper.paper_id
        assert session.scalar(
            select(func.count())
            .select_from(Paper)
            .where(Paper.content_hash == content_hash)
        ) == 1
    finally:
        session.rollback()
        session.query(Paper).filter(
            Paper.content_hash == content_hash
        ).delete()
        session.commit()
        session.close()


@pytest.mark.integration
def test_direct_pdf_upload_persists_sanitized_failed_state():
    session = SessionLocal()
    pdf = make_pdf()
    content_hash = sha256(pdf).hexdigest()

    try:
        service = make_service(session, FailingEmbeddingService())
        paper = service.ingest_pdf(
            document=pdf,
            filename="failed-upload.pdf",
        )
        session.commit()
        session.refresh(paper)

        assert paper.status == "failed"
        assert paper.ingestion_error == (
            "Ingestion failed during document_processing (RuntimeError)"
        )
        assert "provider details" not in paper.ingestion_error
        assert session.scalar(
            select(func.count())
            .select_from(PaperChunk)
            .where(PaperChunk.paper_id == paper.paper_id)
        ) == 0
    finally:
        session.rollback()
        session.query(Paper).filter(
            Paper.content_hash == content_hash
        ).delete()
        session.commit()
        session.close()
