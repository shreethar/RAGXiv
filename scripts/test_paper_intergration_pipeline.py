import os

import pytest
from dotenv import load_dotenv
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.db.models.paper_chunk import PaperChunk
from app.infrastructure.arxiv import ArxivService
from app.infrastructure.embeddings import (
    OpenAIEmbeddingService,
    EMBEDDING_DIMENSIONS,
)
from app.infrastructure.pdf_document_processor import (
    PdfDocumentProcessor,
)
from app.infrastructure.text_chunker import TextChunker
from app.infrastructure.vector_store import DocumentChunk
from app.infrastructure.pg_vector_store import PgVectorStore
from app.repositories.paper import PaperRepository
from app.services.paper_service import PaperService
from openai import OpenAI
import httpx


load_dotenv()


@pytest.mark.integration
def test_real_paper_ingestion():
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not configured")

    arxiv_id = "1706.03762"

    session: Session = SessionLocal()

    try:
        # Clean up from a previous test run.
        existing_paper = (
            PaperRepository(session)
            .get_by_arxiv_id(arxiv_id)
        )

        if existing_paper is not None:
            session.delete(existing_paper)
            session.commit()

        # Infrastructure dependencies.
        http_client = httpx.Client(timeout=60.0)
        openai_client = OpenAI()

        arxiv_service = ArxivService(
            client=http_client,
        )

        document_processor = PdfDocumentProcessor()

        chunker = TextChunker(
            chunk_size=4000,
            overlap=400,
        )

        embedding_service = OpenAIEmbeddingService(
            client=openai_client,
        )

        vector_store = PgVectorStore(
            session=session,
        )

        paper_repository = PaperRepository(
            session,
        )

        paper_service = PaperService(
            paper_repository=paper_repository,
            arxiv_service=arxiv_service,
            document_processor=document_processor,
            chunker=chunker,
            embedding_service=embedding_service,
            vector_store=vector_store,
        )

        # Run the real ingestion.
        paper = paper_service.ingest(
            arxiv_id=arxiv_id,
        )

        # Commit the complete ingestion transaction.
        session.commit()

        # Basic Paper assertions.
        assert paper.arxiv_id == arxiv_id
        assert paper.status == "ready"
        assert paper.title
        assert paper.abstract
        assert paper.authors
        assert paper.version
        assert paper.pdf_url

        # Verify PaperChunk rows.
        chunks = list(
            session.scalars(
                select(PaperChunk)
                .where(
                    PaperChunk.paper_id == paper.paper_id
                )
                .order_by(PaperChunk.chunk_index)
            )
        )

        assert chunks
        assert all(chunk.content for chunk in chunks)
        assert all(
            len(chunk.embedding) == EMBEDDING_DIMENSIONS
            for chunk in chunks
        )

        # Verify chunk indexes.
        assert [
            chunk.chunk_index
            for chunk in chunks
        ] == list(range(len(chunks)))

    except Exception:
        session.rollback()
        raise

    finally:
        # Remove the test paper and its chunks.
        existing_paper = (
            PaperRepository(session)
            .get_by_arxiv_id(arxiv_id)
        )

        if existing_paper is not None:
            session.delete(existing_paper)
            session.commit()

        session.close()