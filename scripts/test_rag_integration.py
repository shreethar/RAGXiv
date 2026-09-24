import httpx
import pytest
from openai import OpenAI
from sqlalchemy import select

from app.db.database import SessionLocal
from app.db.models.paper import Paper
from app.infrastructure.arxiv import ArxivService
from app.infrastructure.text_chunker import TextChunker
from app.infrastructure.pdf_document_processor import PdfDocumentProcessor
from app.infrastructure.embeddings import OpenAIEmbeddingService
from app.infrastructure.llm import OpenAILLMService
from app.infrastructure.pg_vector_store import PgVectorStore
from app.repositories.paper import PaperRepository
from app.services.paper_service import PaperService
from app.services.rag_service import RAGService


ARXIV_ID = "1706.03762"


@pytest.mark.integration
def test_real_rag_pipeline():
    session = SessionLocal()

    try:
        arxiv_service = ArxivService(
            client=httpx.Client()
        )

        embedding_service = OpenAIEmbeddingService(
            client=OpenAI()
        )

        llm_service = OpenAILLMService(
            client=OpenAI()
        )

        paper_repository = PaperRepository(session)

        vector_store = PgVectorStore(
            session=session
        )

        paper_service = PaperService(
            paper_repository=paper_repository,
            arxiv_service=arxiv_service,
            document_processor=PdfDocumentProcessor(),
            chunker=TextChunker(
                chunk_size=4000,
                overlap=400,
            ),
            embedding_service=embedding_service,
            vector_store=vector_store,
        )

        rag_service = RAGService(
            embedding_service=embedding_service,
            vector_store=vector_store,
            llm_service=llm_service,
        )

        # Ingest the paper.
        paper = paper_service.ingest(
            arxiv_id=ARXIV_ID
        )

        session.commit()

        # Ask a question about the paper.
        result = rag_service.query_paper(
            question=(
                "What is the main idea behind multi-head attention?"
            ),
            paper_id=paper.paper_id,
            limit=5,
        )

        print("\n=== RAG RESULT ===")
        print(f"query: {result.query}")
        print(f"answer: {result.answer}")
        print(f"retrieved chunks: {len(result.chunks)}")

        for chunk in result.chunks:
            print(
                f"\nchunk_index={chunk.chunk_index}"
                f" score={chunk.score}"
            )
            print(chunk.text[:300])

        # Basic assertions.
        assert result.query == (
            "What is the main idea behind multi-head attention?"
        )

        assert result.answer.strip()

        assert 0 < len(result.chunks) <= 5

        for chunk in result.chunks:
            assert chunk.text.strip()
            assert isinstance(chunk.score, float)

    finally:
        # Clean up the test paper.
        paper = session.scalar(
            select(Paper).where(
                Paper.arxiv_id == ARXIV_ID
            )
        )

        if paper:
            session.delete(paper)

        session.commit()
        session.close()