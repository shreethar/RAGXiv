from sqlalchemy import func, select

from app.db.database import SessionLocal
from app.db.models.paper import Paper
from app.db.models.paper_chunk import PaperChunk
from app.infrastructure.arxiv import ArxivService
from app.infrastructure.text_chunker import TextChunker
from app.infrastructure.pdf_document_processor import PdfDocumentProcessor
from app.infrastructure.embeddings import OpenAIEmbeddingService
from app.infrastructure.pg_vector_store import PgVectorStore
from app.repositories.paper import PaperRepository
from app.services.paper_service import PaperService
from openai import OpenAI
import httpx


ARXIV_ID = "1706.03762"


def main() -> None:
    session = SessionLocal()

    try:
        arxiv_service = ArxivService(
            client=httpx.Client()
        )

        embedding_service = OpenAIEmbeddingService(
            client=OpenAI()
        )

        paper_repository = PaperRepository(session)

        vector_store = PgVectorStore(
            session=session
        )

        service = PaperService(
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

        paper = service.ingest(
            arxiv_id=ARXIV_ID
        )

        session.commit()

        print("\n=== PAPER ===")
        print(f"paper_id:        {paper.paper_id}")
        print(f"arxiv_id:        {paper.arxiv_id}")
        print(f"title:           {paper.title}")
        print(f"version:         {paper.version}")
        print(f"status:          {paper.status}")
        print(f"project_count:   {paper.project_count}")
        print(f"published_at:    {paper.published_at}")

        chunk_count = session.scalar(
            select(func.count())
            .select_from(PaperChunk)
            .where(PaperChunk.paper_id == paper.paper_id)
        )

        print("\n=== CHUNKS ===")
        print(f"chunk_count:     {chunk_count}")

        first_chunk = session.scalar(
            select(PaperChunk)
            .where(PaperChunk.paper_id == paper.paper_id)
            .order_by(PaperChunk.chunk_index.asc())
            .limit(1)
        )

        if first_chunk:
            print(f"first_chunk_id:  {first_chunk.chunk_id}")
            print(f"chunk_index:     {first_chunk.chunk_index}")
            print(f"text_length:     {len(first_chunk.content)}")
            print(f"embedding_dim:   {len(first_chunk.embedding)}")

        print("\n=== VERIFICATION ===")
        print(f"status ready:    {paper.status == 'ready'}")
        print(f"has chunks:      {chunk_count > 0}")
        print(
            f"embedding 3072:  "
            f"{first_chunk is not None and len(first_chunk.embedding) == 3072}"
        )
        print(f"project_count 0: {paper.project_count == 0}")

    finally:
        # Clean up the test paper and its chunks.
        paper = session.scalar(
            select(Paper)
            .where(Paper.arxiv_id == ARXIV_ID)
        )

        if paper:
            session.execute(
                PaperChunk.__table__.delete().where(
                    PaperChunk.paper_id == paper.paper_id
                )
            )

            session.delete(paper)

        session.commit()
        session.close()


if __name__ == "__main__":
    main()