from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.db.models.paper_chunk import PaperChunk
from app.infrastructure.vector_store import (
    DocumentChunk,
    RetrievedChunk,
    VectorStore,
)


class PgVectorStore(VectorStore):
    """PostgreSQL/pgvector-backed chunk store.

    RetrievedChunk.score is cosine distance from pgvector. Lower scores are
    more similar.
    """

    def __init__(self, session: Session):
        self.session = session

    def add_chunks(
        self,
        *,
        paper_id: UUID,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                "chunks and embeddings must have the same length"
            )

        paper_chunks = [
            PaperChunk(
                paper_id=paper_id,
                chunk_index=chunk.chunk_index,
                content=chunk.text,
                embedding=embedding,
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

        self.session.add_all(paper_chunks)
        self.session.flush()

    def search(
        self,
        *,
        paper_id: UUID,
        query_embedding: list[float],
        limit: int,
    ) -> list[RetrievedChunk]:
        if limit <= 0:
            raise ValueError("limit must be greater than 0")

        distance = PaperChunk.embedding.cosine_distance(
            query_embedding
        ).label("score")

        stmt = (
            select(
                PaperChunk.chunk_id,
                PaperChunk.content,
                PaperChunk.chunk_index,
                distance,
            )
            .where(PaperChunk.paper_id == paper_id)
            .order_by(distance.asc())
            .limit(limit)
        )

        rows = self.session.execute(stmt).all()

        return [
            RetrievedChunk(
                chunk_id=str(row.chunk_id),
                text=row.content,
                score=float(row.score),
                chunk_index=row.chunk_index,
            )
            for row in rows
        ]

    def delete_paper(
        self,
        *,
        paper_id: UUID,
    ) -> None:
        stmt = delete(PaperChunk).where(
            PaperChunk.paper_id == paper_id
        )
        self.session.execute(stmt)
        self.session.flush()
