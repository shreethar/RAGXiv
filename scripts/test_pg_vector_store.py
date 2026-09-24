from uuid import uuid4

from sqlalchemy import select

from app.db.database import SessionLocal
from app.db.models.paper import Paper
from app.db.models.paper_chunk import PaperChunk
from app.infrastructure.pg_vector_store import PgVectorStore
from app.infrastructure.vector_store import DocumentChunk


EMBEDDING_DIMENSIONS = 3072


def embedding(
    *,
    first: float = 0.0,
    second: float = 0.0,
) -> list[float]:
    values = [0.0] * EMBEDDING_DIMENSIONS
    values[0] = first
    values[1] = second
    return values


def create_paper(
    session,
    *,
    title: str,
) -> Paper:
    paper = Paper(
        arxiv_id=f"vs-{uuid4()}",
        title=title,
    )
    session.add(paper)
    session.flush()
    return paper


def main():
    with SessionLocal() as session:
        try:
            vector_store = PgVectorStore(session)

            paper_a = create_paper(
                session,
                title="Vector Store Test Paper A",
            )
            paper_b = create_paper(
                session,
                title="Vector Store Test Paper B",
            )

            chunks_a = [
                DocumentChunk(
                    text="A exact match chunk",
                    chunk_index=0,
                ),
                DocumentChunk(
                    text="A nearby chunk",
                    chunk_index=1,
                ),
                DocumentChunk(
                    text="A far chunk",
                    chunk_index=2,
                ),
            ]
            embeddings_a = [
                embedding(first=1.0),
                embedding(first=0.9, second=0.1),
                embedding(second=1.0),
            ]

            vector_store.add_chunks(
                paper_id=paper_a.paper_id,
                chunks=chunks_a,
                embeddings=embeddings_a,
            )

            vector_store.add_chunks(
                paper_id=paper_b.paper_id,
                chunks=[
                    DocumentChunk(
                        text="B exact match chunk",
                        chunk_index=0,
                    ),
                ],
                embeddings=[
                    embedding(first=1.0),
                ],
            )

            stored_chunks = list(
                session.scalars(
                    select(PaperChunk).where(
                        PaperChunk.paper_id == paper_a.paper_id
                    )
                )
            )
            assert len(stored_chunks) == 3

            try:
                vector_store.add_chunks(
                    paper_id=paper_a.paper_id,
                    chunks=[
                        DocumentChunk(
                            text="Mismatch chunk 0",
                            chunk_index=10,
                        ),
                        DocumentChunk(
                            text="Mismatch chunk 1",
                            chunk_index=11,
                        ),
                        DocumentChunk(
                            text="Mismatch chunk 2",
                            chunk_index=12,
                        ),
                    ],
                    embeddings=[
                        embedding(first=1.0),
                        embedding(first=0.5, second=0.5),
                    ],
                )
                assert False, "Expected mismatched inputs to fail"
            except ValueError:
                print("mismatched add_chunks(): passed")

            results = vector_store.search(
                paper_id=paper_a.paper_id,
                query_embedding=embedding(first=1.0),
                limit=2,
            )

            assert len(results) == 2
            assert [result.chunk_index for result in results] == [0, 1]
            assert results[0].score <= results[1].score
            assert all("B " not in result.text for result in results)

            print("paper-scoped search(): passed")
            print("nearest ordering and limit: passed")

            try:
                vector_store.search(
                    paper_id=paper_a.paper_id,
                    query_embedding=embedding(first=1.0),
                    limit=0,
                )
                assert False, "Expected invalid limit to fail"
            except ValueError:
                print("invalid limit: passed")

            vector_store.delete_paper(
                paper_id=paper_a.paper_id,
            )

            remaining_a_chunks = list(
                session.scalars(
                    select(PaperChunk).where(
                        PaperChunk.paper_id == paper_a.paper_id
                    )
                )
            )
            remaining_b_chunks = list(
                session.scalars(
                    select(PaperChunk).where(
                        PaperChunk.paper_id == paper_b.paper_id
                    )
                )
            )
            remaining_paper_a = session.get(
                Paper,
                paper_a.paper_id,
            )

            assert len(remaining_a_chunks) == 0
            assert len(remaining_b_chunks) == 1
            assert remaining_paper_a is not None

            print("delete_paper(): passed")

            session.delete(paper_a)
            session.delete(paper_b)
            session.commit()

            print()
            print("All PgVectorStore tests passed.")

        except Exception:
            session.rollback()
            raise


if __name__ == "__main__":
    main()
