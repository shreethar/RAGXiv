from uuid import uuid4

import pytest
from sqlalchemy import select

from app.db.database import SessionLocal
from app.db.models.chat import Chat
from app.db.models.message import Message
from app.db.models.message_citation import MessageCitation
from app.db.models.paper import Paper
from app.db.models.paper_chunk import PaperChunk
from app.db.models.project import Project
from app.db.models.user import User
from app.domain.citations import CitationReference
from app.repositories.citation import CitationRepository


@pytest.mark.integration
def test_citation_relationship_validation_order_and_delete_cascade():
    session = SessionLocal()

    try:
        user = User(
            name="Citation Test User",
            email=f"citation-{uuid4()}@example.com",
            password_hash="test",
        )
        session.add(user)
        session.flush()
        project = Project(user_id=user.user_id, name="Citation Project")
        session.add(project)
        session.flush()
        chat = Chat(project_id=project.project_id, title="Citation Chat")
        session.add(chat)
        session.flush()
        assistant = Message(
            chat_id=chat.chat_id,
            role="assistant",
            content="An answer with sources.",
        )
        user_message = Message(
            chat_id=chat.chat_id,
            role="user",
            content="A question.",
        )
        session.add_all([assistant, user_message])
        session.flush()
        paper_a = Paper(
            arxiv_id=f"test-{uuid4()}",
            title="Paper A",
            status="ready",
        )
        paper_b = Paper(
            arxiv_id=f"test-{uuid4()}",
            title="Paper B",
            status="ready",
        )
        session.add_all([paper_a, paper_b])
        session.flush()
        chunk_a = PaperChunk(
            paper_id=paper_a.paper_id,
            chunk_index=0,
            content="Evidence A",
            embedding=[0.0] * 3072,
        )
        chunk_b = PaperChunk(
            paper_id=paper_b.paper_id,
            chunk_index=0,
            content="Evidence B",
            embedding=[0.0] * 3072,
        )
        session.add_all([chunk_a, chunk_b])
        session.flush()
        repository = CitationRepository(session)

        citations = repository.create_many(
            message_id=assistant.message_id,
            references=[
                CitationReference(
                    paper_id=paper_b.paper_id,
                    chunk_id=chunk_b.chunk_id,
                ),
                CitationReference(
                    paper_id=paper_a.paper_id,
                    chunk_id=chunk_a.chunk_id,
                ),
            ],
        )

        assert [citation.position for citation in citations] == [1, 2]
        loaded = repository.get_by_id(citations[0].citation_id)
        assert loaded is not None
        assert loaded.chunk.paper.paper_id == paper_b.paper_id
        assert loaded.chunk.content == "Evidence B"

        with pytest.raises(ValueError, match="does not belong"):
            repository.create_many(
                message_id=assistant.message_id,
                references=[
                    CitationReference(
                        paper_id=paper_a.paper_id,
                        chunk_id=chunk_b.chunk_id,
                    )
                ],
            )

        with pytest.raises(ValueError, match="chunk not found"):
            repository.create_many(
                message_id=assistant.message_id,
                references=[
                    CitationReference(
                        paper_id=paper_a.paper_id,
                        chunk_id=uuid4(),
                    )
                ],
            )

        with pytest.raises(ValueError, match="assistant messages"):
            repository.create_many(
                message_id=user_message.message_id,
                references=[
                    CitationReference(
                        paper_id=paper_a.paper_id,
                        chunk_id=chunk_a.chunk_id,
                    )
                ],
            )

        citation_ids = [citation.citation_id for citation in citations]
        session.delete(assistant)
        session.flush()
        remaining = list(
            session.scalars(
                select(MessageCitation).where(
                    MessageCitation.citation_id.in_(citation_ids)
                )
            )
        )
        assert remaining == []
    finally:
        session.rollback()
        session.close()
