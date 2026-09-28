from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.db.models.message import Message
from app.db.models.message_citation import MessageCitation
from app.db.models.paper_chunk import PaperChunk
from app.domain.citations import CitationReference


class CitationRepository:
    def __init__(self, session: Session):
        self.session = session

    def get_by_id(
        self,
        citation_id: UUID,
    ) -> MessageCitation | None:
        stmt = (
            select(MessageCitation)
            .where(MessageCitation.citation_id == citation_id)
            .options(
                selectinload(MessageCitation.chunk).selectinload(
                    PaperChunk.paper
                )
            )
        )
        return self.session.scalar(stmt)

    def create_many(
        self,
        *,
        message_id: UUID,
        references: Sequence[CitationReference],
    ) -> list[MessageCitation]:
        if not references:
            return []

        message = self.session.get(Message, message_id)

        if message is None:
            raise ValueError("Message not found")

        if message.role != "assistant":
            raise ValueError(
                "Citations can only belong to assistant messages"
            )

        chunk_ids = [reference.chunk_id for reference in references]
        chunks = list(
            self.session.scalars(
                select(PaperChunk).where(
                    PaperChunk.chunk_id.in_(chunk_ids)
                )
            )
        )
        chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}

        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("Duplicate citation chunk")

        for reference in references:
            chunk = chunks_by_id.get(reference.chunk_id)

            if chunk is None:
                raise ValueError(
                    f"Citation chunk not found: {reference.chunk_id}"
                )

            if chunk.paper_id != reference.paper_id:
                raise ValueError(
                    "Citation chunk does not belong to the referenced paper"
                )

        citations = [
            MessageCitation(
                message_id=message_id,
                chunk_id=reference.chunk_id,
                position=position,
            )
            for position, reference in enumerate(references, start=1)
        ]
        self.session.add_all(citations)
        self.session.flush()
        return citations
