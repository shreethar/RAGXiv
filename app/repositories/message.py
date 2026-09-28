from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, selectinload

from app.db.models.message import Message
from app.db.models.message_citation import MessageCitation
from app.db.models.paper_chunk import PaperChunk


class MessageRepository:

    def __init__(self, session: Session):
        self.session = session

    def get_by_id(
        self,
        message_id: UUID,
    ) -> Message | None:
        stmt = (
            select(Message)
            .where(Message.message_id == message_id)
            .options(
                selectinload(Message.citations)
                .selectinload(MessageCitation.chunk)
                .selectinload(PaperChunk.paper)
            )
        )

        return self.session.scalar(stmt)

    def list_by_chat(
        self,
        chat_id: UUID,
    ) -> list[Message]:
        stmt = (
            select(Message)
            .where(Message.chat_id == chat_id)
            .order_by(Message.created_at.asc())
            .options(
                selectinload(Message.citations)
                .selectinload(MessageCitation.chunk)
                .selectinload(PaperChunk.paper)
            )
        )

        return list(self.session.scalars(stmt))

    def create(
        self,
        *,
        chat_id: UUID,
        role: str,
        content: str,
    ) -> Message:
        message = Message(
            chat_id=chat_id,
            role=role,
            content=content,
        )

        self.session.add(message)
        self.session.flush()

        return message

    def delete(
        self,
        message_id: UUID,
    ) -> None:
        message = self.get_by_id(message_id)

        if message is None:
            raise ValueError(
                f"Message {message_id} not found"
            )

        self.session.delete(message)
        self.session.flush()
