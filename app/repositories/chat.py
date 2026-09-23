from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models.chat import Chat


class ChatRepository:

    def __init__(self, session: Session):
        self.session = session

    def get_by_id(self, chat_id: UUID) -> Chat | None:
        stmt = select(Chat).where(
            Chat.chat_id == chat_id
        )

        return self.session.scalar(stmt)

    def list_by_project(
        self,
        project_id: UUID,
    ) -> list[Chat]:
        stmt = (
            select(Chat)
            .where(Chat.project_id == project_id)
            .order_by(Chat.created_at.asc())
        )

        return list(self.session.scalars(stmt))

    def create(
        self,
        *,
        project_id: UUID,
        title: str | None = None,
    ) -> Chat:
        chat = Chat(
            project_id=project_id,
            title=title,
        )

        self.session.add(chat)
        self.session.flush()

        return chat

    def update(
        self,
        chat_id: UUID,
        *,
        title: str | None,
    ) -> Chat:
        chat = self.get_by_id(chat_id)

        if chat is None:
            raise ValueError(
                f"Chat {chat_id} not found"
            )

        chat.title = title

        self.session.flush()

        return chat

    def delete(self, chat_id: UUID) -> None:
        chat = self.get_by_id(chat_id)

        if chat is None:
            raise ValueError(
                f"Chat {chat_id} not found"
            )

        self.session.delete(chat)
        self.session.flush()