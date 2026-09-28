from dataclasses import dataclass
from uuid import UUID

from app.agents.research_agent import ResearchAgent, ResearchMessage
from app.db.models.chat import Chat
from app.db.models.message import Message
from app.db.models.message_citation import MessageCitation
from app.db.models.project import Project
from app.repositories.chat import ChatRepository
from app.repositories.citation import CitationRepository
from app.repositories.message import MessageRepository
from app.repositories.project import ProjectRepository


@dataclass(frozen=True)
class SendMessageResult:
    chat_id: UUID
    user_message_id: UUID
    assistant_message_id: UUID
    answer: str
    citations: tuple[MessageCitation, ...] = ()


class ChatService:
    """Coordinates persistent chat history with the transient research agent."""

    def __init__(
        self,
        *,
        chat_repository: ChatRepository,
        message_repository: MessageRepository,
        citation_repository: CitationRepository,
        project_repository: ProjectRepository,
        research_agent: ResearchAgent,
    ):
        self.chat_repository = chat_repository
        self.message_repository = message_repository
        self.citation_repository = citation_repository
        self.project_repository = project_repository
        self.research_agent = research_agent

    def create_chat(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
        title: str | None = None,
    ) -> Chat:
        self._get_owned_project(
            user_id=user_id,
            project_id=project_id,
        )

        normalized_title = title.strip() if title is not None else None

        if title is not None and not normalized_title:
            raise ValueError("Chat title must not be empty")

        return self.chat_repository.create(
            project_id=project_id,
            title=normalized_title,
        )

    def get_chat(
        self,
        *,
        user_id: UUID,
        chat_id: UUID,
    ) -> Chat:
        return self._get_owned_chat(
            user_id=user_id,
            chat_id=chat_id,
        )

    def list_messages(
        self,
        *,
        user_id: UUID,
        chat_id: UUID,
    ) -> list[Message]:
        self._get_owned_chat(
            user_id=user_id,
            chat_id=chat_id,
        )
        return self.message_repository.list_by_chat(chat_id)

    def get_citation(
        self,
        *,
        user_id: UUID,
        citation_id: UUID,
    ) -> MessageCitation:
        citation = self.citation_repository.get_by_id(citation_id)

        if citation is None:
            raise ValueError("Citation not found")

        message = self.message_repository.get_by_id(
            citation.message_id
        )

        if message is None:
            raise ValueError("Citation message not found")

        self._get_owned_chat(
            user_id=user_id,
            chat_id=message.chat_id,
        )
        return citation

    def send_message(
        self,
        *,
        user_id: UUID,
        chat_id: UUID,
        content: str,
    ) -> SendMessageResult:
        normalized_content = content.strip()

        if not normalized_content:
            raise ValueError("Message content must not be empty")

        self._get_owned_chat(
            user_id=user_id,
            chat_id=chat_id,
        )

        user_message = self.message_repository.create(
            chat_id=chat_id,
            role="user",
            content=normalized_content,
        )

        persisted_history = self.message_repository.list_by_chat(
            chat_id
        )
        agent_messages = [
            self._to_agent_message(message)
            for message in persisted_history
        ]

        agent_result = self.research_agent.invoke(
            messages=agent_messages,
        )

        assistant_message = self.message_repository.create(
            chat_id=chat_id,
            role="assistant",
            content=agent_result.answer,
        )

        citations = self.citation_repository.create_many(
            message_id=assistant_message.message_id,
            references=agent_result.citations,
        ) if agent_result.citations else []

        return SendMessageResult(
            chat_id=chat_id,
            user_message_id=user_message.message_id,
            assistant_message_id=assistant_message.message_id,
            answer=agent_result.answer,
            citations=tuple(citations),
        )

    def _get_owned_chat(
        self,
        *,
        user_id: UUID,
        chat_id: UUID,
    ) -> Chat:
        chat = self.chat_repository.get_by_id(chat_id)

        if chat is None:
            raise ValueError("Chat not found")

        self._get_owned_project(
            user_id=user_id,
            project_id=chat.project_id,
        )

        return chat

    def _get_owned_project(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
    ) -> Project:
        project = self.project_repository.get_by_id(project_id)

        if project is None:
            raise ValueError("Chat project not found")

        if project.user_id != user_id:
            raise ValueError("Chat does not belong to user")

        return project

    @staticmethod
    def _to_agent_message(message: Message) -> ResearchMessage:
        if message.role in {"user", "assistant"}:
            return ResearchMessage(
                role=message.role,
                content=message.content,
            )

        raise ValueError(
            f"Unsupported persisted message role: {message.role}"
        )
