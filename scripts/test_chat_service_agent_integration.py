from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Sequence
from unittest.mock import MagicMock
from uuid import UUID, uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from pydantic import Field, PrivateAttr

from app.agents.research_agent import create_research_agent
from app.agents.tools import AgentToolDependencies
from app.infrastructure.arxiv import ArxivSearchResult
from app.services.chat_service import ChatService


class ScriptedChatModel(BaseChatModel):
    responses: list[AIMessage] = Field(default_factory=list)
    _response_index: int = PrivateAttr(default=0)
    _seen_messages: list[list[BaseMessage]] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "chat-service-scripted-test-model"

    @property
    def seen_messages(self) -> list[list[BaseMessage]]:
        return self._seen_messages

    def bind_tools(
        self,
        tools: Sequence[BaseTool | dict[str, Any]],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> "ScriptedChatModel":
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._seen_messages.append(list(messages))
        response = self.responses[self._response_index]
        self._response_index += 1
        return ChatResult(
            generations=[ChatGeneration(message=response)]
        )


class InMemoryMessageRepository:
    def __init__(self):
        self.messages: list[SimpleNamespace] = []

    def create(
        self,
        *,
        chat_id: UUID,
        role: str,
        content: str,
    ) -> SimpleNamespace:
        message = SimpleNamespace(
            message_id=uuid4(),
            chat_id=chat_id,
            role=role,
            content=content,
        )
        self.messages.append(message)
        return message

    def list_by_chat(self, chat_id: UUID) -> list[SimpleNamespace]:
        return [
            message
            for message in self.messages
            if message.chat_id == chat_id
        ]


def test_chat_service_persists_only_conversation_around_agent_tool_run():
    user_id = uuid4()
    project_id = uuid4()
    chat_id = uuid4()
    arxiv_service = MagicMock()
    arxiv_service.search.return_value = [
        ArxivSearchResult(
            arxiv_id="1706.03762",
            title="Attention Is All You Need",
            abstract="A sequence transduction architecture.",
            authors=["Ashish Vaswani"],
            published_at=datetime(2017, 6, 12, tzinfo=timezone.utc),
            pdf_url="https://arxiv.org/pdf/1706.03762.pdf",
        )
    ]
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_arxiv",
                        "args": {"query": "original transformer paper"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content=(
                    "The original Transformer paper is Attention Is All "
                    "You Need (arXiv:1706.03762)."
                )
            ),
        ]
    )
    research_agent = create_research_agent(
        model=model,
        dependencies=AgentToolDependencies(
            rag_service=MagicMock(),
            arxiv_service=arxiv_service,
            project_service=MagicMock(),
        ),
        user_id=user_id,
    )
    message_repository = InMemoryMessageRepository()
    service = ChatService(
        chat_repository=SimpleNamespace(
            get_by_id=lambda requested_id: SimpleNamespace(
                chat_id=requested_id,
                project_id=project_id,
            )
        ),
        project_repository=SimpleNamespace(
            get_by_id=lambda requested_id: SimpleNamespace(
                project_id=requested_id,
                user_id=user_id,
            )
        ),
        message_repository=message_repository,
        citation_repository=SimpleNamespace(
            create_many=lambda **kwargs: [],
        ),
        research_agent=research_agent,
    )

    result = service.send_message(
        user_id=user_id,
        chat_id=chat_id,
        content="Find the original Transformer paper.",
    )

    assert result.answer.startswith("The original Transformer paper")
    assert [message.role for message in message_repository.messages] == [
        "user",
        "assistant",
    ]
    assert [message.content for message in message_repository.messages] == [
        "Find the original Transformer paper.",
        (
            "The original Transformer paper is Attention Is All You Need "
            "(arXiv:1706.03762)."
        ),
    ]
    assert len(model.seen_messages) == 2
    assert any(
        isinstance(message, ToolMessage)
        for message in model.seen_messages[1]
    )
    assert all(
        message.role in {"user", "assistant"}
        for message in message_repository.messages
    )
