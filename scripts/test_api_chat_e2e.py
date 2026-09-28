from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Sequence
from unittest.mock import MagicMock
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from pydantic import Field, PrivateAttr

from app.agents.research_agent import create_research_agent
from app.agents.tools import AgentToolDependencies
from app.domain.citations import CitationReference, SourceCitation
from app.api.dependencies import get_chat_service, get_current_user_id
from app.infrastructure.arxiv import ArxivSearchResult
from app.infrastructure.vector_store import RetrievedChunk
from app.main import create_app
from app.services.chat_service import ChatService
from app.services.rag_service import RagResult


class ScriptedChatModel(BaseChatModel):
    responses: list[AIMessage] = Field(default_factory=list)
    _response_index: int = PrivateAttr(default=0)
    _seen_messages: list[list[BaseMessage]] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "api-e2e-scripted-model"

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
        return ChatResult(generations=[ChatGeneration(message=response)])


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
            created_at=datetime.now(timezone.utc),
        )
        self.messages.append(message)
        return message

    def list_by_chat(self, chat_id: UUID) -> list[SimpleNamespace]:
        return [item for item in self.messages if item.chat_id == chat_id]


class InMemoryCitationRepository:
    def __init__(self, chunks_by_id):
        self.chunks_by_id = chunks_by_id
        self.citations = []

    def create_many(self, *, message_id, references):
        created = []
        for position, reference in enumerate(references, start=1):
            citation = SimpleNamespace(
                citation_id=uuid4(),
                message_id=message_id,
                chunk_id=reference.chunk_id,
                position=position,
                chunk=self.chunks_by_id[reference.chunk_id],
            )
            self.citations.append(citation)
            created.append(citation)
        return created


def test_http_chat_message_runs_through_chat_service_and_agent_tool_loop():
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
    agent = create_research_agent(
        model=model,
        dependencies=AgentToolDependencies(
            rag_service=MagicMock(),
            arxiv_service=arxiv_service,
            project_service=MagicMock(),
        ),
        user_id=user_id,
    )
    message_repository = InMemoryMessageRepository()
    chat_service = ChatService(
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
        research_agent=agent,
    )
    app = create_app()
    app.dependency_overrides[get_current_user_id] = lambda: user_id
    app.dependency_overrides[get_chat_service] = lambda: chat_service

    with TestClient(app) as client:
        response = client.post(
            f"/chats/{chat_id}/messages",
            json={"content": "Find the original Transformer paper."},
        )

    assert response.status_code == 200
    assert response.json()["role"] == "assistant"
    assert response.json()["content"].startswith(
        "The original Transformer paper"
    )
    assert [message.role for message in message_repository.messages] == [
        "user",
        "assistant",
    ]
    assert len(model.seen_messages) == 2
    assert any(
        isinstance(message, ToolMessage)
        for message in model.seen_messages[1]
    )


def test_http_query_paper_citation_is_persisted_and_returned():
    user_id = uuid4()
    project_id = uuid4()
    chat_id = uuid4()
    paper_id = uuid4()
    chunk_id = uuid4()
    paper = SimpleNamespace(
        paper_id=paper_id,
        arxiv_id="1706.03762",
        title="Attention Is All You Need",
    )
    chunk = SimpleNamespace(
        chunk_id=chunk_id,
        paper_id=paper_id,
        chunk_index=12,
        content="Multi-head attention attends in parallel.",
        paper=paper,
    )
    rag_service = MagicMock()
    rag_service.query_paper.return_value = RagResult(
        query="What is multi-head attention?",
        answer="It uses parallel attention heads.",
        chunks=[
            RetrievedChunk(
                chunk_id=chunk_id,
                text=chunk.content,
                score=0.1,
                chunk_index=chunk.chunk_index,
            )
        ],
        citations=[
            SourceCitation(
                source_id="source_1",
                reference=CitationReference(
                    paper_id=paper_id,
                    chunk_id=chunk_id,
                ),
            )
        ],
    )
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "query_paper",
                        "args": {
                            "question": "What is multi-head attention?",
                            "paper_id": str(paper_id),
                        },
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="It uses parallel attention heads."),
        ]
    )
    agent = create_research_agent(
        model=model,
        dependencies=AgentToolDependencies(
            rag_service=rag_service,
            arxiv_service=MagicMock(),
            project_service=MagicMock(),
        ),
        user_id=user_id,
    )
    messages = InMemoryMessageRepository()
    citations = InMemoryCitationRepository({chunk_id: chunk})
    chat_service = ChatService(
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
        message_repository=messages,
        citation_repository=citations,
        research_agent=agent,
    )
    app = create_app()
    app.dependency_overrides[get_current_user_id] = lambda: user_id
    app.dependency_overrides[get_chat_service] = lambda: chat_service

    with TestClient(app) as client:
        response = client.post(
            f"/chats/{chat_id}/messages",
            json={"content": "What is multi-head attention?"},
        )

    assert response.status_code == 200
    returned = response.json()["citations"]
    assert len(returned) == 1
    assert returned[0]["paper"]["paper_id"] == str(paper_id)
    assert returned[0]["chunk"]["chunk_id"] == str(chunk_id)
    assert returned[0]["chunk"]["chunk_index"] == 12
    assert len(citations.citations) == 1
    assert citations.citations[0].message_id == messages.messages[1].message_id
    assert [message.role for message in messages.messages] == [
        "user",
        "assistant",
    ]
