from datetime import datetime, timezone
from typing import Any, Sequence
from unittest.mock import MagicMock
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from pydantic import Field, PrivateAttr

from app.agents.research_agent import (
    ResearchAgent,
    ResearchMessage,
    create_research_agent,
)
from app.agents.tools import AgentToolDependencies, QueryPaperArtifact
from app.domain.citations import CitationReference, SourceCitation
from app.infrastructure.arxiv import ArxivSearchResult
from app.infrastructure.vector_store import RetrievedChunk
from app.services.rag_service import RagResult


class ScriptedChatModel(BaseChatModel):
    responses: list[AIMessage] = Field(default_factory=list)
    _response_index: int = PrivateAttr(default=0)
    _seen_messages: list[list[BaseMessage]] = PrivateAttr(default_factory=list)
    _bound_tool_names: list[str] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-test-model"

    @property
    def seen_messages(self) -> list[list[BaseMessage]]:
        return self._seen_messages

    @property
    def bound_tool_names(self) -> list[str]:
        return self._bound_tool_names

    def bind_tools(
        self,
        tools: Sequence[BaseTool | dict[str, Any]],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> "ScriptedChatModel":
        self._bound_tool_names = [
            item.name if isinstance(item, BaseTool) else item["function"]["name"]
            for item in tools
        ]
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


def _dependencies(
    *,
    arxiv_service: MagicMock | None = None,
    rag_service: MagicMock | None = None,
) -> AgentToolDependencies:
    return AgentToolDependencies(
        rag_service=rag_service or MagicMock(),
        arxiv_service=arxiv_service or MagicMock(),
        project_service=MagicMock(),
    )


def test_agent_can_complete_a_direct_conversational_response():
    model = ScriptedChatModel(
        responses=[AIMessage(content="How can I help with your research?")]
    )
    agent = create_research_agent(
        model=model,
        dependencies=_dependencies(),
        user_id=uuid4(),
    )

    result = agent.invoke(
        messages=[ResearchMessage(role="user", content="Hello")]
    )

    assert result.answer == (
        "How can I help with your research?"
    )
    assert set(model.bound_tool_names) == {
        "query_paper",
        "search_arxiv",
        "get_paper_metadata",
        "list_project_papers",
    }
    assert len(model.seen_messages) == 1


def test_agent_invokes_tool_feeds_result_back_and_reaches_final_response():
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
                        "args": {"query": "transformer attention"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content=(
                    "I found Attention Is All You Need (arXiv:1706.03762)."
                )
            ),
        ]
    )
    agent = create_research_agent(
        model=model,
        dependencies=_dependencies(arxiv_service=arxiv_service),
        user_id=uuid4(),
    )

    result = agent.invoke(
        messages=[
            ResearchMessage(
                role="user",
                content="Find the original Transformer paper."
            )
        ]
    )

    arxiv_service.search.assert_called_once_with(
        query="transformer attention"
    )
    assert len(model.seen_messages) == 2
    tool_messages = [
        message
        for message in model.seen_messages[1]
        if isinstance(message, ToolMessage)
    ]
    assert len(tool_messages) == 1
    assert "Attention Is All You Need" in tool_messages[0].content
    assert result.answer == (
        "I found Attention Is All You Need (arXiv:1706.03762)."
    )


def test_query_paper_tool_citations_reach_typed_agent_result():
    paper_id = uuid4()
    chunk_id = uuid4()
    rag_service = MagicMock()
    rag_service.query_paper.return_value = RagResult(
        query="What supports the claim?",
        answer="The paper supports the claim.",
        chunks=[
            RetrievedChunk(
                chunk_id=chunk_id,
                text="Supporting evidence.",
                score=0.1,
                chunk_index=4,
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
                            "question": "What supports the claim?",
                            "paper_id": str(paper_id),
                        },
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The paper supports the claim."),
        ]
    )
    agent = create_research_agent(
        model=model,
        dependencies=_dependencies(rag_service=rag_service),
        user_id=uuid4(),
    )

    result = agent.invoke(
        messages=[ResearchMessage(role="user", content="Question")]
    )

    assert result.answer == "The paper supports the claim."
    assert result.citations == (
        CitationReference(paper_id=paper_id, chunk_id=chunk_id),
    )
    tool_messages = [
        message
        for message in model.seen_messages[1]
        if isinstance(message, ToolMessage)
    ]
    assert len(tool_messages) == 1
    assert str(chunk_id) not in tool_messages[0].content


def test_agent_result_can_combine_citations_from_multiple_papers():
    references = (
        CitationReference(paper_id=uuid4(), chunk_id=uuid4()),
        CitationReference(paper_id=uuid4(), chunk_id=uuid4()),
    )

    class FakeGraph:
        def invoke(self, state):
            return {
                "messages": [
                    *state["messages"],
                    ToolMessage(
                        content="First tool result",
                        tool_call_id="call-1",
                        artifact=QueryPaperArtifact(
                            citations=(references[0],)
                        ),
                    ),
                    ToolMessage(
                        content="Second tool result",
                        tool_call_id="call-2",
                        artifact=QueryPaperArtifact(
                            citations=(references[1], references[0])
                        ),
                    ),
                    AIMessage(content="A comparison of both papers."),
                ]
            }

    result = ResearchAgent(FakeGraph()).invoke(
        messages=[ResearchMessage(role="user", content="Compare them")]
    )

    assert result.citations == references
