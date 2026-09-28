from dataclasses import dataclass
from collections.abc import Sequence
from typing import Literal
from uuid import UUID

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langgraph.graph.state import CompiledStateGraph

from app.agents.tools import (
    AgentToolDependencies,
    QueryPaperArtifact,
    create_agent_tools,
)
from app.domain.citations import CitationReference


RESEARCH_AGENT_SYSTEM_PROMPT = """You are RAGXiv, an assistant for research papers.

You can search arXiv, inspect a paper's metadata, inspect the papers in the
authenticated user's projects, and retrieve grounded information from papers
that have already been ingested.

Use tools whenever the user's request depends on paper, arXiv, or project
information that has not already been provided. A project may contain several
papers; query relevant papers separately when information from multiple papers
is needed, then synthesize the results. Do not invent paper details or claims.
If the available context does not support an answer, say so clearly.

You have read-only tools. Do not claim to ingest papers or modify projects.
"""


@dataclass(frozen=True)
class ResearchAgentResult:
    answer: str
    citations: tuple[CitationReference, ...] = ()


@dataclass(frozen=True)
class ResearchMessage:
    role: Literal["user", "assistant"]
    content: str


class ResearchAgent:
    """Typed application-facing adapter around a LangGraph agent."""

    def __init__(self, graph: CompiledStateGraph):
        self._graph = graph

    def invoke(
        self,
        *,
        messages: Sequence[ResearchMessage],
    ) -> ResearchAgentResult:
        state = self._graph.invoke(
            {
                "messages": [
                    self._to_langchain_message(message)
                    for message in messages
                ]
            }
        )

        state_messages = state.get("messages", [])

        if not state_messages:
            raise RuntimeError(
                "Research agent returned no messages"
            )

        final_message = state_messages[-1]

        if not isinstance(final_message, AIMessage):
            raise RuntimeError(
                "Research agent did not return a final assistant message"
            )

        if final_message.tool_calls:
            raise RuntimeError(
                "Research agent stopped before producing a final answer"
            )

        answer = str(final_message.text).strip()

        if not answer:
            raise RuntimeError(
                "Research agent returned an empty final answer"
            )

        return ResearchAgentResult(
            answer=answer,
            citations=self._extract_citations(state_messages),
        )

    @staticmethod
    def _extract_citations(
        messages: Sequence[BaseMessage],
    ) -> tuple[CitationReference, ...]:
        citations: list[CitationReference] = []
        seen: set[tuple[UUID, UUID]] = set()

        for message in messages:
            if not isinstance(message, ToolMessage):
                continue

            artifact = message.artifact

            if not isinstance(artifact, QueryPaperArtifact):
                continue

            for citation in artifact.citations:
                identity = (citation.paper_id, citation.chunk_id)

                if identity in seen:
                    continue

                seen.add(identity)
                citations.append(citation)

        return tuple(citations)

    @staticmethod
    def _to_langchain_message(
        message: ResearchMessage,
    ) -> BaseMessage:
        if message.role == "user":
            return HumanMessage(content=message.content)

        return AIMessage(content=message.content)


def create_research_agent(
    *,
    model: BaseChatModel,
    dependencies: AgentToolDependencies,
    user_id: UUID,
) -> ResearchAgent:
    """Build a LangGraph-backed research agent for one authenticated user."""

    tools = create_agent_tools(
        dependencies=dependencies,
        user_id=user_id,
    )

    return ResearchAgent(
        create_agent(
            model=model,
            tools=tools,
            system_prompt=RESEARCH_AGENT_SYSTEM_PROMPT,
            name="ragxiv_research_agent",
        )
    )
