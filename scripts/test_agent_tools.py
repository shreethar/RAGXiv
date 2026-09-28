from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.agents.tools import AgentToolDependencies, create_agent_tools
from app.infrastructure.arxiv import ArxivPaperMetadata, ArxivSearchResult
from app.infrastructure.vector_store import RetrievedChunk
from app.domain.citations import CitationReference, SourceCitation
from app.services.rag_service import RagResult


@pytest.fixture
def tool_context():
    rag_service = MagicMock()
    arxiv_service = MagicMock()
    project_service = MagicMock()
    user_id = uuid4()

    tools = create_agent_tools(
        dependencies=AgentToolDependencies(
            rag_service=rag_service,
            arxiv_service=arxiv_service,
            project_service=project_service,
        ),
        user_id=user_id,
    )

    return {
        "tools": {agent_tool.name: agent_tool for agent_tool in tools},
        "rag_service": rag_service,
        "arxiv_service": arxiv_service,
        "project_service": project_service,
        "user_id": user_id,
    }


def test_query_paper_invokes_rag_service_and_returns_sources(tool_context):
    paper_id = uuid4()
    chunk_id = uuid4()
    tool_context["rag_service"].query_paper.return_value = RagResult(
        query="What is attention?",
        answer="Attention combines weighted value vectors.",
        chunks=[
            RetrievedChunk(
                chunk_id=chunk_id,
                text="Attention maps a query and key-value pairs to output.",
                score=0.12,
                chunk_index=7,
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

    output = tool_context["tools"]["query_paper"].invoke(
        {
            "question": "What is attention?",
            "paper_id": str(paper_id),
        }
    )

    tool_context["rag_service"].query_paper.assert_called_once_with(
        question="What is attention?",
        paper_id=paper_id,
    )
    assert output["answer"] == "Attention combines weighted value vectors."
    assert output["sources"] == [
        {
            "source_id": "source_1",
            "chunk_index": 7,
            "excerpt": "Attention maps a query and key-value pairs to output.",
            "cited": True,
        }
    ]
    assert output["citations"] == [
        {"source_id": "source_1", "chunk_index": 7}
    ]
    assert str(chunk_id) not in str(output)
    assert "score" not in str(output)
    assert "limit" not in tool_context["tools"]["query_paper"].args


def test_search_arxiv_invokes_service_and_serializes_results(tool_context):
    published_at = datetime(2017, 6, 12, tzinfo=timezone.utc)
    tool_context["arxiv_service"].search.return_value = [
        ArxivSearchResult(
            arxiv_id="1706.03762",
            title="Attention Is All You Need",
            abstract="A sequence transduction architecture.",
            authors=["Ashish Vaswani"],
            published_at=published_at,
            pdf_url="https://arxiv.org/pdf/1706.03762.pdf",
        )
    ]

    output = tool_context["tools"]["search_arxiv"].invoke(
        {"query": "attention transformer"}
    )

    tool_context["arxiv_service"].search.assert_called_once_with(
        query="attention transformer"
    )
    assert output["papers"][0]["arxiv_id"] == "1706.03762"
    assert output["papers"][0]["published_at"] == published_at.isoformat()


def test_get_paper_metadata_invokes_service_and_serializes_result(tool_context):
    published_at = datetime(2017, 6, 12, tzinfo=timezone.utc)
    tool_context["arxiv_service"].get_metadata.return_value = (
        ArxivPaperMetadata(
            arxiv_id="1706.03762",
            title="Attention Is All You Need",
            abstract="A sequence transduction architecture.",
            authors=["Ashish Vaswani"],
            version="v7",
            published_at=published_at,
            pdf_url="https://arxiv.org/pdf/1706.03762.pdf",
        )
    )

    output = tool_context["tools"]["get_paper_metadata"].invoke(
        {"arxiv_id": "1706.03762"}
    )

    tool_context["arxiv_service"].get_metadata.assert_called_once_with(
        arxiv_id="1706.03762"
    )
    assert output["title"] == "Attention Is All You Need"
    assert output["version"] == "v7"
    assert output["published_at"] == published_at.isoformat()


def test_list_project_papers_uses_authenticated_user_context(tool_context):
    project_id = uuid4()
    paper_id = uuid4()
    tool_context["project_service"].list_papers.return_value = [
        SimpleNamespace(
            paper_id=paper_id,
            arxiv_id="1706.03762",
            title="Attention Is All You Need",
            abstract="A sequence transduction architecture.",
            authors="Ashish Vaswani",
            version="v7",
            published_at=None,
            pdf_url="https://arxiv.org/pdf/1706.03762.pdf",
            status="ready",
        )
    ]

    project_tool = tool_context["tools"]["list_project_papers"]
    output = project_tool.invoke({"project_id": str(project_id)})

    tool_context["project_service"].list_papers.assert_called_once_with(
        user_id=tool_context["user_id"],
        project_id=project_id,
    )
    assert "user_id" not in project_tool.args
    assert output["papers"][0]["paper_id"] == str(paper_id)
    assert output["papers"][0]["status"] == "ready"


@pytest.mark.parametrize("tool_name, arguments", [
    (
        "query_paper",
        {"question": "A question", "paper_id": "not-a-uuid"},
    ),
    (
        "list_project_papers",
        {"project_id": "not-a-uuid"},
    ),
    (
        "search_arxiv",
        {"query": ""},
    ),
    (
        "get_paper_metadata",
        {"arxiv_id": ""},
    ),
])
def test_tool_arguments_are_validated(tool_context, tool_name, arguments):
    with pytest.raises(ValidationError):
        tool_context["tools"][tool_name].invoke(arguments)


@pytest.mark.parametrize("tool_name, arguments, service_name, method_name", [
    (
        "query_paper",
        {"question": "A question", "paper_id": str(uuid4())},
        "rag_service",
        "query_paper",
    ),
    (
        "search_arxiv",
        {"query": "transformers"},
        "arxiv_service",
        "search",
    ),
    (
        "get_paper_metadata",
        {"arxiv_id": "1706.03762"},
        "arxiv_service",
        "get_metadata",
    ),
    (
        "list_project_papers",
        {"project_id": str(uuid4())},
        "project_service",
        "list_papers",
    ),
])
def test_service_errors_become_model_visible_tool_errors(
    tool_context,
    tool_name,
    arguments,
    service_name,
    method_name,
):
    service = tool_context[service_name]
    getattr(service, method_name).side_effect = ValueError("not available")

    output = tool_context["tools"][tool_name].invoke(arguments)

    assert output.startswith("Error: ")
    assert "not available" in output
