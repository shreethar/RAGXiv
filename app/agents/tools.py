from dataclasses import dataclass
from typing import Any, Callable, TypeVar
from uuid import UUID

from langchain.tools import tool
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field

from app.infrastructure.arxiv import ArxivService
from app.domain.citations import CitationReference
from app.services.project_service import ProjectService
from app.services.rag_service import RAGService


@dataclass(frozen=True)
class AgentToolDependencies:
    """Services required by the read-only research agent tools."""

    rag_service: RAGService
    arxiv_service: ArxivService
    project_service: ProjectService


@dataclass(frozen=True)
class QueryPaperArtifact:
    citations: tuple[CitationReference, ...]


class QueryPaperInput(BaseModel):
    question: str = Field(
        min_length=1,
        description="The question to answer from the paper.",
    )
    paper_id: UUID = Field(
        description="The internal UUID of the paper to query.",
    )


class SearchArxivInput(BaseModel):
    query: str = Field(
        min_length=1,
        description="An arXiv search query.",
    )


class GetPaperMetadataInput(BaseModel):
    arxiv_id: str = Field(
        min_length=1,
        description="An arXiv identifier, for example 1706.03762.",
    )


class ListProjectPapersInput(BaseModel):
    project_id: UUID = Field(
        description="The UUID of the project whose papers should be listed.",
    )


T = TypeVar("T")


def _call_service(
    operation: str,
    callback: Callable[[], T],
) -> T:
    try:
        return callback()
    except Exception as error:
        raise ToolException(f"{operation} failed: {error}") from error


def _published_at(value: Any) -> str | None:
    return value.isoformat() if value is not None else None


def create_agent_tools(
    *,
    dependencies: AgentToolDependencies,
    user_id: UUID,
) -> list[BaseTool]:
    """Create read-only tools bound to services and an authenticated user."""

    @tool(
        "query_paper",
        args_schema=QueryPaperInput,
        response_format="content_and_artifact",
        description=(
            "Answer a question using an already-ingested paper. Returns a "
            "grounded answer and the source excerpts used to produce it."
        ),
    )
    def query_paper(
        question: str,
        paper_id: UUID,
    ) -> tuple[dict[str, Any], QueryPaperArtifact]:
        result = _call_service(
            "Paper query",
            lambda: dependencies.rag_service.query_paper(
                question=question,
                paper_id=paper_id,
            ),
        )

        citations_by_source = {
            citation.source_id: citation
            for citation in result.citations
        }
        content = {
            "question": result.query,
            "answer": result.answer,
            "sources": [
                {
                    "source_id": f"source_{index}",
                    "chunk_index": chunk.chunk_index,
                    "excerpt": chunk.text,
                    "cited": f"source_{index}" in citations_by_source,
                }
                for index, chunk in enumerate(result.chunks, start=1)
            ],
            "citations": [
                {
                    "source_id": citation.source_id,
                    "chunk_index": next(
                        chunk.chunk_index
                        for chunk in result.chunks
                        if chunk.chunk_id == citation.reference.chunk_id
                    ),
                }
                for citation in result.citations
            ],
        }
        artifact = QueryPaperArtifact(
            citations=tuple(
                citation.reference
                for citation in result.citations
            )
        )
        return content, artifact

    @tool(
        "search_arxiv",
        args_schema=SearchArxivInput,
        description=(
            "Search arXiv for research papers. Returns identifiers, titles, "
            "abstracts, authors, publication dates, and PDF URLs."
        ),
    )
    def search_arxiv(query: str) -> dict[str, Any]:
        results = _call_service(
            "arXiv search",
            lambda: dependencies.arxiv_service.search(query=query),
        )

        return {
            "query": query,
            "papers": [
                {
                    "arxiv_id": paper.arxiv_id,
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "authors": paper.authors,
                    "published_at": _published_at(paper.published_at),
                    "pdf_url": paper.pdf_url,
                }
                for paper in results
            ],
        }

    @tool(
        "get_paper_metadata",
        args_schema=GetPaperMetadataInput,
        description=(
            "Get current arXiv metadata for a paper from its arXiv identifier."
        ),
    )
    def get_paper_metadata(arxiv_id: str) -> dict[str, Any]:
        metadata = _call_service(
            "Paper metadata lookup",
            lambda: dependencies.arxiv_service.get_metadata(
                arxiv_id=arxiv_id,
            ),
        )

        return {
            "arxiv_id": metadata.arxiv_id,
            "title": metadata.title,
            "abstract": metadata.abstract,
            "authors": metadata.authors,
            "version": metadata.version,
            "published_at": _published_at(metadata.published_at),
            "pdf_url": metadata.pdf_url,
        }

    @tool(
        "list_project_papers",
        args_schema=ListProjectPapersInput,
        description=(
            "List the papers in one of the authenticated user's projects."
        ),
    )
    def list_project_papers(project_id: UUID) -> dict[str, Any]:
        papers = _call_service(
            "Project paper listing",
            lambda: dependencies.project_service.list_papers(
                user_id=user_id,
                project_id=project_id,
            ),
        )

        return {
            "project_id": str(project_id),
            "papers": [
                {
                    "paper_id": str(paper.paper_id),
                    "arxiv_id": paper.arxiv_id,
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "authors": paper.authors,
                    "version": paper.version,
                    "published_at": _published_at(paper.published_at),
                    "pdf_url": paper.pdf_url,
                    "status": paper.status,
                }
                for paper in papers
            ],
        }

    tools = [
        query_paper,
        search_arxiv,
        get_paper_metadata,
        list_project_papers,
    ]

    for agent_tool in tools:
        agent_tool.handle_tool_error = (
            lambda error: f"Error: {error}"
        )

    return tools
