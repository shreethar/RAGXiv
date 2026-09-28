from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.api.schemas.common import NonEmptyString


class ChatCreateRequest(BaseModel):
    project_id: UUID
    title: NonEmptyString | None = None


class ChatResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    chat_id: UUID
    project_id: UUID
    title: str | None
    created_at: datetime
    updated_at: datetime


class MessageSendRequest(BaseModel):
    content: NonEmptyString


class AssistantMessageResponse(BaseModel):
    message_id: UUID
    chat_id: UUID
    role: Literal["assistant"] = "assistant"
    content: str
    citations: list["CitationSummaryResponse"] = Field(
        default_factory=list
    )


class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    message_id: UUID
    role: str
    content: str
    created_at: datetime
    citations: list["CitationSummaryResponse"] = Field(
        default_factory=list
    )

    @classmethod
    def from_message(cls, message) -> "MessageResponse":
        return cls(
            message_id=message.message_id,
            role=message.role,
            content=message.content,
            created_at=message.created_at,
            citations=[
                CitationSummaryResponse.from_citation(citation)
                for citation in message.citations
            ],
        )


class MessageListResponse(BaseModel):
    messages: list[MessageResponse]


class CitationPaperResponse(BaseModel):
    paper_id: UUID
    arxiv_id: str
    title: str


class CitationChunkResponse(BaseModel):
    chunk_id: UUID
    chunk_index: int


class CitationSummaryResponse(BaseModel):
    citation_id: UUID
    position: int
    paper: CitationPaperResponse
    chunk: CitationChunkResponse

    @classmethod
    def from_citation(cls, citation) -> "CitationSummaryResponse":
        chunk = citation.chunk
        paper = chunk.paper
        return cls(
            citation_id=citation.citation_id,
            position=citation.position,
            paper=CitationPaperResponse(
                paper_id=paper.paper_id,
                arxiv_id=paper.arxiv_id,
                title=paper.title,
            ),
            chunk=CitationChunkResponse(
                chunk_id=chunk.chunk_id,
                chunk_index=chunk.chunk_index,
            ),
        )


class CitationDetailResponse(CitationSummaryResponse):
    content: str

    @classmethod
    def from_citation(cls, citation) -> "CitationDetailResponse":
        summary = CitationSummaryResponse.from_citation(citation)
        return cls(
            **summary.model_dump(),
            content=citation.chunk.content,
        )
