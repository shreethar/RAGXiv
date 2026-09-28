from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from app.api.schemas.common import NonEmptyString


class PaperIngestRequest(BaseModel):
    arxiv_id: NonEmptyString


class PaperResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    paper_id: UUID
    arxiv_id: str
    title: str
    abstract: str | None
    authors: str | None
    version: str | None
    published_at: datetime | None
    pdf_url: str | None
    status: str


class ArxivPaperResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    published_at: datetime
    pdf_url: str
    version: str | None = None


class ArxivSearchResponse(BaseModel):
    papers: list[ArxivPaperResponse]
