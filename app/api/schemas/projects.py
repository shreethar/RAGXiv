from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.api.schemas.common import NonEmptyString
from app.api.schemas.papers import PaperResponse


class ProjectCreateRequest(BaseModel):
    name: NonEmptyString


class ProjectResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    project_id: UUID
    name: str
    created_at: datetime
    updated_at: datetime


class ProjectListResponse(BaseModel):
    projects: list[ProjectResponse]


class ProjectPaperAddRequest(BaseModel):
    paper_ids: list[UUID] = Field(min_length=1)


class ProjectPapersResponse(BaseModel):
    papers: list[PaperResponse]
