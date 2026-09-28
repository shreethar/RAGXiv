from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.dependencies import (
    get_arxiv_service,
    get_current_user_id,
    get_paper_service,
)
from app.api.errors import call_application
from app.api.schemas.common import NonEmptyString
from app.api.schemas.papers import (
    ArxivPaperResponse,
    ArxivSearchResponse,
    PaperIngestRequest,
    PaperResponse,
)
from app.infrastructure.arxiv import ArxivService
from app.services.paper_service import PaperService


router = APIRouter(prefix="/papers", tags=["papers"])


@router.post("/ingest", response_model=PaperResponse)
def ingest_paper(
    request: PaperIngestRequest,
    _user_id: Annotated[UUID, Depends(get_current_user_id)],
    service: Annotated[PaperService, Depends(get_paper_service)],
) -> PaperResponse:
    paper = call_application(
        lambda: service.ingest(arxiv_id=request.arxiv_id)
    )
    return PaperResponse.model_validate(paper)


@router.get("/search", response_model=ArxivSearchResponse)
def search_arxiv(
    q: Annotated[NonEmptyString, Query()],
    _user_id: Annotated[UUID, Depends(get_current_user_id)],
    service: Annotated[ArxivService, Depends(get_arxiv_service)],
) -> ArxivSearchResponse:
    papers = call_application(lambda: service.search(query=q))
    return ArxivSearchResponse(
        papers=[ArxivPaperResponse.model_validate(item) for item in papers]
    )


@router.get("/arxiv/{arxiv_id:path}", response_model=ArxivPaperResponse)
def get_arxiv_metadata(
    arxiv_id: str,
    _user_id: Annotated[UUID, Depends(get_current_user_id)],
    service: Annotated[ArxivService, Depends(get_arxiv_service)],
) -> ArxivPaperResponse:
    normalized_arxiv_id = arxiv_id.strip()

    if not normalized_arxiv_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="arxiv_id must not be empty",
        )

    metadata = call_application(
        lambda: service.get_metadata(arxiv_id=normalized_arxiv_id)
    )
    return ArxivPaperResponse.model_validate(metadata)
