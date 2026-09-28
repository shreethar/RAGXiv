from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends

from app.api.dependencies import get_chat_service, get_current_user_id
from app.api.errors import call_application
from app.api.schemas.chats import CitationDetailResponse
from app.services.chat_service import ChatService


router = APIRouter(prefix="/citations", tags=["citations"])


@router.get("/{citation_id}", response_model=CitationDetailResponse)
def get_citation(
    citation_id: UUID,
    user_id: Annotated[UUID, Depends(get_current_user_id)],
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> CitationDetailResponse:
    citation = call_application(
        lambda: service.get_citation(
            user_id=user_id,
            citation_id=citation_id,
        )
    )
    return CitationDetailResponse.from_citation(citation)
