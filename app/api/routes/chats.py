from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, status

from app.api.dependencies import get_chat_service, get_current_user_id
from app.api.errors import call_application
from app.api.schemas.chats import (
    AssistantMessageResponse,
    CitationSummaryResponse,
    ChatCreateRequest,
    ChatResponse,
    MessageListResponse,
    MessageResponse,
    MessageSendRequest,
)
from app.services.chat_service import ChatService


router = APIRouter(prefix="/chats", tags=["chats"])


@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_chat(
    request: ChatCreateRequest,
    user_id: Annotated[UUID, Depends(get_current_user_id)],
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> ChatResponse:
    chat = call_application(
        lambda: service.create_chat(
            user_id=user_id,
            project_id=request.project_id,
            title=request.title,
        )
    )
    return ChatResponse.model_validate(chat)


@router.get("/{chat_id}", response_model=ChatResponse)
def get_chat(
    chat_id: UUID,
    user_id: Annotated[UUID, Depends(get_current_user_id)],
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> ChatResponse:
    chat = call_application(
        lambda: service.get_chat(user_id=user_id, chat_id=chat_id)
    )
    return ChatResponse.model_validate(chat)


@router.get(
    "/{chat_id}/messages",
    response_model=MessageListResponse,
)
def list_messages(
    chat_id: UUID,
    user_id: Annotated[UUID, Depends(get_current_user_id)],
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> MessageListResponse:
    messages = call_application(
        lambda: service.list_messages(
            user_id=user_id,
            chat_id=chat_id,
        )
    )
    return MessageListResponse(
        messages=[MessageResponse.from_message(item) for item in messages]
    )


@router.post(
    "/{chat_id}/messages",
    response_model=AssistantMessageResponse,
)
def send_message(
    chat_id: UUID,
    request: MessageSendRequest,
    user_id: Annotated[UUID, Depends(get_current_user_id)],
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> AssistantMessageResponse:
    result = call_application(
        lambda: service.send_message(
            user_id=user_id,
            chat_id=chat_id,
            content=request.content,
        )
    )
    return AssistantMessageResponse(
        message_id=result.assistant_message_id,
        chat_id=result.chat_id,
        content=result.answer,
        citations=[
            CitationSummaryResponse.from_citation(citation)
            for citation in result.citations
        ],
    )
