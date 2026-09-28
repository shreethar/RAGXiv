from collections.abc import Generator
from typing import Annotated
from uuid import UUID

import httpx
from fastapi import Depends, Header, HTTPException, status
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from openai import OpenAI
from sqlalchemy.orm import Session

from app.agents.research_agent import ResearchAgent, create_research_agent
from app.agents.tools import AgentToolDependencies
from app.db.database import SessionLocal
from app.infrastructure.arxiv import ArxivService
from app.infrastructure.embeddings import OpenAIEmbeddingService
from app.infrastructure.llm import DEFAULT_LLM_MODEL, OpenAILLMService
from app.infrastructure.pdf_document_processor import PdfDocumentProcessor
from app.infrastructure.pg_vector_store import PgVectorStore
from app.infrastructure.text_chunker import TextChunker
from app.repositories.chat import ChatRepository
from app.repositories.citation import CitationRepository
from app.repositories.message import MessageRepository
from app.repositories.paper import PaperRepository
from app.repositories.project import ProjectRepository
from app.services.chat_service import ChatService
from app.services.paper_service import PaperService
from app.services.project_service import ProjectService
from app.services.rag_service import RAGService


def get_current_user_id(
    x_user_id: Annotated[
        str | None,
        Header(alias="X-User-ID"),
    ] = None,
) -> UUID:
    """Temporary development authentication via the X-User-ID header."""

    if x_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-User-ID header is required",
        )

    try:
        return UUID(x_user_id)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-User-ID header must be a valid UUID",
        ) from error


def get_session() -> Generator[Session, None, None]:
    session = SessionLocal()

    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_http_client() -> Generator[httpx.Client, None, None]:
    with httpx.Client(timeout=60.0) as client:
        yield client


def get_openai_client() -> Generator[OpenAI, None, None]:
    client = OpenAI()

    try:
        yield client
    finally:
        client.close()


def get_agent_chat_model() -> BaseChatModel:
    return ChatOpenAI(model=DEFAULT_LLM_MODEL)


def get_project_repository(
    session: Annotated[Session, Depends(get_session)],
) -> ProjectRepository:
    return ProjectRepository(session)


def get_paper_repository(
    session: Annotated[Session, Depends(get_session)],
) -> PaperRepository:
    return PaperRepository(session)


def get_chat_repository(
    session: Annotated[Session, Depends(get_session)],
) -> ChatRepository:
    return ChatRepository(session)


def get_message_repository(
    session: Annotated[Session, Depends(get_session)],
) -> MessageRepository:
    return MessageRepository(session)


def get_citation_repository(
    session: Annotated[Session, Depends(get_session)],
) -> CitationRepository:
    return CitationRepository(session)


def get_arxiv_service(
    client: Annotated[httpx.Client, Depends(get_http_client)],
) -> ArxivService:
    return ArxivService(client=client)


def get_project_service(
    project_repository: Annotated[
        ProjectRepository,
        Depends(get_project_repository),
    ],
    paper_repository: Annotated[
        PaperRepository,
        Depends(get_paper_repository),
    ],
) -> ProjectService:
    return ProjectService(
        project_repository=project_repository,
        paper_repository=paper_repository,
    )


def get_rag_service(
    session: Annotated[Session, Depends(get_session)],
    openai_client: Annotated[OpenAI, Depends(get_openai_client)],
    paper_repository: Annotated[
        PaperRepository,
        Depends(get_paper_repository),
    ],
) -> RAGService:
    return RAGService(
        embedding_service=OpenAIEmbeddingService(
            client=openai_client,
        ),
        vector_store=PgVectorStore(session=session),
        llm_service=OpenAILLMService(client=openai_client),
        paper_repository=paper_repository,
    )


def get_paper_service(
    session: Annotated[Session, Depends(get_session)],
    paper_repository: Annotated[
        PaperRepository,
        Depends(get_paper_repository),
    ],
    arxiv_service: Annotated[
        ArxivService,
        Depends(get_arxiv_service),
    ],
    openai_client: Annotated[OpenAI, Depends(get_openai_client)],
) -> PaperService:
    return PaperService(
        paper_repository=paper_repository,
        arxiv_service=arxiv_service,
        document_processor=PdfDocumentProcessor(),
        chunker=TextChunker(),
        embedding_service=OpenAIEmbeddingService(
            client=openai_client,
        ),
        vector_store=PgVectorStore(session=session),
    )


def get_research_agent(
    user_id: Annotated[UUID, Depends(get_current_user_id)],
    model: Annotated[BaseChatModel, Depends(get_agent_chat_model)],
    rag_service: Annotated[RAGService, Depends(get_rag_service)],
    arxiv_service: Annotated[
        ArxivService,
        Depends(get_arxiv_service),
    ],
    project_service: Annotated[
        ProjectService,
        Depends(get_project_service),
    ],
) -> ResearchAgent:
    return create_research_agent(
        model=model,
        dependencies=AgentToolDependencies(
            rag_service=rag_service,
            arxiv_service=arxiv_service,
            project_service=project_service,
        ),
        user_id=user_id,
    )


def get_chat_service(
    chat_repository: Annotated[
        ChatRepository,
        Depends(get_chat_repository),
    ],
    message_repository: Annotated[
        MessageRepository,
        Depends(get_message_repository),
    ],
    citation_repository: Annotated[
        CitationRepository,
        Depends(get_citation_repository),
    ],
    project_repository: Annotated[
        ProjectRepository,
        Depends(get_project_repository),
    ],
    research_agent: Annotated[
        ResearchAgent,
        Depends(get_research_agent),
    ],
) -> ChatService:
    return ChatService(
        chat_repository=chat_repository,
        message_repository=message_repository,
        citation_repository=citation_repository,
        project_repository=project_repository,
        research_agent=research_agent,
    )
