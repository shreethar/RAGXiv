import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes.chats import router as chats_router
from app.api.routes.citations import router as citations_router
from app.api.routes.papers import router as papers_router
from app.api.routes.projects import router as projects_router


logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    application = FastAPI(title="RAGXiv API")
    application.include_router(projects_router)
    application.include_router(papers_router)
    application.include_router(chats_router)
    application.include_router(citations_router)

    @application.exception_handler(Exception)
    async def handle_unexpected_error(
        request: Request,
        error: Exception,
    ) -> JSONResponse:
        logger.exception(
            "Unhandled API error for %s %s",
            request.method,
            request.url.path,
            exc_info=error,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    return application


app = create_app()
