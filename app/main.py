import logging
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes.chats import router as chats_router
from app.api.routes.citations import router as citations_router
from app.api.routes.health import router as health_router
from app.api.routes.papers import router as papers_router
from app.api.routes.projects import router as projects_router
from app.observability.logging import configure_logging


logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    configure_logging()
    application = FastAPI(title="RAGXiv API")
    application.include_router(health_router)
    application.include_router(projects_router)
    application.include_router(papers_router)
    application.include_router(chats_router)
    application.include_router(citations_router)

    @application.middleware("http")
    async def log_request(request: Request, call_next):
        supplied_request_id = request.headers.get("X-Request-ID", "").strip()
        request_id = (
            supplied_request_id
            if supplied_request_id and len(supplied_request_id) <= 128
            else str(uuid4())
        )
        request.state.request_id = request_id
        started_at = perf_counter()
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            logger.info(
                "HTTP request completed",
                extra={
                    "event": "http_request_completed",
                    "request_id": request_id,
                    "http_method": request.method,
                    "http_path": request.url.path,
                    "http_status": status_code,
                    "duration_ms": round(
                        (perf_counter() - started_at) * 1000,
                        2,
                    ),
                },
            )

    @application.exception_handler(Exception)
    async def handle_unexpected_error(
        request: Request,
        error: Exception,
    ) -> JSONResponse:
        logger.exception(
            "Unhandled API error",
            extra={
                "event": "http_request_failed",
                "request_id": getattr(
                    request.state,
                    "request_id",
                    None,
                ),
                "http_method": request.method,
                "http_path": request.url.path,
            },
            exc_info=error,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    return application


app = create_app()
