import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.dependencies import get_session
from app.api.schemas.health import HealthResponse


logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Process liveness check; does not call external dependencies."""
    return HealthResponse(status="healthy")


@router.get("/ready", response_model=HealthResponse)
def readiness(
    session: Annotated[Session, Depends(get_session)],
) -> HealthResponse:
    """Readiness check requiring a working database connection."""
    try:
        session.execute(text("SELECT 1"))
    except SQLAlchemyError as error:
        logger.warning(
            "Readiness check failed",
            extra={"event": "readiness_check_failed"},
            exc_info=error,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is not ready",
        ) from error

    return HealthResponse(status="ready")
