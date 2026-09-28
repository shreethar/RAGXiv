from collections.abc import Callable
from typing import TypeVar

from fastapi import HTTPException, status


T = TypeVar("T")


def call_application(callback: Callable[[], T]) -> T:
    """Translate established service validation errors to HTTP errors."""

    try:
        return callback()
    except ValueError as error:
        message = str(error)
        normalized = message.lower()

        if "not found" in normalized or "does not belong" in normalized:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Resource not found",
            ) from error

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message,
        ) from error
