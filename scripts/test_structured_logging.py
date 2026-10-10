import json
import logging
from uuid import UUID

from fastapi.testclient import TestClient

from app.api.dependencies import get_current_user_id, get_project_service
from app.main import create_app
from app.observability.logging import JsonFormatter


def test_json_formatter_emits_structured_context():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="app.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="Operation completed",
        args=(),
        exc_info=None,
    )
    record.event = "test_completed"
    record.paper_id = UUID("00000000-0000-0000-0000-000000000001")

    payload = json.loads(formatter.format(record))

    assert payload["level"] == "INFO"
    assert payload["logger"] == "app.test"
    assert payload["message"] == "Operation completed"
    assert payload["event"] == "test_completed"
    assert payload["paper_id"] == "00000000-0000-0000-0000-000000000001"
    assert "timestamp" in payload


def test_http_middleware_propagates_request_id():
    application = create_app()
    application.dependency_overrides[get_current_user_id] = lambda: UUID(
        "00000000-0000-0000-0000-000000000001"
    )
    project_service = type(
        "ProjectServiceStub",
        (),
        {"list_projects": lambda self, *, user_id: []},
    )()
    application.dependency_overrides[get_project_service] = lambda: (
        project_service
    )

    with TestClient(application) as client:
        response = client.get(
            "/projects",
            headers={"X-Request-ID": "request-123"},
        )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "request-123"
