from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.api import dependencies
from app.api.dependencies import (
    get_arxiv_service,
    get_chat_service,
    get_current_user_id,
    get_paper_service,
    get_project_service,
    get_session,
)
from app.main import create_app


def _project(*, user_id=None, project_id=None, name="Robotics Papers"):
    now = datetime.now(timezone.utc)
    return SimpleNamespace(
        project_id=project_id or uuid4(),
        user_id=user_id or uuid4(),
        name=name,
        created_at=now,
        updated_at=now,
    )


def _paper(*, paper_id=None):
    return SimpleNamespace(
        paper_id=paper_id or uuid4(),
        arxiv_id="1706.03762",
        title="Attention Is All You Need",
        abstract="A sequence transduction architecture.",
        authors="Ashish Vaswani",
        version="v7",
        published_at=datetime(2017, 6, 12, tzinfo=timezone.utc),
        pdf_url="https://arxiv.org/pdf/1706.03762.pdf",
        status="ready",
    )


def _arxiv_paper(*, include_version=False):
    values = {
        "arxiv_id": "1706.03762",
        "title": "Attention Is All You Need",
        "abstract": "A sequence transduction architecture.",
        "authors": ["Ashish Vaswani"],
        "published_at": datetime(2017, 6, 12, tzinfo=timezone.utc),
        "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf",
    }
    if include_version:
        values["version"] = "v7"
    return SimpleNamespace(**values)


def _chat(*, project_id=None, chat_id=None):
    now = datetime.now(timezone.utc)
    return SimpleNamespace(
        chat_id=chat_id or uuid4(),
        project_id=project_id or uuid4(),
        title="Research chat",
        created_at=now,
        updated_at=now,
    )


def _citation():
    paper = _paper()
    chunk = SimpleNamespace(
        chunk_id=uuid4(),
        paper_id=paper.paper_id,
        chunk_index=12,
        content="The cited source text.",
        paper=paper,
    )
    return SimpleNamespace(
        citation_id=uuid4(),
        message_id=uuid4(),
        chunk_id=chunk.chunk_id,
        position=1,
        chunk=chunk,
    )


@pytest.fixture
def api_context():
    app = create_app()
    user_id = uuid4()
    project_service = MagicMock()
    paper_service = MagicMock()
    arxiv_service = MagicMock()
    chat_service = MagicMock()

    app.dependency_overrides[get_current_user_id] = lambda: user_id
    app.dependency_overrides[get_project_service] = lambda: project_service
    app.dependency_overrides[get_paper_service] = lambda: paper_service
    app.dependency_overrides[get_arxiv_service] = lambda: arxiv_service
    app.dependency_overrides[get_chat_service] = lambda: chat_service

    with TestClient(app) as client:
        yield SimpleNamespace(
            app=app,
            client=client,
            user_id=user_id,
            project_service=project_service,
            paper_service=paper_service,
            arxiv_service=arxiv_service,
            chat_service=chat_service,
        )


@pytest.mark.parametrize(
    "headers, expected_detail",
    [
        ({}, "X-User-ID header is required"),
        (
            {"X-User-ID": "not-a-uuid"},
            "X-User-ID header must be a valid UUID",
        ),
    ],
)
def test_development_auth_rejects_missing_or_invalid_user_id(
    headers,
    expected_detail,
):
    app = create_app()
    app.dependency_overrides[get_project_service] = MagicMock

    with TestClient(app) as client:
        response = client.get("/projects", headers=headers)

    assert response.status_code == 401
    assert response.json() == {"detail": expected_detail}


def test_create_and_list_projects_use_authenticated_user(api_context):
    project = _project(user_id=api_context.user_id)
    api_context.project_service.create.return_value = project
    api_context.project_service.list_projects.return_value = [project]

    created = api_context.client.post(
        "/projects",
        json={"name": "  Robotics Papers  "},
    )
    listed = api_context.client.get("/projects")

    assert created.status_code == 201
    assert created.json()["project_id"] == str(project.project_id)
    api_context.project_service.create.assert_called_once_with(
        user_id=api_context.user_id,
        name="Robotics Papers",
    )
    assert listed.status_code == 200
    assert listed.json()["projects"][0]["name"] == "Robotics Papers"
    api_context.project_service.list_projects.assert_called_once_with(
        user_id=api_context.user_id
    )


def test_get_owned_project_and_hide_cross_user_project(api_context):
    project = _project(user_id=api_context.user_id)
    api_context.project_service.get.return_value = project

    response = api_context.client.get(f"/projects/{project.project_id}")

    assert response.status_code == 200
    api_context.project_service.get.assert_called_once_with(
        user_id=api_context.user_id,
        project_id=project.project_id,
    )

    api_context.project_service.get.side_effect = ValueError(
        "Project does not belong to user"
    )
    forbidden = api_context.client.get(f"/projects/{uuid4()}")

    assert forbidden.status_code == 404
    assert forbidden.json() == {"detail": "Resource not found"}


def test_project_paper_operations_delegate_to_project_service(api_context):
    project_id = uuid4()
    paper = _paper()
    api_context.project_service.list_papers.return_value = [paper]
    api_context.project_service.add_papers.return_value = [paper]

    listed = api_context.client.get(f"/projects/{project_id}/papers")
    added = api_context.client.post(
        f"/projects/{project_id}/papers",
        json={"paper_ids": [str(paper.paper_id)]},
    )
    removed = api_context.client.delete(
        f"/projects/{project_id}/papers/{paper.paper_id}"
    )

    assert listed.status_code == 200
    assert listed.json()["papers"][0]["paper_id"] == str(paper.paper_id)
    api_context.project_service.list_papers.assert_called_once_with(
        user_id=api_context.user_id,
        project_id=project_id,
    )
    assert added.status_code == 200
    api_context.project_service.add_papers.assert_called_once_with(
        user_id=api_context.user_id,
        project_id=project_id,
        paper_ids=[paper.paper_id],
    )
    assert removed.status_code == 204
    api_context.project_service.remove_paper.assert_called_once_with(
        user_id=api_context.user_id,
        project_id=project_id,
        paper_id=paper.paper_id,
    )


@pytest.mark.parametrize("method", ["list_papers", "add_papers", "remove_paper"])
def test_cross_user_project_paper_access_is_hidden(api_context, method):
    project_id = uuid4()
    paper_id = uuid4()
    getattr(api_context.project_service, method).side_effect = ValueError(
        "Project does not belong to user"
    )

    if method == "list_papers":
        response = api_context.client.get(
            f"/projects/{project_id}/papers"
        )
    elif method == "add_papers":
        response = api_context.client.post(
            f"/projects/{project_id}/papers",
            json={"paper_ids": [str(paper_id)]},
        )
    else:
        response = api_context.client.delete(
            f"/projects/{project_id}/papers/{paper_id}"
        )

    assert response.status_code == 404
    assert response.json() == {"detail": "Resource not found"}


def test_delete_project_delegates_with_authenticated_user(api_context):
    project_id = uuid4()

    response = api_context.client.delete(f"/projects/{project_id}")

    assert response.status_code == 204
    api_context.project_service.delete.assert_called_once_with(
        user_id=api_context.user_id,
        project_id=project_id,
    )


def test_ingest_paper_and_validate_request(api_context):
    paper = _paper()
    api_context.paper_service.ingest.return_value = paper

    response = api_context.client.post(
        "/papers/ingest",
        json={"arxiv_id": "  1706.03762  "},
    )
    invalid = api_context.client.post(
        "/papers/ingest",
        json={"arxiv_id": "   "},
    )

    assert response.status_code == 200
    assert response.json()["title"] == "Attention Is All You Need"
    api_context.paper_service.ingest.assert_called_once_with(
        arxiv_id="1706.03762"
    )
    assert invalid.status_code == 422


def test_arxiv_search_and_metadata_lookup(api_context):
    search_result = _arxiv_paper()
    metadata = _arxiv_paper(include_version=True)
    api_context.arxiv_service.search.return_value = [search_result]
    api_context.arxiv_service.get_metadata.return_value = metadata

    search = api_context.client.get("/papers/search?q=transformer")
    lookup = api_context.client.get("/papers/arxiv/1706.03762")

    assert search.status_code == 200
    assert search.json()["papers"][0]["arxiv_id"] == "1706.03762"
    api_context.arxiv_service.search.assert_called_once_with(
        query="transformer"
    )
    assert lookup.status_code == 200
    assert lookup.json()["version"] == "v7"
    api_context.arxiv_service.get_metadata.assert_called_once_with(
        arxiv_id="1706.03762"
    )


def test_create_and_get_chat_delegate_to_chat_service(api_context):
    chat = _chat()
    api_context.chat_service.create_chat.return_value = chat
    api_context.chat_service.get_chat.return_value = chat

    created = api_context.client.post(
        "/chats",
        json={
            "project_id": str(chat.project_id),
            "title": "  Research chat  ",
        },
    )
    fetched = api_context.client.get(f"/chats/{chat.chat_id}")

    assert created.status_code == 201
    api_context.chat_service.create_chat.assert_called_once_with(
        user_id=api_context.user_id,
        project_id=chat.project_id,
        title="Research chat",
    )
    assert fetched.status_code == 200
    api_context.chat_service.get_chat.assert_called_once_with(
        user_id=api_context.user_id,
        chat_id=chat.chat_id,
    )


def test_cross_user_chat_and_message_history_are_hidden(api_context):
    chat_id = uuid4()
    api_context.chat_service.get_chat.side_effect = ValueError(
        "Chat does not belong to user"
    )
    api_context.chat_service.list_messages.side_effect = ValueError(
        "Chat does not belong to user"
    )

    chat_response = api_context.client.get(f"/chats/{chat_id}")
    messages_response = api_context.client.get(
        f"/chats/{chat_id}/messages"
    )

    assert chat_response.status_code == 404
    assert messages_response.status_code == 404


def test_get_message_history_returns_chronological_service_result(api_context):
    chat_id = uuid4()
    now = datetime.now(timezone.utc)
    messages = [
        SimpleNamespace(
            message_id=uuid4(),
            role="user",
            content="Question",
            created_at=now,
            citations=[],
        ),
        SimpleNamespace(
            message_id=uuid4(),
            role="assistant",
            content="Answer",
            created_at=now,
            citations=[],
        ),
    ]
    api_context.chat_service.list_messages.return_value = messages

    response = api_context.client.get(f"/chats/{chat_id}/messages")

    assert response.status_code == 200
    assert [item["role"] for item in response.json()["messages"]] == [
        "user",
        "assistant",
    ]
    api_context.chat_service.list_messages.assert_called_once_with(
        user_id=api_context.user_id,
        chat_id=chat_id,
    )


def test_historical_assistant_messages_include_persisted_citations(
    api_context,
):
    chat_id = uuid4()
    citation = _citation()
    api_context.chat_service.list_messages.return_value = [
        SimpleNamespace(
            message_id=citation.message_id,
            role="assistant",
            content="Cited answer",
            created_at=datetime.now(timezone.utc),
            citations=[citation],
        )
    ]

    response = api_context.client.get(f"/chats/{chat_id}/messages")

    assert response.status_code == 200
    item = response.json()["messages"][0]["citations"][0]
    assert item["citation_id"] == str(citation.citation_id)
    assert item["paper"]["arxiv_id"] == "1706.03762"
    assert item["chunk"]["chunk_index"] == 12
    assert "content" not in item["chunk"]


def test_send_message_calls_only_chat_service_and_returns_final_answer(
    api_context,
):
    chat_id = uuid4()
    assistant_message_id = uuid4()
    api_context.chat_service.send_message.return_value = SimpleNamespace(
        chat_id=chat_id,
        user_message_id=uuid4(),
        assistant_message_id=assistant_message_id,
        answer="Self-attention relates each token to other tokens.",
        citations=[],
    )

    response = api_context.client.post(
        f"/chats/{chat_id}/messages",
        json={"content": "  Explain self-attention.  "},
    )

    assert response.status_code == 200
    assert response.json() == {
        "message_id": str(assistant_message_id),
        "chat_id": str(chat_id),
        "role": "assistant",
        "content": "Self-attention relates each token to other tokens.",
        "citations": [],
    }
    api_context.chat_service.send_message.assert_called_once_with(
        user_id=api_context.user_id,
        chat_id=chat_id,
        content="Explain self-attention.",
    )


def test_new_assistant_response_includes_citation_summaries(api_context):
    chat_id = uuid4()
    citation = _citation()
    api_context.chat_service.send_message.return_value = SimpleNamespace(
        chat_id=chat_id,
        assistant_message_id=citation.message_id,
        answer="Cited answer",
        citations=[citation],
    )

    response = api_context.client.post(
        f"/chats/{chat_id}/messages",
        json={"content": "Question"},
    )

    assert response.status_code == 200
    returned = response.json()["citations"][0]
    assert returned["citation_id"] == str(citation.citation_id)
    assert returned["paper"]["title"] == "Attention Is All You Need"
    assert returned["chunk"]["chunk_index"] == 12


def test_citation_inspection_returns_source_content_and_enforces_owner(
    api_context,
):
    citation = _citation()
    api_context.chat_service.get_citation.return_value = citation

    response = api_context.client.get(
        f"/citations/{citation.citation_id}"
    )

    assert response.status_code == 200
    assert response.json()["content"] == "The cited source text."
    api_context.chat_service.get_citation.assert_called_once_with(
        user_id=api_context.user_id,
        citation_id=citation.citation_id,
    )

    api_context.chat_service.get_citation.side_effect = ValueError(
        "Chat does not belong to user"
    )
    forbidden = api_context.client.get(
        f"/citations/{uuid4()}"
    )
    assert forbidden.status_code == 404


def test_send_message_hides_cross_user_chat(api_context):
    api_context.chat_service.send_message.side_effect = ValueError(
        "Chat does not belong to user"
    )

    response = api_context.client.post(
        f"/chats/{uuid4()}/messages",
        json={"content": "Question"},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Resource not found"}


def test_domain_validation_and_unexpected_errors_are_safely_mapped(api_context):
    api_context.paper_service.ingest.side_effect = ValueError(
        "No text could be extracted"
    )
    validation = api_context.client.post(
        "/papers/ingest",
        json={"arxiv_id": "1234.5678"},
    )

    assert validation.status_code == 400
    assert validation.json() == {"detail": "No text could be extracted"}

    api_context.paper_service.ingest.side_effect = RuntimeError(
        "secret provider failure"
    )
    with TestClient(api_context.app, raise_server_exceptions=False) as client:
        unexpected = client.post(
            "/papers/ingest",
            json={"arxiv_id": "1234.5678"},
        )

    assert unexpected.status_code == 500
    assert unexpected.json() == {"detail": "Internal server error"}
    assert "secret provider failure" not in unexpected.text


def test_session_dependency_commits_or_rolls_back_and_always_closes(monkeypatch):
    successful_session = MagicMock()
    monkeypatch.setattr(
        dependencies,
        "SessionLocal",
        lambda: successful_session,
    )
    generator = get_session()
    assert next(generator) is successful_session
    with pytest.raises(StopIteration):
        next(generator)
    successful_session.commit.assert_called_once()
    successful_session.rollback.assert_not_called()
    successful_session.close.assert_called_once()

    failed_session = MagicMock()
    monkeypatch.setattr(
        dependencies,
        "SessionLocal",
        lambda: failed_session,
    )
    generator = get_session()
    assert next(generator) is failed_session
    with pytest.raises(RuntimeError, match="request failed"):
        generator.throw(RuntimeError("request failed"))
    failed_session.commit.assert_not_called()
    failed_session.rollback.assert_called_once()
    failed_session.close.assert_called_once()
