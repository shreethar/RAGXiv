from uuid import uuid4
from unittest.mock import Mock

import pytest

from app.db.models.paper import Paper
from app.db.models.project import Project
from app.services.project_service import ProjectService


def make_service():
    project_repository = Mock()
    paper_repository = Mock()

    service = ProjectService(
        project_repository=project_repository,
        paper_repository=paper_repository,
    )

    return service, project_repository, paper_repository


def make_project(*, user_id, project_id=None):
    project = Mock(spec=Project)
    project.project_id = project_id or uuid4()
    project.user_id = user_id
    return project


def make_paper(*, paper_id=None):
    paper = Mock(spec=Paper)
    paper.paper_id = paper_id or uuid4()
    return paper


def test_list_papers_returns_project_papers_for_owner():
    service, project_repository, _ = make_service()

    user_id = uuid4()
    project_id = uuid4()

    project = make_project(
        user_id=user_id,
        project_id=project_id,
    )

    expected_papers = [
        make_paper(),
        make_paper(),
    ]

    project_repository.get_by_id.return_value = project
    project_repository.list_papers.return_value = expected_papers

    result = service.list_papers(
        user_id=user_id,
        project_id=project_id,
    )

    assert result == expected_papers

    project_repository.get_by_id.assert_called_once_with(
        project_id
    )
    project_repository.list_papers.assert_called_once_with(
        project_id
    )


def test_list_papers_raises_when_project_does_not_exist():
    service, project_repository, _ = make_service()

    project_id = uuid4()
    user_id = uuid4()

    project_repository.get_by_id.return_value = None

    with pytest.raises(ValueError, match="Project not found"):
        service.list_papers(
            user_id=user_id,
            project_id=project_id,
        )

    project_repository.list_papers.assert_not_called()


def test_list_papers_raises_when_user_does_not_own_project():
    service, project_repository, _ = make_service()

    owner_id = uuid4()
    different_user_id = uuid4()
    project_id = uuid4()

    project = make_project(
        user_id=owner_id,
        project_id=project_id,
    )

    project_repository.get_by_id.return_value = project

    with pytest.raises(
        ValueError,
        match="Project does not belong to user",
    ):
        service.list_papers(
            user_id=different_user_id,
            project_id=project_id,
        )

    project_repository.list_papers.assert_not_called()


def test_add_papers_adds_all_papers_for_owner():
    service, project_repository, paper_repository = make_service()

    user_id = uuid4()
    project_id = uuid4()
    paper_1_id = uuid4()
    paper_2_id = uuid4()

    project = make_project(
        user_id=user_id,
        project_id=project_id,
    )

    paper_1 = make_paper(paper_id=paper_1_id)
    paper_2 = make_paper(paper_id=paper_2_id)

    project_repository.get_by_id.return_value = project
    paper_repository.get_by_id.side_effect = [
        paper_1,
        paper_2,
    ]

    result = service.add_papers(
        user_id=user_id,
        project_id=project_id,
        paper_ids=[paper_1_id, paper_2_id],
    )

    assert result == [paper_1, paper_2]

    assert paper_repository.get_by_id.call_count == 2

    paper_repository.get_by_id.assert_any_call(paper_1_id)
    paper_repository.get_by_id.assert_any_call(paper_2_id)

    assert project_repository.add_paper.call_count == 2

    project_repository.add_paper.assert_any_call(
        project_id=project_id,
        paper_id=paper_1_id,
    )
    project_repository.add_paper.assert_any_call(
        project_id=project_id,
        paper_id=paper_2_id,
    )


def test_add_papers_raises_when_project_does_not_exist():
    service, project_repository, paper_repository = make_service()

    project_id = uuid4()
    user_id = uuid4()

    project_repository.get_by_id.return_value = None

    with pytest.raises(ValueError, match="Project not found"):
        service.add_papers(
            user_id=user_id,
            project_id=project_id,
            paper_ids=[],
        )

    paper_repository.get_by_id.assert_not_called()
    project_repository.add_paper.assert_not_called()


def test_add_papers_raises_when_user_does_not_own_project():
    service, project_repository, paper_repository = make_service()

    owner_id = uuid4()
    different_user_id = uuid4()
    project_id = uuid4()

    project = make_project(
        user_id=owner_id,
        project_id=project_id,
    )

    project_repository.get_by_id.return_value = project

    with pytest.raises(
        ValueError,
        match="Project does not belong to user",
    ):
        service.add_papers(
            user_id=different_user_id,
            project_id=project_id,
            paper_ids=[],
        )

    paper_repository.get_by_id.assert_not_called()
    project_repository.add_paper.assert_not_called()


def test_add_papers_raises_when_paper_does_not_exist():
    service, project_repository, paper_repository = make_service()

    user_id = uuid4()
    project_id = uuid4()
    paper_id = uuid4()

    project = make_project(
        user_id=user_id,
        project_id=project_id,
    )

    project_repository.get_by_id.return_value = project
    paper_repository.get_by_id.return_value = None

    with pytest.raises(
        ValueError,
        match=f"Paper not found: {paper_id}",
    ):
        service.add_papers(
            user_id=user_id,
            project_id=project_id,
            paper_ids=[paper_id],
        )

    project_repository.add_paper.assert_not_called()


def test_remove_paper_removes_association_for_owner():
    service, project_repository, _ = make_service()

    user_id = uuid4()
    project_id = uuid4()
    paper_id = uuid4()

    project = make_project(
        user_id=user_id,
        project_id=project_id,
    )

    project_repository.get_by_id.return_value = project

    service.remove_paper(
        user_id=user_id,
        project_id=project_id,
        paper_id=paper_id,
    )

    project_repository.remove_paper.assert_called_once_with(
        project_id=project_id,
        paper_id=paper_id,
    )


def test_remove_paper_raises_when_project_does_not_exist():
    service, project_repository, _ = make_service()

    project_id = uuid4()
    paper_id = uuid4()
    user_id = uuid4()

    project_repository.get_by_id.return_value = None

    with pytest.raises(ValueError, match="Project not found"):
        service.remove_paper(
            user_id=user_id,
            project_id=project_id,
            paper_id=paper_id,
        )

    project_repository.remove_paper.assert_not_called()


def test_remove_paper_raises_when_user_does_not_own_project():
    service, project_repository, _ = make_service()

    owner_id = uuid4()
    different_user_id = uuid4()
    project_id = uuid4()
    paper_id = uuid4()

    project = make_project(
        user_id=owner_id,
        project_id=project_id,
    )

    project_repository.get_by_id.return_value = project

    with pytest.raises(
        ValueError,
        match="Project does not belong to user",
    ):
        service.remove_paper(
            user_id=different_user_id,
            project_id=project_id,
            paper_id=paper_id,
        )

    project_repository.remove_paper.assert_not_called()