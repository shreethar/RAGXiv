from uuid import UUID

from app.db.models.paper import Paper
from app.db.models.project import Project
from app.repositories.paper import PaperRepository
from app.repositories.project import ProjectRepository


class ProjectService:
    def __init__(
        self,
        *,
        project_repository: ProjectRepository,
        paper_repository: PaperRepository,
    ):
        self.project_repository = project_repository
        self.paper_repository = paper_repository

    def create(
        self,
        *,
        user_id: UUID,
        name: str,
    ) -> Project:
        normalized_name = name.strip()

        if not normalized_name:
            raise ValueError("Project name must not be empty")

        return self.project_repository.create(
            user_id=user_id,
            name=normalized_name,
        )

    def get(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
    ) -> Project:
        return self._get_owned_project(
            user_id=user_id,
            project_id=project_id,
        )

    def list_projects(self, *, user_id: UUID) -> list[Project]:
        return self.project_repository.list_by_user(user_id)

    def delete(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
    ) -> None:
        self._get_owned_project(
            user_id=user_id,
            project_id=project_id,
        )
        self.project_repository.delete(project_id)

    def list_papers(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
    ) -> list[Paper]:
        self._get_owned_project(
            user_id=user_id,
            project_id=project_id,
        )

        return self.project_repository.list_papers(project_id)

    def add_papers(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
        paper_ids: list[UUID],
    ) -> list[Paper]:
        self._get_owned_project(
            user_id=user_id,
            project_id=project_id,
        )

        papers: list[Paper] = []

        for paper_id in paper_ids:
            paper = self.paper_repository.get_by_id(paper_id)

            if paper is None:
                raise ValueError(
                    f"Paper not found: {paper_id}"
                )

            papers.append(paper)

        for paper_id in paper_ids:
            self.project_repository.add_paper(
                project_id=project_id,
                paper_id=paper_id,
            )

        return papers

    def remove_paper(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
        paper_id: UUID,
    ) -> None:
        self._get_owned_project(
            user_id=user_id,
            project_id=project_id,
        )

        self.project_repository.remove_paper(
            project_id=project_id,
            paper_id=paper_id,
        )

    def _get_owned_project(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
    ) -> Project:
        project = self.project_repository.get_by_id(project_id)

        if project is None:
            raise ValueError("Project not found")

        if project.user_id != user_id:
            raise ValueError("Project does not belong to user")

        return project
