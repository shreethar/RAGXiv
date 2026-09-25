from uuid import UUID

from app.db.models.paper import Paper
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

    def list_papers(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
    ) -> list[Paper]:
        project = self.project_repository.get_by_id(project_id)

        if project is None:
            raise ValueError("Project not found")

        if project.user_id != user_id:
            raise ValueError("Project does not belong to user")

        return self.project_repository.list_papers(project_id)

    def add_papers(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
        paper_ids: list[UUID],
    ) -> list[Paper]:
        project = self.project_repository.get_by_id(project_id)

        if project is None:
            raise ValueError("Project not found")

        if project.user_id != user_id:
            raise ValueError("Project does not belong to user")

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
        project = self.project_repository.get_by_id(project_id)

        if project is None:
            raise ValueError("Project not found")

        if project.user_id != user_id:
            raise ValueError("Project does not belong to user")

        self.project_repository.remove_paper(
            project_id=project_id,
            paper_id=paper_id,
        )