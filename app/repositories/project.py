from uuid import UUID

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from app.db.models.paper import Paper
from app.db.models.project import Project
from app.db.models.project_paper import ProjectPaper


class ProjectRepository:

    def __init__(self, session: Session):
        self.session = session

    def get_by_id(self, project_id: UUID) -> Project | None:
        stmt = select(Project).where(
            Project.project_id == project_id
        )

        return self.session.scalar(stmt)

    def list_by_user(self, user_id: UUID) -> list[Project]:
        stmt = (
            select(Project)
            .where(Project.user_id == user_id)
            .order_by(Project.created_at.asc())
        )

        return list(self.session.scalars(stmt))

    def create(
        self,
        *,
        user_id: UUID,
        name: str,
    ) -> Project:
        project = Project(
            user_id=user_id,
            name=name,
        )

        self.session.add(project)
        self.session.flush()

        return project

    def update(
        self,
        project_id: UUID,
        *,
        name: str,
    ) -> Project:
        project = self.get_by_id(project_id)

        if project is None:
            raise ValueError(
                f"Project {project_id} not found"
            )

        project.name = name

        self.session.flush()

        return project

    def delete(self, project_id: UUID) -> None:
        project = self.get_by_id(project_id)

        if project is None:
            raise ValueError(
                f"Project {project_id} not found"
            )

        paper_ids_stmt = (
            select(ProjectPaper.paper_id)
            .where(ProjectPaper.project_id == project_id)
        )

        paper_ids = list(self.session.scalars(paper_ids_stmt))

        for paper_id in paper_ids:
            counter_stmt = (
                update(Paper)
                .where(
                    Paper.paper_id == paper_id,
                    Paper.project_count > 0,
                )
                .values(
                    project_count=Paper.project_count - 1
                )
            )

            result = self.session.execute(counter_stmt)

            if result.rowcount != 1:
                raise RuntimeError(
                    f"Could not decrement project_count "
                    f"for paper {paper_id}"
                )

        self.session.delete(project)
        self.session.flush()

    def has_paper(
        self,
        project_id: UUID,
        paper_id: UUID,
    ) -> bool:
        stmt = (
            select(ProjectPaper.project_paper_id)
            .where(
                ProjectPaper.project_id == project_id,
                ProjectPaper.paper_id == paper_id,
            )
        )

        return self.session.scalar(stmt) is not None

    def add_paper(
        self,
        project_id: UUID,
        paper_id: UUID,
    ) -> bool:
        project = self.get_by_id(project_id)

        if project is None:
            raise ValueError(
                f"Project {project_id} not found"
            )

        paper_stmt = select(Paper).where(
            Paper.paper_id == paper_id
        )

        paper = self.session.scalar(paper_stmt)

        if paper is None:
            raise ValueError(
                f"Paper {paper_id} not found"
            )

        if self.has_paper(project_id, paper_id):
            return False

        project_paper = ProjectPaper(
            project_id=project_id,
            paper_id=paper_id,
        )

        self.session.add(project_paper)

        counter_stmt = (
            update(Paper)
            .where(Paper.paper_id == paper_id)
            .values(
                project_count=Paper.project_count + 1
            )
        )

        self.session.execute(counter_stmt)

        self.session.flush()

        return True

    def remove_paper(
        self,
        project_id: UUID,
        paper_id: UUID,
    ) -> bool:


        stmt = (
            delete(ProjectPaper)
            .where(
                ProjectPaper.project_id == project_id,
                ProjectPaper.paper_id == paper_id,
            )
        )

        result = self.session.execute(stmt)

        if result.rowcount != 1:
            return False

        counter_stmt = (
            update(Paper)
            .where(
                Paper.paper_id == paper_id,
                Paper.project_count > 0,
            )
            .values(
                project_count=Paper.project_count - 1
            )
        )

        counter_result = self.session.execute(counter_stmt)

        if counter_result.rowcount != 1:
            raise RuntimeError(
                f"Could not decrement project_count "
                f"for paper {paper_id}"
            )

        self.session.flush()

        return True

    def list_papers(
        self,
        project_id: UUID,
    ) -> list[Paper]:
        stmt = (
            select(Paper)
            .join(
                ProjectPaper,
                ProjectPaper.paper_id == Paper.paper_id,
            )
            .where(
                ProjectPaper.project_id == project_id
            )
            .order_by(Paper.title.asc())
        )

        return list(self.session.scalars(stmt))