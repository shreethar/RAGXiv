from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select, update, delete
from sqlalchemy.orm import Session

from app.db.models.paper import Paper

class PaperRepository:
    def __init__(self, session: Session):
        self.session = session

    def get_by_id(self, paper_id: UUID) -> Paper | None:
        stmt = select(Paper).where(Paper.paper_id == paper_id)
        return self.session.scalar(stmt)

    def get_by_arxiv_id(self, arxiv_id: str) -> Paper | None:
        stmt = select(Paper).where(Paper.arxiv_id == arxiv_id)
        return self.session.scalar(stmt)

    def create(
        self,
        *,
        arxiv_id: str,
        title: str,
        abstract: str | None = None,
        authors: str | None = None,
        version: str | None = None,
        published_at: datetime | None = None,
        pdf_url: str | None = None,
        status: str = "pending"
        ) -> Paper:

        paper = Paper(
            arxiv_id=arxiv_id,
            title=title,
            abstract=abstract,
            authors=authors,
            version=version,
            publised_at=published_at,
            pdf_url=pdf_url,
            status=status
        )

        self.session.add(paper)
        self.session.flush()

        return paper

    def mark_accessed(self, paper_id: UUID) -> bool:
        stmt = (
            update(Paper)
            .where(Paper.paper_id == paper_id)
            .values(last_accessed_at=datetime.now(timezone.utc))
        )
        result = self.session.execute(stmt)

        return result.rowcount == 1

    def list_cleanup_candidates(
        self,
        *,
        accessed_before: datetime
        ) -> list[Paper]:

        stmt = (
            select(Paper)
            .where(
                Paper.last_accessed_at.is_not(None),
                Paper.last_accessed_at < accessed_before,
            )
            .order_by(Paper.last_accessed_at.asc())
        )

        return list(self.session.scalars(stmt))

    def delete(self, paper_id: UUID) -> bool:
        stmt = delete(Paper).where(Paper.paper_id == paper_id)
        result = self.session.execute(stmt)
        return result.rowcount == 1