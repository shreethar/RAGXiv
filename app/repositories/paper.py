from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import delete, select, update
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

    def get_by_content_hash(self, content_hash: str) -> Paper | None:
        stmt = select(Paper).where(Paper.content_hash == content_hash)
        return self.session.scalar(stmt)

    def create(
        self,
        *,
        source_type: str,
        title: str,
        arxiv_id: str | None = None,
        content_hash: str | None = None,
        original_filename: str | None = None,
        abstract: str | None = None,
        authors: str | None = None,
        version: str | None = None,
        published_at: datetime | None = None,
        pdf_url: str | None = None,
        status: str = "pending",
        ingestion_error: str | None = None,
    ) -> Paper:

        paper = Paper(
            source_type=source_type,
            arxiv_id=arxiv_id,
            content_hash=content_hash,
            original_filename=original_filename,
            title=title,
            abstract=abstract,
            authors=authors,
            version=version,
            published_at=published_at,
            pdf_url=pdf_url,
            status=status,
            ingestion_error=ingestion_error,
        )

        self.session.add(paper)
        self.session.flush()

        return paper

    def update_metadata(
        self,
        paper: Paper,
        *,
        title: str,
        abstract: str | None,
        authors: str | None,
        version: str | None,
        published_at: datetime | None,
        pdf_url: str | None,
    ) -> None:
        paper.title = title
        paper.abstract = abstract
        paper.authors = authors
        paper.version = version
        paper.published_at = published_at
        paper.pdf_url = pdf_url
        self.session.flush()

    def set_ingestion_status(
        self,
        paper: Paper,
        *,
        status: str,
        ingestion_error: str | None = None,
    ) -> None:
        paper.status = status
        paper.ingestion_error = ingestion_error
        self.session.flush()

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
