import uuid

from sqlalchemy import ForeignKey, Uuid, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class ProjectPaper(Base):
    __tablename__ = "project_papers"

    project_paper_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    project_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("projects.project_id",
                   ondelete="CASCADE"),
        nullable=False,
    )

    paper_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("papers.paper_id"),
        nullable=False,
        index=True
    )

    project: Mapped["Project"] = relationship(
        back_populates="project_papers",
    )

    paper: Mapped["Paper"] = relationship(
        back_populates="project_papers",
    )

    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "paper_id",
            name="uq_project_paper",
        ),
    )