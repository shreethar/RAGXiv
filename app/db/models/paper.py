import uuid
from datetime import datetime

from sqlalchemy import DateTime, String, Text, Uuid, func, Integer, CheckConstraint, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class Paper(Base):
    __tablename__ = "papers"

    paper_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    arxiv_id: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        unique=True,
    )

    title: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    abstract: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    authors: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    version: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
    )

    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    pdf_url: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        server_default="pending",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    project_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )

    last_accessed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )

    __table_args__ = (
        CheckConstraint(
            "project_count >= 0",
            name="ck_papers_project_count_nonnegative",
        ),
    )

    project_papers: Mapped[list["ProjectPaper"]] = relationship(
        back_populates="paper",
    )