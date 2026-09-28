import uuid

from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    Integer,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class MessageCitation(Base):
    __tablename__ = "message_citations"

    citation_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    message_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("messages.message_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    chunk_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("paper_chunks.chunk_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    position: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    message: Mapped["Message"] = relationship(
        back_populates="citations",
    )

    chunk: Mapped["PaperChunk"] = relationship(
        back_populates="citations",
    )

    __table_args__ = (
        CheckConstraint(
            "position > 0",
            name="ck_message_citations_position_positive",
        ),
        UniqueConstraint(
            "message_id",
            "position",
            name="uq_message_citation_position",
        ),
        UniqueConstraint(
            "message_id",
            "chunk_id",
            name="uq_message_citation_chunk",
        ),
    )
