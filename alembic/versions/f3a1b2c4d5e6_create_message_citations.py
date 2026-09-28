"""create message citations

Revision ID: f3a1b2c4d5e6
Revises: d2ef1a7c9b8e
Create Date: 2026-09-29 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "f3a1b2c4d5e6"
down_revision: Union[str, Sequence[str], None] = "d2ef1a7c9b8e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "message_citations",
        sa.Column("citation_id", sa.Uuid(), nullable=False),
        sa.Column("message_id", sa.Uuid(), nullable=False),
        sa.Column("chunk_id", sa.Uuid(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.CheckConstraint(
            "position > 0",
            name="ck_message_citations_position_positive",
        ),
        sa.ForeignKeyConstraint(
            ["chunk_id"],
            ["paper_chunks.chunk_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["message_id"],
            ["messages.message_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("citation_id"),
        sa.UniqueConstraint(
            "message_id",
            "chunk_id",
            name="uq_message_citation_chunk",
        ),
        sa.UniqueConstraint(
            "message_id",
            "position",
            name="uq_message_citation_position",
        ),
    )
    op.create_index(
        op.f("ix_message_citations_chunk_id"),
        "message_citations",
        ["chunk_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_message_citations_message_id"),
        "message_citations",
        ["message_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_message_citations_message_id"),
        table_name="message_citations",
    )
    op.drop_index(
        op.f("ix_message_citations_chunk_id"),
        table_name="message_citations",
    )
    op.drop_table("message_citations")
