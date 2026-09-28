"""create paper chunks

Revision ID: d2ef1a7c9b8e
Revises: 9b05f7d7c5a1
Create Date: 2026-09-24 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = "d2ef1a7c9b8e"
down_revision: Union[str, Sequence[str], None] = "9b05f7d7c5a1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "paper_chunks",
        sa.Column("chunk_id", sa.Uuid(), nullable=False),
        sa.Column("paper_id", sa.Uuid(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(3072), nullable=False),
        sa.ForeignKeyConstraint(
            ["paper_id"],
            ["papers.paper_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("chunk_id"),
        sa.UniqueConstraint(
            "paper_id",
            "chunk_index",
            name="uq_paper_chunk_index",
        ),
    )
    op.create_index(
        op.f("ix_paper_chunks_paper_id"),
        "paper_chunks",
        ["paper_id"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        op.f("ix_paper_chunks_paper_id"),
        table_name="paper_chunks",
    )
    op.drop_table("paper_chunks")
