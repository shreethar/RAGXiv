"""add uploads and ingestion states

Revision ID: a7c9e2f4b6d8
Revises: f3a1b2c4d5e6
Create Date: 2026-10-10 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "a7c9e2f4b6d8"
down_revision: Union[str, Sequence[str], None] = "f3a1b2c4d5e6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "papers",
        sa.Column(
            "source_type",
            sa.String(length=20),
            server_default="arxiv",
            nullable=False,
        ),
    )
    op.add_column(
        "papers",
        sa.Column("content_hash", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "papers",
        sa.Column("original_filename", sa.Text(), nullable=True),
    )
    op.add_column(
        "papers",
        sa.Column("ingestion_error", sa.Text(), nullable=True),
    )
    op.alter_column(
        "papers",
        "arxiv_id",
        existing_type=sa.String(length=50),
        nullable=True,
    )
    op.create_unique_constraint(
        "uq_papers_content_hash",
        "papers",
        ["content_hash"],
    )
    op.create_check_constraint(
        "ck_papers_ingestion_status",
        "papers",
        "status IN ('pending', 'processing', 'ready', 'failed')",
    )
    op.create_check_constraint(
        "ck_papers_source_identity",
        "papers",
        "(source_type = 'arxiv' AND arxiv_id IS NOT NULL "
        "AND content_hash IS NULL) OR "
        "(source_type = 'upload' AND arxiv_id IS NULL "
        "AND content_hash IS NOT NULL)",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_papers_source_identity",
        "papers",
        type_="check",
    )
    op.drop_constraint(
        "ck_papers_ingestion_status",
        "papers",
        type_="check",
    )
    op.drop_constraint(
        "uq_papers_content_hash",
        "papers",
        type_="unique",
    )
    op.execute(
        "DELETE FROM project_papers WHERE paper_id IN "
        "(SELECT paper_id FROM papers WHERE arxiv_id IS NULL)"
    )
    op.execute("DELETE FROM papers WHERE arxiv_id IS NULL")
    op.alter_column(
        "papers",
        "arxiv_id",
        existing_type=sa.String(length=50),
        nullable=False,
    )
    op.drop_column("papers", "ingestion_error")
    op.drop_column("papers", "original_filename")
    op.drop_column("papers", "content_hash")
    op.drop_column("papers", "source_type")
