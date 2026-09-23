"""add missing paper project count constraint

Revision ID: 9b05f7d7c5a1
Revises: 6550e4154b6f
Create Date: 2026-09-24 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "9b05f7d7c5a1"
down_revision: Union[str, Sequence[str], None] = "6550e4154b6f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_check_constraint(
        "ck_papers_project_count_nonnegative",
        "papers",
        "project_count >= 0",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint(
        "ck_papers_project_count_nonnegative",
        "papers",
        type_="check",
    )
