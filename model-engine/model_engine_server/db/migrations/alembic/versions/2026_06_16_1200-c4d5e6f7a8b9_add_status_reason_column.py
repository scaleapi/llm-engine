"""add status_reason column

Revision ID: c4d5e6f7a8b9
Revises: a1b2c3d4e5f6
Create Date: 2026-06-16 12:00:00.000000

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = 'c4d5e6f7a8b9'
down_revision = 'a1b2c3d4e5f6'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'endpoints',
        sa.Column('status_reason', sa.Text, nullable=True),
        schema='hosted_model_inference',
    )


def downgrade() -> None:
    op.drop_column(
        'endpoints',
        'status_reason',
        schema='hosted_model_inference',
    )
