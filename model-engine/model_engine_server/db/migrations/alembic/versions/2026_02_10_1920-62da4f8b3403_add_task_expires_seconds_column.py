"""add task_expires_seconds column

Revision ID: 62da4f8b3403
Revises: 221aa19d3f32
Create Date: 2026-02-10 19:20:00.000000

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = '62da4f8b3403'
down_revision = '221aa19d3f32'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'endpoints',
        sa.Column('task_expires_seconds', sa.Integer, nullable=True),
        schema='hosted_model_inference',
    )


def downgrade() -> None:
    op.drop_column(
        'endpoints',
        'task_expires_seconds',
        schema='hosted_model_inference',
    )
