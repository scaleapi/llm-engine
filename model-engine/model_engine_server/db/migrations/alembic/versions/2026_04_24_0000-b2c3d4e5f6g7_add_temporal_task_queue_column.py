"""add temporal_task_queue column

Revision ID: b2c3d4e5f6g7
Revises: a1b2c3d4e5f6
Create Date: 2026-04-24 00:00:00.000000

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = 'b2c3d4e5f6g7'
down_revision = 'a1b2c3d4e5f6'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'endpoints',
        sa.Column('temporal_task_queue', sa.String, nullable=True),
        schema='hosted_model_inference',
    )


def downgrade() -> None:
    op.drop_column(
        'endpoints',
        'temporal_task_queue',
        schema='hosted_model_inference',
    )
