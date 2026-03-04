"""add queue_message_timeout_seconds column

Revision ID: a1b2c3d4e5f6
Revises: 62da4f8b3403
Create Date: 2026-02-20 12:00:00.000000

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = 'a1b2c3d4e5f6'
down_revision = '62da4f8b3403'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'endpoints',
        sa.Column('queue_message_timeout_seconds', sa.Integer, nullable=True),
        schema='hosted_model_inference',
    )


def downgrade() -> None:
    op.drop_column(
        'endpoints',
        'queue_message_timeout_seconds',
        schema='hosted_model_inference',
    )
