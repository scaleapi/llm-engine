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


def _has_column(table: str, column: str, schema: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return any(col['name'] == column for col in inspector.get_columns(table, schema=schema))


def upgrade() -> None:
    if not _has_column('endpoints', 'queue_message_timeout_seconds', 'hosted_model_inference'):
        op.add_column(
            'endpoints',
            sa.Column('queue_message_timeout_seconds', sa.Integer, nullable=True),
            schema='hosted_model_inference',
        )


def downgrade() -> None:
    if _has_column('endpoints', 'queue_message_timeout_seconds', 'hosted_model_inference'):
        op.drop_column(
            'endpoints',
            'queue_message_timeout_seconds',
            schema='hosted_model_inference',
        )
