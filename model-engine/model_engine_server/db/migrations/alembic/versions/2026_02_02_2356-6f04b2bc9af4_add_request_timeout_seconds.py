"""add request timeout seconds column

Revision ID: 6f04b2bc9af4
Revises: 221aa19d3f32
Create Date: 2026-02-02 23:56:00.000000

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = '6f04b2bc9af4'
down_revision = '221aa19d3f32'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'bundles',
        sa.Column('runnable_image_request_timeout_seconds', sa.Integer(), nullable=True),
        schema='hosted_model_inference',
    )


def downgrade() -> None:
    op.drop_column(
        'bundles',
        'runnable_image_request_timeout_seconds',
        schema='hosted_model_inference',
    )

