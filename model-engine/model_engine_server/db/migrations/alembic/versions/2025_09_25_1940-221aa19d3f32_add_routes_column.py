"""add routes column

Revision ID: 221aa19d3f32
Revises: e580182d6bfd
Create Date: 2025-09-25 19:40:24.927198

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = '221aa19d3f32'
down_revision = 'e580182d6bfd'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'bundles',
        sa.Column('runnable_image_routes', sa.ARRAY(sa.Text), nullable=True),
        schema='hosted_model_inference',
    )


def downgrade() -> None:
    op.drop_column(
        'bundles',
        'runnable_image_routes',
        schema='hosted_model_inference',
    )
