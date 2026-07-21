"""Add passthrough forwarder

Revision ID: e580182d6bfd
Revises: f55525c81eb5
Create Date: 2025-09-16 17:41:10.254233

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = 'e580182d6bfd'
down_revision = 'f55525c81eb5'
branch_labels = None
depends_on = None


def _has_column(table: str, column: str, schema: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return any(col["name"] == column for col in inspector.get_columns(table, schema=schema))


def upgrade() -> None:
    if not _has_column("bundles", "runnable_image_forwarder_type", "hosted_model_inference"):
        op.add_column(
            "bundles",
            sa.Column("runnable_image_forwarder_type", sa.String(), nullable=True),
            schema="hosted_model_inference",
        )


def downgrade() -> None:
    if _has_column("bundles", "runnable_image_forwarder_type", "hosted_model_inference"):
        op.drop_column(
            "bundles",
            "runnable_image_forwarder_type",
            schema="hosted_model_inference",
        )