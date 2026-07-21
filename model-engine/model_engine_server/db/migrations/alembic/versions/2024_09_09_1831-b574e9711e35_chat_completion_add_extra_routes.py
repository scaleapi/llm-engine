"""chat completion - Add extra_routes

Revision ID: b574e9711e35
Revises: fa3267c80731
Create Date: 2024-09-09 18:31:59.422082

"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ARRAY

# revision identifiers, used by Alembic.
revision = "b574e9711e35"
down_revision = "fa3267c80731"
branch_labels = None
depends_on = None


def _has_column(table: str, column: str, schema: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return any(col["name"] == column for col in inspector.get_columns(table, schema=schema))


def upgrade() -> None:
    if not _has_column("bundles", "runnable_image_extra_routes", "hosted_model_inference"):
        op.add_column(
            "bundles",
            sa.Column("runnable_image_extra_routes", ARRAY(sa.Text), nullable=True),
            schema="hosted_model_inference",
        )


def downgrade():
    if _has_column("bundles", "runnable_image_extra_routes", "hosted_model_inference"):
        op.drop_column(
            "bundles",
            "runnable_image_extra_routes",
            schema="hosted_model_inference",
        )
