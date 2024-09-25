"""multinode_bundle

Revision ID: f55525c81eb5
Revises: b574e9711e35
Create Date: 2024-09-24 14:56:36.287001

"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ARRAY

# revision identifiers, used by Alembic.
revision = "f55525c81eb5"
down_revision = "b574e9711e35"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "bundles",
        sa.Column("runnable_image_worker_command", ARRAY(sa.Text), nullable=True),
        schema="hosted_model_inference",
    )
    op.add_column(
        "bundles",
        sa.Column("runnable_image_worker_args", sa.JSON, nullable=True),
        schema="hosted_model_inference",
    )


def downgrade() -> None:
    op.drop_column(
        "bundles",
        "runnable_image_worker_command",
        schema="hosted_model_inference",
    )
    op.drop_column(
        "bundles",
        "runnable_image_worker_args",
        schema="hosted_model_inference",
    )
