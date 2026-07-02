"""“initial”

Revision ID: fa3267c80731
Revises: 
Create Date: 2024-09-09 17:36:30.097136

"""

from pathlib import Path

INITIAL_MIGRATION_PATH = Path(__file__).parent / "../../initial.sql"


import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "fa3267c80731"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # If the schema already exists (e.g. the database was initialized by
    # init_database's create_all before alembic was stamped), adopt the
    # existing schema instead of replaying initial.sql, whose bare
    # CREATE SCHEMA statements would fail with DuplicateSchema.
    inspector = sa.inspect(op.get_bind())
    if inspector.has_table("endpoints", schema="hosted_model_inference"):
        print(
            "Table hosted_model_inference.endpoints already exists; "
            "adopting existing schema and skipping initial.sql."
        )
        return
    with open(INITIAL_MIGRATION_PATH) as fd:
        op.execute(fd.read())


def downgrade() -> None:
    pass
