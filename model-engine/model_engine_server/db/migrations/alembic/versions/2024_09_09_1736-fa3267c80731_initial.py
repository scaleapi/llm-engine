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
    with open(INITIAL_MIGRATION_PATH) as fd:
        op.execute(fd.read())


def downgrade() -> None:
    pass
