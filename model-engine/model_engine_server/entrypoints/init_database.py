# flake8: noqa
import os
import subprocess
from pathlib import Path

import psycopg2
from model_engine_server.db.base import Base, get_engine_url
from model_engine_server.db.models import *
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine, make_url
from tenacity import Retrying, stop_after_attempt, wait_exponential

SCHEMAS = ["hosted_model_inference", "model"]

MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "db" / "migrations"

# Must match ALEMBIC_TABLE_NAME / version_table_schema in db/migrations/alembic/env.py.
ALEMBIC_VERSION_TABLE = "alembic_version_model_engine"
ALEMBIC_VERSION_TABLE_SCHEMA = "public"


def init_database(database_url: str, psycopg_connection):
    with psycopg_connection.cursor() as cursor:
        for schema in SCHEMAS:
            cursor.execute(f"create schema if not exists {schema}")
    psycopg_connection.commit()
    psycopg_connection.close()
    engine = create_engine(database_url, echo=False, future=True)
    Base.metadata.create_all(engine)


def init_database_and_engine(database_url) -> Engine:
    engine = create_engine(database_url, echo=False, future=True)
    psycopg_connection = psycopg2.connect(
        database=engine.url.database,
        user=engine.url.username,
        password=engine.url.password,
        host=engine.url.host,
        port=engine.url.port,
    )
    # There's a bit of code redundancy here, and this is because of some testing setup weirdness.
    # Can probably revisit this in the future.
    init_database(database_url, psycopg_connection)  # type: ignore
    return engine


def schema_already_initialized(database_url: str) -> bool:
    # Return True if hosted_model_inference.endpoints already exists, i.e. the
    # schema was initialized by a previous run (possibly of an older image whose
    # models lacked columns that current migrations would add). In that case
    # this run's create_all only created missing tables, so we must not claim
    # the database is at head.
    engine = create_engine(database_url, echo=False, future=True)
    try:
        with engine.connect() as connection:
            return inspect(connection).has_table("endpoints", schema="hosted_model_inference")
    finally:
        engine.dispose()


def alembic_is_stamped(database_url: str) -> bool:
    # Return True if the alembic version table already records a revision.
    # In that case we must NOT `alembic stamp head`: stamping moves the version
    # table without running migrations, so a database stamped at an older
    # revision would silently skip every pending (and future) migration.
    engine = create_engine(database_url, echo=False, future=True)
    try:
        with engine.connect() as connection:
            if not inspect(connection).has_table(
                ALEMBIC_VERSION_TABLE, schema=ALEMBIC_VERSION_TABLE_SCHEMA
            ):
                return False
            row = connection.execute(
                text(
                    f"SELECT version_num FROM "  # nosec: identifiers are constants
                    f"{ALEMBIC_VERSION_TABLE_SCHEMA}.{ALEMBIC_VERSION_TABLE} LIMIT 1"
                )
            ).first()
            return row is not None
    finally:
        engine.dispose()


def stamp_alembic_head() -> None:
    # Mark the database as being at the latest alembic revision, so that a
    # database initialized via create_all is not left unstamped (which would
    # cause a subsequent `alembic upgrade head` to replay migrations from the
    # very beginning). Only call this when the database has no alembic revision
    # yet (see alembic_is_stamped). This mirrors run_database_migration.sh,
    # which invokes alembic from the migrations directory; env.py resolves the
    # database URL itself (from ML_INFRA_DATABASE_URL if set, otherwise from
    # cloud secrets), and the subprocess inherits our environment.
    subprocess.run(["alembic", "stamp", "head"], cwd=MIGRATIONS_DIR, check=True)


if __name__ == "__main__":
    url = os.getenv("ML_INFRA_DATABASE_URL")
    # If we are at this point, we want to init the db.
    if url is None:
        print("No k8s secret for DB url found, trying AWS secret")
        url = get_engine_url(read_only=False, sync=True).url
    schema_pre_existed = None
    for attempt in Retrying(
        stop=stop_after_attempt(6),
        wait=wait_exponential(),
        reraise=True,
    ):
        with attempt:
            # Record (once, on the first attempt that can connect) whether the
            # schema existed before this run's create_all, so we only stamp
            # databases we actually initialized from scratch.
            if schema_pre_existed is None:
                schema_pre_existed = schema_already_initialized(url)
            init_database_and_engine(url)

    if alembic_is_stamped(url):
        # An existing revision means migrations manage this database already.
        # Stamping here would fast-forward the version table past unapplied
        # migrations (e.g. new add-column revisions shipped in this image),
        # so leave it alone and let the migration job run `alembic upgrade head`.
        print("Database already has an alembic revision; skipping `alembic stamp head`.")
    elif schema_pre_existed:
        # Unstamped, but the schema predates this run (likely create_all from an
        # older image): its tables may be missing columns that migrations would
        # add, and create_all never alters existing tables. Leave it unstamped so
        # the migration job's initial revision adopts it and the guarded
        # add-column revisions bring it to head.
        print(
            "Schema already existed but has no alembic revision; skipping "
            "`alembic stamp head` so migrations can adopt it."
        )
    else:
        stamp_alembic_head()

    safe_url = make_url(url)
    print(f"Successfully initialized database {safe_url.database} at {safe_url.host}")
