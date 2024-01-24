import asyncio
import os
import sys
from typing import Iterator, Optional

import sqlalchemy
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
from azure.keyvault.secrets import SecretClient
from model_engine_server.core.aws.secrets import get_key_file
from model_engine_server.core.config import infra_config
from model_engine_server.core.loggers import logger_name, make_logger
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import async_scoped_session, async_sessionmaker, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

logger = make_logger(logger_name())


def get_key_file_name(environment: str) -> str:
    if infra_config().cloud_provider == "azure":
        return f"{environment}-ml-infra-pg".replace("training", "prod").replace("-new", "")
    return f"{environment}/ml_infra_pg".replace("training", "prod").replace("-new", "")


def get_engine_url(env: Optional[str] = None, read_only: bool = True, sync: bool = True) -> str:
    """Gets the URL of the Postgresql engine depending on the environment."""
    if os.getenv("ML_INFRA_DATABASE_URL"):
        # In CircleCI environment, we set up a test in another container and specify the URL.
        engine_url = os.getenv("ML_INFRA_DATABASE_URL")
    elif "pytest" in sys.modules:
        # If we are in a local testing environment, we can set up a test psql instance.
        # pylint: disable=import-outside-toplevel
        import testing.postgresql

        Postgresql = testing.postgresql.PostgresqlFactory(
            cache_initialized_db=True,
        )
        postgresql = Postgresql()
        engine_url = postgresql.url()
    else:
        key_file = os.environ.get("DB_SECRET_NAME")
        if env is None:
            env = infra_config().env
        if key_file is None:
            key_file = get_key_file_name(env)  # type: ignore
        logger.info(f"Using key file {key_file}")

        if infra_config().cloud_provider == "azure":
            client = SecretClient(
                vault_url=f"https://{os.environ.get('KEYVAULT_NAME')}.vault.azure.net",
                credential=ManagedIdentityCredential(
                    client_id=os.getenv("AZURE_KEYVAULT_IDENTITY_CLIENT_ID")
                ),  # uses a different managed identity than the default
            )
            db = client.get_secret(key_file).value
            user = os.environ.get("AZURE_KUBERNETES_CLUSTER_IDENTITY_NAME")
            password = (
                DefaultAzureCredential()
                .get_token("https://ossrdbms-aad.database.windows.net")
                .token
            )
            logger.info(f"Connecting to db {db} as user {user}")

            engine_url = f"postgresql://{user}:{password}@{db}"
        else:
            db_secret_aws_profile = os.environ.get("DB_SECRET_AWS_PROFILE")
            creds = get_key_file(key_file, db_secret_aws_profile)

            user = creds.get("username")
            password = creds.get("password")
            host = creds.get("clusterHostRo") if read_only else creds.get("clusterHost")
            port = str(creds.get("port"))
            dbname = creds.get("dbname")
            logger.info(f"Connecting to db {host}:{port}, name {dbname}")

            engine_url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

    assert engine_url

    # For async postgres, we need to use an async dialect.
    if not sync:
        engine_url = engine_url.replace("postgresql://", "postgresql+asyncpg://")
    return engine_url


# Try pool_pre_ping=True, see
# https://docs.sqlalchemy.org/en/14/core/engines.html
#   ?highlight=create_engine#sqlalchemy.create_engine.params.pool_pre_ping
# tl;dr is hopefully it stops the psycopg errors from happening
# Another probably less impactful (ie it shouldn't increase latency by as much,
# but also shouldn't get rid of as many errors e.g. 95% compared to 99.9%)
# option is to try pool_recycle = something kinda short e.g. a minute
# pool_pre_ping=True seems to not increase latency by very much
# (I profiled 2.7 ms -> 3.3 ms on GET model_bundles/)
# but hopefully should completely eliminate
# any of the postgres connection errors we've been seeing.

pg_engine = create_engine(
    get_engine_url(read_only=False, sync=True),
    echo=False,
    future=True,
    pool_pre_ping=True,
    pool_size=20,
    max_overflow=30,
)
pg_engine_read_only = create_engine(
    get_engine_url(read_only=True, sync=True),
    echo=False,
    future=True,
    pool_pre_ping=True,
    pool_size=20,
    max_overflow=30,
)
pg_engine_async = create_async_engine(
    get_engine_url(read_only=False, sync=False),
    echo=False,
    future=True,
    pool_pre_ping=True,
    pool_size=20,
    max_overflow=30,
)
pg_engine_read_only_async = create_async_engine(
    get_engine_url(read_only=True, sync=False),
    echo=False,
    future=True,
    pool_pre_ping=True,
    pool_size=20,
    max_overflow=30,
)
pg_engine_async_null_pool = create_async_engine(
    get_engine_url(read_only=False, sync=False),
    echo=False,
    future=True,
    poolclass=NullPool,
    pool_pre_ping=True,
)

# Synchronous sessions (Session and SessionReadOnly) are fairly straightforward, and both
# can be used at any time. To use asynchronous sqlalchemy, use the SessionAsyncNullPool
# if you're running a synchronous program where concurrency of database connections is not
# super important (e.g. Celery workers that use long-standing connections, and Celery is currently
# synchronous). Use SessionAsync and SessionReadOnlyAsync in ASGI applications.
Session = sessionmaker(autocommit=False, autoflush=False, bind=pg_engine)
SessionReadOnly = sessionmaker(autocommit=False, autoflush=False, bind=pg_engine_read_only)
SessionAsync = async_scoped_session(
    session_factory=async_sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=pg_engine_async,
        expire_on_commit=False,
    ),
    scopefunc=asyncio.current_task,
)
SessionAsyncNullPool = async_scoped_session(
    session_factory=async_sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=pg_engine_async_null_pool,
        expire_on_commit=False,
    ),
    scopefunc=asyncio.current_task,
)
SessionReadOnlyAsync = async_scoped_session(
    async_sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=pg_engine_read_only_async,
        expire_on_commit=False,
    ),
    scopefunc=asyncio.current_task,
)
Base = declarative_base()


def get_session_iterator() -> Iterator[sqlalchemy.orm.Session]:
    """Utility to return an iterator with an instantiated session in the ML Infra database."""
    session = Session()
    try:
        yield session
    finally:
        session.close()


def get_read_only_session_iterator() -> Iterator[sqlalchemy.orm.Session]:
    """Utility to return an iterator with an instantiated session in the ML Infra database."""
    session = SessionReadOnly()
    try:
        yield session
    finally:
        session.close()
