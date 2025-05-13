import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Iterator, Optional

import sqlalchemy
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from google.cloud.secretmanager_v1 import SecretManagerServiceClient
from model_engine_server.core.aws.secrets import get_key_file
from model_engine_server.core.config import InfraConfig, infra_config
from model_engine_server.core.loggers import logger_name, make_logger
from sqlalchemy import Engine, create_engine
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

logger = make_logger(logger_name())


def get_key_file_name(environment: str) -> str:
    # azure and gcp don't support "/" in the key file secret name
    # so we use dashes
    if infra_config().cloud_provider == "azure" or infra_config().cloud_provider == "gcp":
        return f"{environment}-ml-infra-pg".replace("training", "prod").replace("-new", "")

    # aws does support "/" in the key file secret name
    return f"{environment}/ml_infra_pg".replace("training", "prod").replace("-new", "")


@dataclass
class DBConnection:
    url: str
    expiry_in_sec: Optional[int] = None


def get_engine_url(
    env: Optional[str] = None,
    read_only: bool = True,
    sync: bool = True,
) -> DBConnection:
    """Gets the URL of the Postgresql engine depending on the environment."""
    expiry_in_sec: Optional[int] = None
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
        logger.debug(f"Using key file {key_file}")

        if infra_config().cloud_provider == "azure":
            az_secret_client = SecretClient(
                vault_url=f"https://{os.environ.get('KEYVAULT_NAME')}.vault.azure.net",
                credential=DefaultAzureCredential(),
            )
            db = az_secret_client.get_secret(key_file).value
            user = os.environ.get("AZURE_IDENTITY_NAME")
            token = DefaultAzureCredential().get_token(
                "https://ossrdbms-aad.database.windows.net/.default"
            )
            password = token.token
            logger.info(f"Connecting to db {db} as user {user}")

            # TODO: https://docs.sqlalchemy.org/en/20/core/engines.html#generating-dynamic-authentication-tokens
            # for recommendations on how to work with rotating auth credentials
            engine_url = f"postgresql://{user}:{password}@{db}?sslmode=require"
            expiry_in_sec = token.expires_on
        elif infra_config().cloud_provider == "gcp":
            gcp_secret_manager_client = (
                SecretManagerServiceClient()
            )  # uses application default credentials (see: https://cloud.google.com/secret-manager/docs/reference/libraries#client-libraries-usage-python)
            secret_version = gcp_secret_manager_client.access_secret_version(
                request={
                    "name": f"projects/{infra_config().ml_account_id}/secrets/{key_file}/versions/latest"
                }
            )
            creds = json.loads(secret_version.payload.data.decode("utf-8"))

            user = creds.get("username")
            password = creds.get("password")
            host = creds.get("host")
            port = str(creds.get("port"))
            dbname = creds.get("dbname")

            logger.info(f"Connecting to db {host}:{port}, name {dbname}")

            engine_url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
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
        engine_url = engine_url.replace("postgresql://", "postgresql+asyncpg://").replace(
            "sslmode", "ssl"
        )
    return DBConnection(engine_url, expiry_in_sec)


@dataclass
class SyncDBSession:
    engine: Engine
    session: sessionmaker


@dataclass
class AsyncDBSession:
    engine: AsyncEngine
    session: async_sessionmaker


@dataclass
class DBSessions:
    session_sync: SyncDBSession
    session_sync_ro: SyncDBSession
    session_async: AsyncDBSession
    session_async_ro: AsyncDBSession
    session_async_null_pool: AsyncDBSession


@dataclass
class DBEngineConfig:
    pool_pre_ping: bool
    pool_size: int
    max_overflow: int
    echo: bool
    echo_pool: bool


class DBManager:
    sessions: DBSessions
    config: DBEngineConfig

    credential_expiration_timestamp: Optional[float] = None
    credential_expiration_buffer_sec: int = 300

    def _get_engine_url(self, read_only: bool, sync: bool) -> DBConnection:
        return get_engine_url(read_only=read_only, sync=sync)

    def __init__(self, infra_config: InfraConfig):
        self.pool_pre_ping = infra_config.db_engine_disconnect_strategy == "pessimistic"
        self.pool_size = infra_config.db_engine_pool_size
        self.max_overflow = infra_config.db_engine_max_overflow
        self.echo = infra_config.db_engine_echo
        self.echo_pool = infra_config.db_engine_echo_pool
        self.sessions = self.refresh_sessions()

    def refresh_sessions(self) -> DBSessions:
        db_connection = get_engine_url(read_only=False, sync=True)
        # use sync engine as proxy for credential expiration
        self.credential_expiration_timestamp = db_connection.expiry_in_sec
        pg_engine = create_engine(
            db_connection.url,
            echo=self.echo,
            echo_pool=self.echo_pool,
            pool_pre_ping=self.pool_pre_ping,
            pool_size=self.pool_size,
            max_overflow=self.max_overflow,
            future=True,
            logging_name="sync",
        )
        session_sync = SyncDBSession(
            engine=pg_engine,
            session=sessionmaker(autocommit=False, autoflush=False, bind=pg_engine),
        )

        pg_engine_ro = create_engine(
            url=get_engine_url(read_only=True, sync=True).url,
            echo=self.echo,
            echo_pool=self.echo_pool,
            pool_pre_ping=self.pool_pre_ping,
            pool_size=self.pool_size,
            max_overflow=self.max_overflow,
            future=True,
            logging_name="sync_ro",
        )
        session_sync_ro = SyncDBSession(
            engine=pg_engine_ro,
            session=sessionmaker(autocommit=False, autoflush=False, bind=pg_engine_ro),
        )

        pg_engine_async = create_async_engine(
            url=get_engine_url(read_only=False, sync=False).url,
            echo=self.echo,
            echo_pool=self.echo_pool,
            pool_pre_ping=self.pool_pre_ping,
            pool_size=self.pool_size,
            max_overflow=self.max_overflow,
            future=True,
            logging_name="async",
        )
        session_async = AsyncDBSession(
            engine=pg_engine_async,
            session=async_sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=pg_engine_async,
                expire_on_commit=False,
            ),
        )

        pg_engine_async_ro = create_async_engine(
            url=get_engine_url(read_only=True, sync=False).url,
            echo=self.echo,
            echo_pool=self.echo_pool,
            pool_pre_ping=self.pool_pre_ping,
            pool_size=self.pool_size,
            max_overflow=self.max_overflow,
            future=True,
            logging_name="async_ro",
        )
        session_async_ro = AsyncDBSession(
            engine=pg_engine_async_ro,
            session=async_sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=pg_engine_async_ro,
                expire_on_commit=False,
            ),
        )

        pg_engine_async_null_pool = create_async_engine(
            url=get_engine_url(read_only=False, sync=False).url,
            echo=self.echo,
            echo_pool=self.echo_pool,
            future=True,
            poolclass=NullPool,
            logging_name="async_null",
        )

        session_async_null_pool = AsyncDBSession(
            engine=pg_engine_async_null_pool,
            session=async_sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=pg_engine_async_null_pool,
                expire_on_commit=False,
            ),
        )

        return DBSessions(
            session_sync=session_sync,
            session_sync_ro=session_sync_ro,
            session_async=session_async,
            session_async_ro=session_async_ro,
            session_async_null_pool=session_async_null_pool,
        )

    def _is_credentials_expired(self):
        return (
            self.credential_expiration_timestamp is not None
            and time.time()
            > self.credential_expiration_timestamp - self.credential_expiration_buffer_sec
        )

    def _maybe_refresh_sessions(self):
        if self._is_credentials_expired():
            old_sessions = self.sessions
            self.sessions = self.refresh_sessions()
            old_sessions.session_sync.engine.dispose()
            old_sessions.session_sync_ro.engine.dispose()
            old_sessions.session_async.engine.dispose()
            old_sessions.session_async_ro.engine.dispose()
            old_sessions.session_async_null_pool.engine.dispose()

    def get_session_sync(self) -> sessionmaker:
        self._maybe_refresh_sessions()
        return self.sessions.session_sync.session

    def get_session_sync_ro(self) -> sessionmaker:
        self._maybe_refresh_sessions()
        return self.sessions.session_sync_ro.session

    def get_session_async(self) -> async_sessionmaker:
        self._maybe_refresh_sessions()
        return self.sessions.session_async.session

    def get_session_async_ro(self) -> async_sessionmaker:
        self._maybe_refresh_sessions()
        return self.sessions.session_async_ro.session

    def get_session_async_null_pool(self) -> async_sessionmaker:
        self._maybe_refresh_sessions()
        return self.sessions.session_async_null_pool.session


db_manager: Optional[DBManager] = None


def get_db_manager():
    global db_manager
    if db_manager is None:
        db_manager = DBManager(infra_config())
    return db_manager


def get_session():
    return get_db_manager().get_session_sync()


def get_session_read_only():
    return get_db_manager().get_session_sync_ro()


def get_session_async():
    return get_db_manager().get_session_async()


def get_session_async_null_pool():
    return get_db_manager().get_session_async_null_pool()


def get_session_read_only_async():
    return get_db_manager().get_session_async_ro()


Base = declarative_base()


def get_session_iterator() -> Iterator[sqlalchemy.orm.Session]:
    """Utility to return an iterator with an instantiated session in the ML Infra database."""
    Session = get_session()
    session = Session()
    try:
        yield session
    finally:
        session.close()


def get_read_only_session_iterator() -> Iterator[sqlalchemy.orm.Session]:
    """Utility to return an iterator with an instantiated session in the ML Infra database."""
    SessionReadOnly = get_session_read_only()
    session = SessionReadOnly()
    try:
        yield session
    finally:
        session.close()
