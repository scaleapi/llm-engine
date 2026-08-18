import asyncio
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Union, cast

import sqlalchemy
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
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
    if infra_config().cloud_provider == "azure":
        return f"{environment}-ml-infra-pg".replace("training", "prod").replace("-new", "")
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

        if infra_config().cloud_provider == "onprem":
            user = os.environ.get("DB_USER", "postgres")
            password = os.environ.get("DB_PASSWORD", "postgres")
            host = os.environ.get("DB_HOST_RO") or os.environ.get("DB_HOST", "localhost")
            port = os.environ.get("DB_PORT", "5432")
            dbname = os.environ.get("DB_NAME", "llm_engine")
            logger.info(f"Connecting to db {host}:{port}, name {dbname}")

            engine_url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

        elif infra_config().cloud_provider == "gcp":
            db_secret_name = os.environ.get("DB_SECRET_NAME")
            if db_secret_name:
                from model_engine_server.core.gcp.secrets import get_key_file as get_gcp_key_file

                db_secret_gcp_project_id = os.environ.get("DB_SECRET_GCP_PROJECT_ID")
                creds = get_gcp_key_file(db_secret_name, db_secret_gcp_project_id)
                user = creds.get("username")
                password = creds.get("password")
                if read_only:
                    host = creds.get("clusterHostRo") or creds.get("host")
                else:
                    host = creds.get("clusterHost") or creds.get("host")
                port = str(creds.get("port"))
                dbname = creds.get("dbname")
            else:
                user = os.environ.get("DB_USER", "postgres")
                password = os.environ.get("DB_PASSWORD", "postgres")
                host = os.environ.get("DB_HOST_RO") or os.environ.get("DB_HOST", "localhost")
                port = os.environ.get("DB_PORT", "5432")
                dbname = os.environ.get("DB_NAME", "llm_engine")
            logger.info(f"Connecting to db {host}:{port}, name {dbname}")

            engine_url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

        elif infra_config().cloud_provider == "azure":
            client = SecretClient(
                vault_url=f"https://{os.environ.get('KEYVAULT_NAME')}.vault.azure.net",
                credential=DefaultAzureCredential(),
            )
            db = client.get_secret(key_file).value
            user = os.environ.get("AZURE_IDENTITY_NAME", "")
            token = DefaultAzureCredential().get_token(
                "https://ossrdbms-aad.database.windows.net/.default"
            )
            password = token.token
            logger.info(f"Connecting to db {db} as user {user}")

            # TODO: https://docs.sqlalchemy.org/en/20/core/engines.html#generating-dynamic-authentication-tokens
            # for recommendations on how to work with rotating auth credentials
            engine_url = f"postgresql://{user}:{password}@{db}?sslmode=require"
            expiry_in_sec = token.expires_on
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
class DBEngineConfig:
    pool_pre_ping: bool
    pool_size: int
    max_overflow: int
    echo: bool
    echo_pool: bool


class DBManager:
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
        # Engines are built lazily, one per session kind on first use. Every worker
        # process gets its own DBManager, so eagerly building all five engines
        # multiplies idle connection pools by processes x pods even for engines the
        # process never uses (the API gateway only ever touches the async pair).
        self._sessions: Dict[str, Union[SyncDBSession, AsyncDBSession]] = {}
        # Sessions are fetched from both the event loop and threadpool threads; the
        # lock keeps a cold kind from being built (and its loser leaked) twice.
        self._build_lock = threading.Lock()

    def _pooled_engine_kwargs(self) -> Dict[str, Any]:
        return dict(
            echo=self.echo,
            echo_pool=self.echo_pool,
            pool_pre_ping=self.pool_pre_ping,
            pool_size=self.pool_size,
            max_overflow=self.max_overflow,
            future=True,
        )

    def _build_session(self, kind: str) -> Union[SyncDBSession, AsyncDBSession]:
        built: Union[SyncDBSession, AsyncDBSession]
        if kind in ("sync", "sync_ro"):
            db_connection = self._get_engine_url(read_only=kind == "sync_ro", sync=True)
            engine = create_engine(
                url=db_connection.url,
                logging_name=kind,
                **self._pooled_engine_kwargs(),
            )
            built = SyncDBSession(
                engine=engine,
                session=sessionmaker(autocommit=False, autoflush=False, bind=engine),
            )
        elif kind in ("async", "async_ro"):
            db_connection = self._get_engine_url(read_only=kind == "async_ro", sync=False)
            async_engine = create_async_engine(
                url=db_connection.url,
                logging_name=kind,
                **self._pooled_engine_kwargs(),
            )
            built = AsyncDBSession(
                engine=async_engine,
                session=async_sessionmaker(
                    autocommit=False,
                    autoflush=False,
                    bind=async_engine,
                    expire_on_commit=False,
                ),
            )
        else:
            assert kind == "async_null_pool", f"Unknown DB session kind: {kind}"
            db_connection = self._get_engine_url(read_only=False, sync=False)
            null_pool_engine = create_async_engine(
                url=db_connection.url,
                echo=self.echo,
                echo_pool=self.echo_pool,
                future=True,
                poolclass=NullPool,
                logging_name="async_null",
            )
            built = AsyncDBSession(
                engine=null_pool_engine,
                session=async_sessionmaker(
                    autocommit=False,
                    autoflush=False,
                    bind=null_pool_engine,
                    expire_on_commit=False,
                ),
            )
        if self.credential_expiration_timestamp is None:
            # use the first built engine's credentials as proxy for expiration
            self.credential_expiration_timestamp = db_connection.expiry_in_sec
        return built

    def _is_credentials_expired(self):
        return (
            self.credential_expiration_timestamp is not None
            and time.time()
            > self.credential_expiration_timestamp - self.credential_expiration_buffer_sec
        )

    def _take_expired_sessions(self) -> list[Union[SyncDBSession, AsyncDBSession]]:
        if not self._is_credentials_expired():
            return []
        old_sessions = list(self._sessions.values())
        self._sessions = {}
        self.credential_expiration_timestamp = None
        return old_sessions

    def _get_session(self, kind: str) -> Union[SyncDBSession, AsyncDBSession]:
        with self._build_lock:
            old_sessions = self._take_expired_sessions()
        for old_session in old_sessions:
            if isinstance(old_session, AsyncDBSession):
                try:
                    asyncio.get_running_loop()
                except RuntimeError:
                    asyncio.run(old_session.engine.dispose())
                else:
                    # Sync getter called on a thread that already runs an event loop:
                    # asyncio.run() would raise. Dispose the underlying pool directly.
                    old_session.engine.sync_engine.dispose()
            else:
                old_session.engine.dispose()
        with self._build_lock:
            if kind not in self._sessions:
                self._sessions[kind] = self._build_session(kind)
            return self._sessions[kind]

    async def _get_async_session(self, kind: str) -> AsyncDBSession:
        with self._build_lock:
            old_sessions = self._take_expired_sessions()
        for old_session in old_sessions:
            if isinstance(old_session, AsyncDBSession):
                await old_session.engine.dispose()
            else:
                old_session.engine.dispose()
        with self._build_lock:
            if kind not in self._sessions:
                self._sessions[kind] = self._build_session(kind)
            return cast(AsyncDBSession, self._sessions[kind])

    def get_session_sync(self) -> sessionmaker:
        return cast(SyncDBSession, self._get_session("sync")).session

    def get_session_sync_ro(self) -> sessionmaker:
        return cast(SyncDBSession, self._get_session("sync_ro")).session

    async def get_session_async(self) -> async_sessionmaker:
        return (await self._get_async_session("async")).session

    async def get_session_async_ro(self) -> async_sessionmaker:
        return (await self._get_async_session("async_ro")).session

    async def get_session_async_null_pool(self) -> async_sessionmaker:
        return (await self._get_async_session("async_null_pool")).session


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


async def get_session_async():
    return await get_db_manager().get_session_async()


async def get_session_async_null_pool():
    return await get_db_manager().get_session_async_null_pool()


async def get_session_read_only_async():
    return await get_db_manager().get_session_async_ro()


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
