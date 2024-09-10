import logging
import os
from logging.config import fileConfig

from alembic import context
from model_engine_server.db.base import get_engine_url
from sqlalchemy import engine_from_config, pool

env = os.environ.get("ENV")
if env is None:
    assert (
        os.getenv("ML_INFRA_DATABASE_URL") is not None
    ), "Expected ML_INFRA_DATABASE_URL to be set if ENV is not set."

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

config.set_main_option("sqlalchemy.url", get_engine_url(env, read_only=False).url)

ALEMBIC_TABLE_NAME = "alembic_version_model_engine"

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
target_metadata = None

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table=ALEMBIC_TABLE_NAME,
    )

    try:
        with context.begin_transaction():
            context.run_migrations()
    except Exception as e:
        logging.exception("Error during migration: %s", str(e))
        raise e


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            version_table=ALEMBIC_TABLE_NAME,
        )

        try:
            with context.begin_transaction():
                context.run_migrations()
        except Exception as e:
            logging.exception("Error during migration: %s", str(e))
            raise e
        finally:
            connection.close()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
