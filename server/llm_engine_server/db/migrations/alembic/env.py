import logging
import os
from logging.config import fileConfig

from alembic import context
from llm_engine_server.db.base import get_engine_url
from sqlalchemy import engine_from_config, pool

env = os.environ.get("ENV")
assert env is not None, "Expected ENV to be a nonempty environment variable."

config = context.config

config.set_main_option("sqlalchemy.url", get_engine_url(env, read_only=False))

# Interpret the config file for Python logging.
# This line sets up loggers basically.
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


def run_migrations_offline():
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
        include_schemas=True,
    )

    try:
        with context.begin_transaction():
            context.run_migrations()
    except Exception as e:
        logging.exception("Error during migration: %s", str(e))
        raise e


def run_migrations_online():
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
        context.configure(connection=connection, target_metadata=target_metadata)

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
