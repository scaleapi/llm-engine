# flake8: noqa
import os

import psycopg2
from model_engine_server.db.base import Base, get_engine_url
from model_engine_server.db.models import *
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from tenacity import Retrying, stop_after_attempt, wait_exponential

SCHEMAS = ["hosted_model_inference", "model"]


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


if __name__ == "__main__":
    url = os.getenv("ML_INFRA_DATABASE_URL")
    # If we are at this point, we want to init the db.
    if url is None:
        print("No k8s secret for DB url found, trying AWS secret")
        url = get_engine_url(read_only=False, sync=True).url
    for attempt in Retrying(
        stop=stop_after_attempt(6),
        wait=wait_exponential(),
        reraise=True,
    ):
        with attempt:
            init_database_and_engine(url)

    print(f"Successfully initialized database at {url}")
