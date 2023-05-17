from dataclasses import dataclass

from llm_engine_server.db.models.common.query import Query


@dataclass
class ExampleQuery(Query):
    """
    Example query
    """

    id: str
    name: str


def test_query():
    query = ExampleQuery(id="123", name="test")
    assert query.to_sqlalchemy_query() == {"id": "123", "name": "test"}
