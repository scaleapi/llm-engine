from __future__ import annotations

from typing import Generic, Optional, Sequence, TypeVar

from model_engine_server.db.base import Base
from model_engine_server.db.models.common.query import Query
from model_engine_server.db.models.exceptions import EntityNotFoundError
from sqlalchemy import select
from sqlalchemy.orm import Session

T = TypeVar("T", bound="Record")


class Record(Base, Generic[T]):
    """
    Base class for all records. This class provides the basic CRUD operations for records.
    """

    __abstract__ = True

    @staticmethod
    def create(session: Session, record: T):
        session.add(record)
        session.commit()
        # session.refresh(record)
        # TODO: Need better control over the lifecycle of the session
        # so we know how to manage every operation of the session
        return record

    @classmethod
    def select_all(
        cls, session: Session, query: Query, sort_by=None, sort_order="desc"
    ) -> Sequence[T]:
        statement = select(cls).filter_by(**query.to_sqlalchemy_query())
        if sort_by is not None:
            column = getattr(cls, sort_by)
            if sort_order == "desc":
                statement = statement.order_by(column.desc())
            else:
                statement = statement.order_by(column.asc())
        records = session.execute(statement).scalars().all()
        return records

    @classmethod
    def select_one_or_none(cls, session: Session, query: Query) -> Optional[T]:
        statement = select(cls).filter_by(**query.to_sqlalchemy_query())
        record = session.execute(statement).scalar_one_or_none()
        return record

    @classmethod
    def select_by_id(cls, session: Session, record_id: str) -> Optional[T]:
        statement = select(cls).filter_by(id=record_id)
        record = session.execute(statement).scalar_one_or_none()
        return record

    @classmethod
    def update(cls, session: Session, record_id: str, query: Query) -> T:
        record = cls.select_by_id(session=session, record_id=record_id)
        if not record:
            raise EntityNotFoundError(f"Item with id {record_id} not found")
        for key, value in query.to_sqlalchemy_query().items():
            setattr(record, key, value)
        session.commit()
        return record

    @staticmethod
    def delete(session: Session, record: T):
        session.delete(record)
        session.commit()
