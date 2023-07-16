from dataclasses import dataclass
from functools import wraps
from typing import Callable

from sqlalchemy.ext.asyncio import AsyncSession

from llm_engine_server.core.domain_exceptions import ReadOnlyDatabaseException


def raise_if_read_only(func: Callable):
    @wraps(func)
    def inner(repo: DbRepositoryMixin, *args, **kwargs):
        if repo.read_only:
            raise ReadOnlyDatabaseException
        return func(repo, *args, **kwargs)

    return inner


@dataclass
class DbRepositoryMixin:
    session: Callable[[], AsyncSession]
    read_only: bool
