from dataclasses import dataclass
from functools import wraps
from typing import Callable

from model_engine_server.domain.exceptions import ReadOnlyDatabaseException
from sqlalchemy.ext.asyncio import AsyncSession


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
