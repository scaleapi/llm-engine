# An implementation of a row-based lock.
import asyncio as aio
import hashlib
import time
from contextlib import AbstractContextManager

from llm_engine_server.core.loggers import filename_wo_ext, make_logger
from sqlalchemy import BIGINT, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.session import Session

logger = make_logger(filename_wo_ext(__file__))

BLOCKING_LOCK_TIMEOUT_SECONDS = 120
BLOCKING_LOCK_TIMEOUT_POLL_FREQ_SECONDS = 0.5


def get_lock_key(user_id: str, endpoint_name: str) -> int:
    uid_hash = int.from_bytes(
        hashlib.sha256(bytes(user_id, "utf-8")).digest()[:4],
        byteorder="little",
        signed=False,
    )
    endpoint_name_hash = int.from_bytes(
        hashlib.sha256(bytes(endpoint_name, "utf-8")).digest()[:4],
        byteorder="little",
        signed=False,
    )
    return 2**32 * uid_hash + endpoint_name_hash - 2**63


def try_obtain_lock(session: Session, lock_id: int):
    """
    If able to obtain lock, obtains it and returns True. Otherwise returns False.
    Args:
        session: SQLAlchemy session object
        lock_id: id of lock to take out

    Returns:
        Whether lock was obtained.
    """
    lock_fn = func.pg_try_advisory_lock
    return session.execute(select(lock_fn(cast(lock_id, BIGINT)))).scalar()


def release_lock(session: Session, lock_id: int):
    """
    Releases an obtained lock.
    Args:
        session:
        lock_id:

    Returns:
        nothing
    """
    lock_fn = func.pg_advisory_unlock
    return session.execute(select(lock_fn(cast(lock_id, BIGINT)))).scalar()


async def try_obtain_lock_async(async_session: AsyncSession, lock_id: int):
    """
    If able to obtain lock, obtains it and returns True. Otherwise returns False.
    Args:
        async_session: SQLAlchemy async session object
        lock_id: id of lock to take out

    Returns:
        Whether lock was obtained.
    """
    lock_fn = func.pg_try_advisory_lock
    result = await async_session.execute(select(lock_fn(cast(lock_id, BIGINT))))
    return result.scalar()


async def release_lock_async(async_session: AsyncSession, lock_id: int):
    """
    Releases an obtained lock.
    Args:
        async_session:
        lock_id:

    Returns:
        nothing
    """
    lock_fn = func.pg_advisory_unlock
    result = await async_session.execute(select(lock_fn(cast(lock_id, BIGINT))))
    return result.scalar()


class AdvisoryLockContextManager(AbstractContextManager):
    """
    Implements locking on a given lock_id (which must be between -2 ** 63 and 2 ** 63 - 1)
    Has both blocking and non-blocking modes. Blocking mode polls until acquisition, but times out
    after a few minutes.

    Notes on usage:
    The session passed in should be a short-lived session ideally, as locks get closed out when the
    session ends.
    This reduces the impact of locks failing to release correctly.
    Currently obtained locks can be found via
    ``select * from pg_locks where locktype = 'advisory' limit 10``
    inside of dbeaver
    TODO should we just create the sessions inside of here? They have to be context-managed though
    """

    def __init__(self, session, lock_id: int, blocking: bool = False):
        """

        Args:
            session: A SQLAlchemy session
            lock_id: Lock id, needs to fit in a signed 64 bit int
            blocking: Whether this lock should attempt ot block
        """
        self.session = session
        self.lock_id = lock_id
        self.blocking = blocking
        self._lock_acquired = False

    def __enter__(self):
        if self.blocking:
            start_time = time.time()
            while time.time() - start_time < BLOCKING_LOCK_TIMEOUT_SECONDS:
                lock_acquired = try_obtain_lock(self.session, self.lock_id)
                if lock_acquired:
                    self._lock_acquired = True
                    break
                time.sleep(
                    BLOCKING_LOCK_TIMEOUT_POLL_FREQ_SECONDS
                )  # Should we add a bit of jitter?
        else:
            self._lock_acquired = try_obtain_lock(self.session, self.lock_id)
        logger.debug(
            f"Low-level: acquired lock? {self._lock_acquired}, classid, objid = "
            f"{self.lock_id // 2 ** 32}, {self.lock_id % 2 ** 32}"
        )
        return self

    def __exit__(self, exc_type, value, traceback):
        if self._lock_acquired:
            lock_release = release_lock(self.session, self.lock_id)
            logger.debug(f"Low-level: Releasing lock: {lock_release}")

    async def __aenter__(self):
        if self.blocking:
            start_time = time.time()
            while time.time() - start_time < BLOCKING_LOCK_TIMEOUT_SECONDS:
                lock_acquired = await try_obtain_lock_async(self.session, self.lock_id)
                if lock_acquired:
                    self._lock_acquired = True
                    break
                await aio.sleep(
                    BLOCKING_LOCK_TIMEOUT_POLL_FREQ_SECONDS
                )  # Should we add a bit of jitter?
        else:
            self._lock_acquired = await try_obtain_lock_async(self.session, self.lock_id)
        logger.debug(
            f"Low-level: acquired lock? {self._lock_acquired}, classid, objid = "
            f"{self.lock_id // 2 ** 32}, {self.lock_id % 2 ** 32}"
        )
        return self

    async def __aexit__(self, exc_type, value, traceback):
        if self._lock_acquired:
            lock_release = await release_lock_async(self.session, self.lock_id)
            logger.debug(f"Low-level: Releasing lock: {lock_release}")

    def lock_acquired(self):
        return self._lock_acquired
