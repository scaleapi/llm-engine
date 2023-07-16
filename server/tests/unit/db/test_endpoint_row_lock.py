# Since the bulk of the file involves actually connecting to postgres, we're only gonna test that the
# `get_lock_key` function doesn't error and returns nonnegative ints from 0 to 2**64-1

from spellbook_serve.db.base import Session
from spellbook_serve.db.endpoint_row_lock import AdvisoryLockContextManager, get_lock_key


def test_get_lock_key():
    pairs = [
        ("userid1", "endpointname1"),
        ("userid2", "endpointname2"),
        ("userid", "1endpointname1"),
        ("endpointname1", "userid1"),
    ] + [(str(i), str(i)) for i in range(10000)]
    keys = [get_lock_key(uid, name) for uid, name in pairs]
    assert len(keys) == len(set(keys)), "Key collision found"
    assert all([-(2**63) <= key < 2**63 for key in keys]), "Key falls outside of range"


def test_lock_context_manager(dbsession: Session):
    with AdvisoryLockContextManager(session=dbsession, lock_id=10) as lock:
        assert lock.lock_acquired()
