from __future__ import annotations

from xid import XID


def get_xid() -> str:
    return XID().string()
