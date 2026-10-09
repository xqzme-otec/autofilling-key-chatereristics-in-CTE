"""Session storage.

The in-memory implementation is enough for a single aggregator instance. A database-backed
repository (e.g. PostgreSQL) can replace it later by implementing the same protocol.
"""
import asyncio
from collections import OrderedDict
from typing import Protocol

from aggregator.domain import Session


class SessionRepository(Protocol):
    async def get(self, session_id: str) -> Session | None: ...

    async def save(self, session: Session) -> None: ...


class InMemorySessionRepository:
    """LRU-bounded store: the least recently used session is evicted when the limit is reached."""

    def __init__(self, max_sessions: int):
        self._max = max_sessions
        self._items: OrderedDict[str, Session] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, session_id: str) -> Session | None:
        async with self._lock:
            session = self._items.get(session_id)
            if session is not None:
                self._items.move_to_end(session_id)
            return session

    async def save(self, session: Session) -> None:
        async with self._lock:
            self._items[session.id] = session
            self._items.move_to_end(session.id)
            while len(self._items) > self._max:
                self._items.popitem(last=False)
