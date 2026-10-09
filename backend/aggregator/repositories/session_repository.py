"""Session storage in Redis.

Each session is one JSON value under `session:<id>`. The TTL is renewed on every read and
write, so inactive sessions expire on their own.
"""
import json
from dataclasses import asdict
from datetime import datetime
from typing import Protocol

from redis.asyncio import Redis
from redis.exceptions import RedisError

from aggregator.domain import AnalysisResult, Answer, Session, StorageUnavailableError

KEY_PREFIX = "session:"


class SessionRepository(Protocol):
    async def get(self, session_id: str) -> Session | None: ...

    async def save(self, session: Session) -> None: ...


class RedisSessionRepository:
    def __init__(self, redis: Redis, ttl_s: int):
        self._redis = redis
        self._ttl_s = ttl_s

    async def get(self, session_id: str) -> Session | None:
        try:
            raw = await self._redis.getex(KEY_PREFIX + session_id, ex=self._ttl_s)
        except RedisError as exc:
            raise StorageUnavailableError(f"Session storage is unavailable: {exc}") from exc
        return None if raw is None else _load(raw)

    async def save(self, session: Session) -> None:
        try:
            await self._redis.set(KEY_PREFIX + session.id, _dump(session), ex=self._ttl_s)
        except RedisError as exc:
            raise StorageUnavailableError(f"Session storage is unavailable: {exc}") from exc


def _dump(session: Session) -> str:
    return json.dumps({
        "id": session.id,
        "raw_text": session.raw_text,
        "result": session.result.model_dump(mode="json", by_alias=True),
        "answers": [asdict(a) for a in session.answers],
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
    }, ensure_ascii=False)


def _load(raw: str | bytes) -> Session:
    data = json.loads(raw)
    return Session(
        id=data["id"],
        raw_text=data["raw_text"],
        result=AnalysisResult.model_validate(data["result"]),
        answers=[Answer(**a) for a in data["answers"]],
        created_at=datetime.fromisoformat(data["created_at"]),
        updated_at=datetime.fromisoformat(data["updated_at"]),
    )
