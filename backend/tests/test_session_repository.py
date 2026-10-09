import asyncio

import fakeredis
import pytest

from aggregator.domain import AnalysisResult, Answer, Session, StorageUnavailableError
from aggregator.repositories import RedisSessionRepository
from aggregator.repositories.session_repository import KEY_PREFIX
from tests.conftest import model_result

TTL_S = 60


def _session() -> Session:
    return Session(
        id="s1",
        raw_text="стл на металлическом каркасе",
        result=AnalysisResult.model_validate({**model_result("стл на металлическом каркасе", []), "extra": [1]}),
        answers=[Answer(type="characteristic", value="Нет", characteristic="Наличие подлокотников"),
                 Answer(type="product", value="Стул")],
    )


def _repo(server: fakeredis.FakeServer) -> tuple[RedisSessionRepository, fakeredis.FakeAsyncRedis]:
    redis = fakeredis.FakeAsyncRedis(server=server, decode_responses=True)
    return RedisSessionRepository(redis, TTL_S), redis


def test_roundtrip_keeps_all_fields():
    session = _session()

    async def scenario():
        repo, _ = _repo(fakeredis.FakeServer())
        await repo.save(session)
        return await repo.get("s1")

    loaded = asyncio.run(scenario())

    assert loaded == session
    assert loaded.result.corrections[0].from_ == "стл"
    assert loaded.result.model_extra == {"extra": [1]}


def test_missing_session():
    repo, _ = _repo(fakeredis.FakeServer())

    assert asyncio.run(repo.get("nope")) is None


def test_ttl_is_set_and_renewed_on_read():
    async def scenario():
        repo, redis = _repo(fakeredis.FakeServer())
        await repo.save(_session())
        after_save = await redis.ttl(KEY_PREFIX + "s1")
        await redis.expire(KEY_PREFIX + "s1", 5)
        await repo.get("s1")
        return after_save, await redis.ttl(KEY_PREFIX + "s1")

    after_save, after_get = asyncio.run(scenario())

    assert 0 < after_save <= TTL_S
    assert 5 < after_get <= TTL_S


def test_unavailable_redis():
    server = fakeredis.FakeServer()
    server.connected = False
    repo, _ = _repo(server)

    with pytest.raises(StorageUnavailableError):
        asyncio.run(repo.get("s1"))
    with pytest.raises(StorageUnavailableError):
        asyncio.run(repo.save(_session()))
