import asyncio

from aggregator.domain import AnalysisResult, Session
from aggregator.repositories import InMemorySessionRepository


def _session(session_id: str) -> Session:
    return Session(id=session_id, raw_text="t", text="t", tokens=None, result=AnalysisResult(input="t", text="t"))


def test_evicts_least_recently_used():
    async def scenario():
        repo = InMemorySessionRepository(max_sessions=2)
        await repo.save(_session("a"))
        await repo.save(_session("b"))
        await repo.get("a")  # "a" becomes the most recently used
        await repo.save(_session("c"))
        return [await repo.get(i) is not None for i in ("a", "b", "c")]

    assert asyncio.run(scenario()) == [True, False, True]
