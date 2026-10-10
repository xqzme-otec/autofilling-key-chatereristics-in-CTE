"""One Pipeline per process: loaded once, used under a lock.

Pipeline.analyze() is NOT thread-safe: it keeps per-call state on shared objects
(ValueMatcher.last_consumed, ValueMatcher._bool_conflict, ...). Calls are therefore serialized with
a lock; requests wait in the threadpool while the event loop stays free (/health keeps answering).
One analyze() takes tens of milliseconds, so the queue drains fast. To scale, run more processes
(each with its own Pipeline), not more threads.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol

logger = logging.getLogger(__name__)

LOADING, READY, FAILED = "loading", "ready", "failed"


class Analyzer(Protocol):
    def analyze(self, request: dict) -> dict: ...


class PipelineNotReady(RuntimeError):
    def __init__(self, status: str):
        super().__init__(status)
        self.status = status


class PipelineHolder:
    def __init__(self, factory: Callable[[], Analyzer]):
        self._factory = factory
        self._pipeline: Analyzer | None = None
        self._lock = threading.Lock()
        self._done = threading.Event()
        self.status = LOADING
        self.error: str | None = None
        self.load_seconds: float | None = None

    def load(self) -> None:
        """Builds the pipeline (model, KTRU index); never raises — a failure is kept in status/error."""
        logger.info("loading pipeline")
        start = time.perf_counter()
        try:
            pipeline = self._factory()
        except Exception as e:
            logger.exception("pipeline failed to load")
            self.error = f"{type(e).__name__}: {e}"[:300]
            self.status = FAILED
        else:
            self._pipeline = pipeline
            self.load_seconds = round(time.perf_counter() - start, 2)
            self.status = READY
            logger.info("pipeline loaded in %.2f s", self.load_seconds)
        finally:
            self._done.set()

    def load_in_background(self) -> threading.Thread:
        thread = threading.Thread(target=self.load, name="pipeline-loader", daemon=True)
        thread.start()
        return thread

    @property
    def pipeline(self) -> Analyzer | None:
        """The loaded pipeline, None until it is ready."""
        return self._pipeline

    def wait(self, timeout: float | None = None) -> bool:
        """True when loading has finished (successfully or not)."""
        return self._done.wait(timeout)

    def analyze(self, request: dict[str, Any]) -> tuple[dict, float]:
        """Runs the pipeline; returns (result, ms spent waiting for the lock)."""
        if self.status != READY or self._pipeline is None:
            raise PipelineNotReady(self.status)
        start = time.perf_counter()
        with self._lock:
            queued_ms = (time.perf_counter() - start) * 1000
            return self._pipeline.analyze(request), queued_ms
