"""Logging for the model service (AGENTS.md, section «Логирование»).

Every log record gets a `request_id` attribute taken from a context variable, so log lines written
anywhere while a request is processed (including the pipeline running in a worker thread) carry
the id of that request. Outside of a request it is "-".
"""
from __future__ import annotations

import logging
import sys
from contextvars import ContextVar

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s"
NO_REQUEST = "-"

request_id_var: ContextVar[str] = ContextVar("request_id", default=NO_REQUEST)

_HANDLER_MARK = "_ml_service_handler"
_factory_installed = False


def _install_record_factory() -> None:
    """Adds request_id to every LogRecord, whatever handler formats it."""
    global _factory_installed
    if _factory_installed:
        return
    previous = logging.getLogRecordFactory()

    def factory(*args, **kwargs):
        record = previous(*args, **kwargs)
        if not hasattr(record, "request_id"):
            record.request_id = request_id_var.get()
        return record

    logging.setLogRecordFactory(factory)
    _factory_installed = True


def setup_logging(level: str = "INFO") -> None:
    """Root logger -> stdout with the common format. Idempotent: repeated calls replace our handler."""
    _install_record_factory()
    root = logging.getLogger()
    for h in [h for h in root.handlers if getattr(h, _HANDLER_MARK, False)]:
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    setattr(handler, _HANDLER_MARK, True)
    root.addHandler(handler)
    root.setLevel(level)
    # third-party libraries are noisy at DEBUG and do not help to trace a request
    lib_level = max(logging.getLevelName(level), logging.INFO)
    for name in ("urllib3", "httpx", "httpcore", "filelock", "huggingface_hub", "transformers", "asyncio",
                 "multipart", "python_multipart"):
        logging.getLogger(name).setLevel(lib_level)
    # SentenceTransformer.encode() draws a tqdm progress bar on every call when its logger is at INFO/DEBUG
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
