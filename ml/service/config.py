"""Model service settings read from environment variables."""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


class ConfigError(ValueError):
    """An environment variable has an invalid value; the service must not start."""


@dataclass(frozen=True)
class Settings:
    host: str = "0.0.0.0"
    port: int = 8001
    log_level: str = "INFO"
    # None -> ml.ktru.DEFAULT_PATH (data/processed/ktru.parquet)
    ktru_path: Path | None = None
    # in characters; the aggregator itself allows 2000, so 5000 leaves room for direct calls
    max_text_length: int = 5000
    max_answers: int = 100
    # requests with a larger Content-Length are rejected before the body is read
    max_body_bytes: int = 256 * 1024
    # None -> new characteristics are neither saved nor looked up (ml/service/repository.py);
    # holds the password: kept out of repr and never logged
    postgres_dsn: str | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        level = env.get("LOG_LEVEL", "").strip().upper() or cls.log_level
        if level not in LOG_LEVELS:
            raise ConfigError(f"LOG_LEVEL must be one of {', '.join(LOG_LEVELS)}, got {env['LOG_LEVEL']!r}")
        ktru = env.get("KTRU_PATH", "").strip()
        return cls(
            host=env.get("HOST", "").strip() or cls.host,
            port=_positive_int(env, "PORT", cls.port),
            log_level=level,
            ktru_path=Path(ktru) if ktru else None,
            max_text_length=_positive_int(env, "MAX_TEXT_LENGTH", cls.max_text_length),
            max_answers=_positive_int(env, "MAX_ANSWERS", cls.max_answers),
            max_body_bytes=_positive_int(env, "MAX_BODY_BYTES", cls.max_body_bytes),
            postgres_dsn=env.get("POSTGRES_DSN", "").strip() or None,
        )


def _positive_int(env: Mapping[str, str], key: str, default: int) -> int:
    value = env.get(key, "").strip()
    if not value:
        return default
    try:
        n = int(value)
    except ValueError:
        n = 0
    if n <= 0:
        raise ConfigError(f"{key} must be a positive integer, got {value!r}")
    return n
