"""Service settings read from environment variables."""
import os
from dataclasses import dataclass, field


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    model_service_url: str = "http://localhost:8001"
    model_timeout_s: float = 30.0
    max_text_length: int = 2000
    redis_url: str = "redis://localhost:6379/0"
    session_ttl_s: int = 3600
    cors_origins: list[str] = field(default_factory=lambda: ["*"])

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model_service_url=os.getenv("MODEL_SERVICE_URL", cls.model_service_url),
            model_timeout_s=float(os.getenv("MODEL_TIMEOUT_S", cls.model_timeout_s)),
            max_text_length=int(os.getenv("MAX_TEXT_LENGTH", cls.max_text_length)),
            redis_url=os.getenv("REDIS_URL", cls.redis_url),
            session_ttl_s=int(os.getenv("SESSION_TTL_S", cls.session_ttl_s)),
            cors_origins=_csv(os.getenv("CORS_ORIGINS", "*")),
        )
