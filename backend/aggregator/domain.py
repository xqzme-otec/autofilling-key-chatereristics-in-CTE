"""Domain entities and errors shared by all layers.

AnalysisResult mirrors the output of ml.pipeline.Pipeline.analyze(). Only the fields the
aggregator relies on are typed; unknown fields are kept (extra="allow") and passed through
to the frontend, so adding fields on the model side does not require changes here.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AnswerType = Literal["product", "characteristic", "code"]


class _Open(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class Correction(_Open):
    from_: str = Field(alias="from")
    to: str
    span: list[int] | None = None
    reason: str | None = None
    alternatives: list[str] | None = None


class Characteristic(_Open):
    name: str
    value: str
    unit: str | None = None
    user_value: str | None = None
    source: str | None = None
    span: list[int] | None = None
    method: str | None = None
    confidence: float | None = None


class Candidate(_Open):
    code: str
    name: str
    score: float | None = None


class AnalysisError(_Open):
    type: str
    message: str
    characteristic: str | None = None
    source: str | None = None
    span: list[int] | None = None
    suggestion: str | None = None


class NextStep(_Open):
    type: AnswerType
    message: str
    characteristic: str | None = None
    options: list[str]
    counts: list[int] | None = None
    required: bool | None = None


class Step(_Open):
    stage: str
    count: int


class AnalysisResult(_Open):
    input: str
    text: str
    corrections: list[Correction] = []
    characteristics: list[Characteristic] = []
    candidates: list[Candidate] = []
    candidates_total: int = 0
    errors: list[AnalysisError] = []
    next_step: NextStep | None = None
    ktru: str | None = None
    ktru_name: str | None = None
    steps: list[Step] = []
    timings_ms: dict[str, float] = {}


class SessionStatus(str, Enum):
    NEEDS_ANSWER = "needs_answer"  # model asks a clarifying question (next_step is set)
    DONE = "done"                  # final KTRU code is determined
    NOT_FOUND = "not_found"        # no code and nothing to ask: product not found in KTRU


@dataclass(frozen=True)
class Answer:
    type: AnswerType
    value: str
    characteristic: str | None = None

    def to_model(self) -> dict:
        out = {"type": self.type, "value": self.value}
        if self.characteristic is not None:
            out["characteristic"] = self.characteristic
        return out


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Session:
    id: str
    raw_text: str
    text: str
    tokens: list[dict] | None
    result: AnalysisResult
    answers: list[Answer] = field(default_factory=list)
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)

    @property
    def status(self) -> SessionStatus:
        if self.result.ktru:
            return SessionStatus.DONE
        if self.result.next_step:
            return SessionStatus.NEEDS_ANSWER
        return SessionStatus.NOT_FOUND


class DomainError(Exception):
    code = "internal_error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class InvalidTextError(DomainError):
    code = "invalid_text"


class SessionNotFoundError(DomainError):
    code = "session_not_found"


class NoPendingQuestionError(DomainError):
    code = "no_pending_question"


class InvalidAnswerError(DomainError):
    code = "invalid_answer"


class ModelUnavailableError(DomainError):
    code = "model_unavailable"


class ModelTimeoutError(DomainError):
    code = "model_timeout"


class ModelBadResponseError(DomainError):
    code = "model_bad_response"
