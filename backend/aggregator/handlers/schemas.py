"""HTTP request/response schemas.

The response is a reduced view of the model output agreed with the frontend: only the fields
the UI shows, plus session_id to continue the dialog.
"""
from pydantic import BaseModel, ConfigDict, Field, model_serializer

from aggregator.domain import Session


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, description="Free-form product description")


class AnswerRequest(BaseModel):
    value: str = Field(min_length=1, description="One of next_step.options")


class _DropNone(BaseModel):
    """Optional fields are omitted from the output instead of being sent as null."""

    @model_serializer(mode="wrap")
    def _drop_none(self, handler):
        return {k: v for k, v in handler(self).items() if v is not None}


class CorrectionOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    from_: str = Field(alias="from")
    to: str
    reason: str | None = None


class CharacteristicOut(_DropNone):
    name: str
    value: str
    unit: str | None = None
    source: str | None = None


class CandidateOut(BaseModel):
    code: str
    name: str
    score: float | None = None


class ErrorOut(_DropNone):
    type: str
    message: str
    characteristic: str | None = None
    suggestion: str | None = None


class NextStepOut(BaseModel):
    message: str
    characteristic: str | None = None
    options: list[str]


class StepOut(BaseModel):
    stage: str
    count: int


class SessionResponse(BaseModel):
    session_id: str
    corrections: list[CorrectionOut]
    characteristics: list[CharacteristicOut]
    candidates: list[CandidateOut]
    errors: list[ErrorOut]
    next_step: NextStepOut | None
    ktru: str | None
    ktru_name: str | None
    steps: list[StepOut]

    @classmethod
    def from_domain(cls, session: Session) -> "SessionResponse":
        r = session.result
        return cls(
            session_id=session.id,
            corrections=[CorrectionOut(from_=c.from_, to=c.to, reason=c.reason) for c in r.corrections],
            characteristics=[CharacteristicOut(name=c.name, value=c.value, unit=c.unit, source=c.source)
                             for c in r.characteristics],
            candidates=[CandidateOut(code=c.code, name=c.name, score=c.score) for c in r.candidates],
            errors=[ErrorOut(type=e.type, message=e.message, characteristic=e.characteristic,
                             suggestion=e.suggestion) for e in r.errors],
            next_step=NextStepOut(message=r.next_step.message, characteristic=r.next_step.characteristic,
                                  options=r.next_step.options) if r.next_step else None,
            ktru=r.ktru,
            ktru_name=r.ktru_name,
            steps=[StepOut(stage=s.stage, count=s.count) for s in r.steps],
        )


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
