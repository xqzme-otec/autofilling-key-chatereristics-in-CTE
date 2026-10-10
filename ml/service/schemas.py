"""Request body of POST /analyze (docs/api.md, section 4 — the aggregator payload)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator

AnswerType = Literal["product", "characteristic", "code"]


class Answer(BaseModel):
    """An answer to a previous next_step. Unknown fields are ignored."""
    model_config = ConfigDict(extra="ignore")

    # the pipeline treats a missing type as "characteristic" (ML.md); the aggregator always sends it
    type: AnswerType = "characteristic"
    value: StrictStr
    characteristic: StrictStr | None = None

    @model_validator(mode="after")
    def _characteristic_required(self) -> Answer:
        if self.type == "characteristic" and not self.characteristic:
            raise ValueError("field 'characteristic' is required when type is 'characteristic'")
        return self


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    raw: StrictStr = Field(description="Product description (already preprocessed by the aggregator)")
    # null is accepted as "no answers": a nil Go slice is marshalled as null
    answers: list[Answer] | None = Field(default=None, description="All answers given so far, in order")
    tokens: list[dict[str, Any]] | None = Field(default=None, description="Optional tokens {text, lemma, span, known}")

    def to_pipeline(self) -> dict[str, Any]:
        request: dict[str, Any] = {
            "raw": self.raw,
            "answers": [a.model_dump(exclude_none=True) for a in self.answers or []],
        }
        if self.tokens is not None:
            request["tokens"] = self.tokens
        return request
