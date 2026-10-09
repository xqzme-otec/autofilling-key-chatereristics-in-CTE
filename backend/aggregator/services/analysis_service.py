"""Analysis dialog: validate text -> ask the model -> keep the session for follow-up answers.

The model service is stateless: every call receives the full text and all answers given so far.
The session keeps them so the frontend only sends the next answer.
"""
import logging
import uuid
from dataclasses import replace
from datetime import datetime, timezone

from aggregator.domain import (
    Answer,
    InvalidAnswerError,
    InvalidTextError,
    NoPendingQuestionError,
    Session,
    SessionNotFoundError,
)
from aggregator.repositories import ModelRepository, SessionRepository

log = logging.getLogger(__name__)


class AnalysisService:
    def __init__(
        self,
        sessions: SessionRepository,
        model: ModelRepository,
        max_text_length: int,
    ):
        self._sessions = sessions
        self._model = model
        self._max_text_length = max_text_length

    async def start(self, raw_text: str) -> Session:
        if not raw_text.strip():
            raise InvalidTextError("Text must not be empty")
        if len(raw_text) > self._max_text_length:
            raise InvalidTextError(f"Text is longer than {self._max_text_length} characters")

        result = await self._model.analyze(raw_text, [])
        session = Session(id=str(uuid.uuid4()), raw_text=raw_text, result=result)
        await self._sessions.save(session)
        log.info("session %s started: status=%s candidates=%d",
                 session.id, session.status.value, result.candidates_total)
        return session

    async def answer(self, session_id: str, value: str) -> Session:
        session = await self._sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(f"Session {session_id} not found")
        answer = self._to_answer(session, value)

        answers = [*session.answers, answer]
        result = await self._model.analyze(session.raw_text, answers)
        updated = replace(session, answers=answers, result=result, updated_at=datetime.now(timezone.utc))
        await self._sessions.save(updated)
        log.info("session %s answered %s: status=%s candidates=%d",
                 session.id, answer.type, updated.status.value, result.candidates_total)
        return updated

    @staticmethod
    def _to_answer(session: Session, value: str) -> Answer:
        """The frontend sends only the chosen option; the question type comes from the last next_step."""
        question = session.result.next_step
        if question is None:
            raise NoPendingQuestionError("There is no pending question in this session")
        if value not in question.options:
            raise InvalidAnswerError(f"Value '{value}' is not one of the offered options")
        characteristic = question.characteristic if question.type == "characteristic" else None
        return Answer(type=question.type, value=value, characteristic=characteristic)
