"""HTTP handlers for the analysis dialog. Routes are registered in aggregator.main.create_app."""
from typing import Annotated

from fastapi import Depends, Request

from aggregator.handlers.schemas import AnalyzeRequest, AnswerRequest, SessionResponse
from aggregator.services import AnalysisService


def get_service(request: Request) -> AnalysisService:
    return request.app.state.analysis_service


Service = Annotated[AnalysisService, Depends(get_service)]


async def analyze(body: AnalyzeRequest, service: Service) -> SessionResponse:
    """Start a new session from a free-form product description."""
    session = await service.start(body.text)
    return SessionResponse.from_domain(session)


async def answer(session_id: str, body: AnswerRequest, service: Service) -> SessionResponse:
    """Answer the clarifying question (next_step) of the session with one of its options."""
    session = await service.answer(session_id, body.value)
    return SessionResponse.from_domain(session)
