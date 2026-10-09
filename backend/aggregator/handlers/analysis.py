"""HTTP handlers for the analysis dialog."""
from typing import Annotated

from fastapi import APIRouter, Depends, Request

from aggregator.handlers.schemas import AnalyzeRequest, AnswerRequest, ErrorResponse, SessionResponse
from aggregator.services import AnalysisService

router = APIRouter(prefix="/api/v1", tags=["analysis"])

ERRORS = {code: {"model": ErrorResponse} for code in (404, 409, 422, 502, 503, 504)}


def get_service(request: Request) -> AnalysisService:
    return request.app.state.analysis_service


Service = Annotated[AnalysisService, Depends(get_service)]


@router.post("/analyze", response_model=SessionResponse, responses=ERRORS)
async def analyze(body: AnalyzeRequest, service: Service) -> SessionResponse:
    """Start a new session from a free-form product description."""
    session = await service.start(body.text)
    return SessionResponse.from_domain(session)


@router.post("/sessions/{session_id}/answers", response_model=SessionResponse, responses=ERRORS)
async def answer(session_id: str, body: AnswerRequest, service: Service) -> SessionResponse:
    """Answer the clarifying question (next_step) of the session with one of its options."""
    session = await service.answer(session_id, body.value)
    return SessionResponse.from_domain(session)
