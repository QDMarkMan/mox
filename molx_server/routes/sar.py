"""SAR analysis API endpoints."""

import asyncio
import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from molx_agent.agents import run_sar_agent
from molx_server.schemas.models import (
    AgentErrorResponse,
    AgentStatus,
    SARAnalysisRequest,
    SARAnalysisResponse,
)
from molx_server.session import get_session_manager
from molx_server.streaming import stream_agent_response


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/sar", tags=["SAR Analysis"])


@router.post(
    "/analyze",
    response_model=SARAnalysisResponse,
    responses={500: {"model": AgentErrorResponse}},
)
async def analyze_sar(request: SARAnalysisRequest) -> SARAnalysisResponse:
    """
    Execute SAR analysis.

    Performs comprehensive SAR analysis including:
    - Scaffold identification
    - R-group decomposition
    - Activity cliff detection
    - Structure optimization suggestions
    """
    start_time = time.time()
    session_manager = get_session_manager()

    try:
        # Get or create session
        session_data = await session_manager.get_or_create_session_async(request.session_id)

        # Run SAR analysis
        report_state = await asyncio.to_thread(run_sar_agent, request.query)
        report = report_state.get("final_response", "")
        structured = report_state.get("results", {})

        # Build metadata
        elapsed = time.time() - start_time
        metadata: dict[str, Any] = {
            "elapsed_seconds": round(elapsed, 3),
            "analysis_type": "sar",
        }

        response = SARAnalysisResponse(
            report=report,
            structured_data=structured,
            session_id=session_data.session_id,
            metadata=metadata,
        )
        await session_manager.save_session(session_data)
        return response

    except Exception as e:
        logger.exception(f"SAR analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=AgentErrorResponse(
                error=str(e),
                error_type=type(e).__name__,
                detail="SAR analysis failed",
                session_id=request.session_id,
            ).model_dump(),
        )


@router.post("/stream")
async def stream_sar_analysis(request: SARAnalysisRequest) -> StreamingResponse:
    """
    Execute SAR analysis with streaming response.

    Returns Server-Sent Events (SSE) stream with analysis progress.
    """
    session_manager = get_session_manager()

    try:
        session_data = await session_manager.get_or_create_session_async(request.session_id)

        # For SAR, we use the general agent streaming
        async def _stream_with_persist():
            try:
                async for event in stream_agent_response(
                    session_data, f"SAR Analysis: {request.query}"
                ):
                    yield event
            finally:
                await session_manager.save_session(session_data)

        return StreamingResponse(
            _stream_with_persist(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Session-ID": session_data.session_id,
            },
        )

    except Exception as e:
        logger.exception(f"SAR stream setup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/report",
    response_model=SARAnalysisResponse,
    responses={500: {"model": AgentErrorResponse}},
)
async def generate_sar_report(request: SARAnalysisRequest) -> SARAnalysisResponse:
    """
    Generate a comprehensive SAR report.

    Similar to /analyze but optimized for report generation.
    """
    # Delegate to analyze endpoint with report-focused configuration
    return await analyze_sar(request)
