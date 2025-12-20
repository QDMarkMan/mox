"""
Agent API endpoints.

Provides endpoints for invoking the MolX agent with sync, streaming, and batch modes.
"""

import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from molx_server.schemas.models import (
    AgentErrorResponse,
    AgentRequest,
    AgentResponse,
    AgentStatus,
    BatchItemResult,
    BatchRequest,
    BatchResponse,
)
from molx_server.session import get_session_manager
from molx_server.streaming import stream_agent_response


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agent", tags=["Agent"])


@router.post(
    "/invoke",
    response_model=AgentResponse,
    responses={500: {"model": AgentErrorResponse}},
)
async def invoke_agent(request: AgentRequest) -> AgentResponse:
    """
    Invoke the agent synchronously.

    Processes the query and returns the complete response.
    """
    start_time = time.time()
    session_manager = get_session_manager()

    try:
        # Get or create session
        session_data = session_manager.get_or_create_session(request.session_id)

        # Execute agent
        result = session_data.chat_session.send(request.query)

        # Build metadata
        elapsed = time.time() - start_time
        metadata: dict[str, Any] = {
            "elapsed_seconds": round(elapsed, 3),
            "message_count": session_data.message_count,
        }

        return AgentResponse(
            result=result,
            structured_data=None,
            session_id=session_data.session_id,
            metadata=metadata,
            status=AgentStatus.COMPLETED,
        )

    except Exception as e:
        logger.exception(f"Agent invocation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=AgentErrorResponse(
                error=str(e),
                error_type=type(e).__name__,
                session_id=request.session_id,
            ).model_dump(),
        )


@router.post("/stream")
async def stream_agent(request: AgentRequest) -> StreamingResponse:
    """
    Invoke the agent with streaming response.

    Returns Server-Sent Events (SSE) stream with incremental output.
    """
    session_manager = get_session_manager()

    try:
        session_data = session_manager.get_or_create_session(request.session_id)

        return StreamingResponse(
            stream_agent_response(session_data, request.query),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Session-ID": session_data.session_id,
            },
        )

    except Exception as e:
        logger.exception(f"Stream setup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/batch",
    response_model=BatchResponse,
    responses={500: {"model": AgentErrorResponse}},
)
async def batch_invoke(request: BatchRequest) -> BatchResponse:
    """
    Process multiple queries in batch.

    Processes all queries sequentially and returns aggregated results.
    """
    session_manager = get_session_manager()
    session_data = session_manager.get_or_create_session(request.session_id)

    results: list[BatchItemResult] = []
    successful = 0
    failed = 0

    for i, query in enumerate(request.queries):
        try:
            result = session_data.chat_session.send(query)
            results.append(
                BatchItemResult(
                    index=i,
                    query=query,
                    result=result,
                    error=None,
                    status=AgentStatus.COMPLETED,
                )
            )
            successful += 1
        except Exception as e:
            logger.error(f"Batch item {i} failed: {e}")
            results.append(
                BatchItemResult(
                    index=i,
                    query=query,
                    result=None,
                    error=str(e),
                    status=AgentStatus.FAILED,
                )
            )
            failed += 1

    return BatchResponse(
        results=results,
        total=len(request.queries),
        successful=successful,
        failed=failed,
        session_id=session_data.session_id,
    )
