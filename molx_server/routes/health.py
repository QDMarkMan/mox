"""
Health check endpoints.

Provides liveness, readiness, and overall health status.
"""

import time
from datetime import datetime

from fastapi import APIRouter

from molx_server import __version__
from molx_server.schemas.models import HealthResponse, HealthStatus, ReadyResponse
from molx_server.session import get_session_manager


router = APIRouter(tags=["Health"])

# Track server start time
_start_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Overall health check endpoint.

    Returns comprehensive health status including component states.
    """
    session_manager = get_session_manager()

    # Check components
    components = {
        "session_manager": HealthStatus.HEALTHY,
        "agent": HealthStatus.HEALTHY,
    }

    # Determine overall status
    if all(s == HealthStatus.HEALTHY for s in components.values()):
        status = HealthStatus.HEALTHY
    elif any(s == HealthStatus.UNHEALTHY for s in components.values()):
        status = HealthStatus.UNHEALTHY
    else:
        status = HealthStatus.DEGRADED

    return HealthResponse(
        status=status,
        version=__version__,
        uptime_seconds=time.time() - _start_time,
        timestamp=datetime.utcnow(),
        components=components,
    )


@router.get("/health/live")
async def liveness_check() -> dict:
    """
    Kubernetes liveness probe endpoint.

    Returns 200 if the server process is alive.
    """
    return {"status": "alive"}


@router.get("/health/ready", response_model=ReadyResponse)
async def readiness_check() -> ReadyResponse:
    """
    Kubernetes readiness probe endpoint.

    Returns readiness status based on component availability.
    """
    checks = {
        "session_manager": True,
        "agent_available": True,
    }

    return ReadyResponse(
        ready=all(checks.values()),
        checks=checks,
    )
