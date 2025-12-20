"""
API Key authentication middleware.

Optional middleware for securing API endpoints with API keys.
"""

import logging
from typing import Callable, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from molx_server.config import get_server_settings


logger = logging.getLogger(__name__)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Middleware for API key authentication.

    Validates API key in:
    - X-API-Key header
    - api_key query parameter
    - Authorization: Bearer <key> header

    Skips validation for:
    - Health check endpoints
    - OpenAPI documentation endpoints
    """

    # Paths that don't require authentication
    EXEMPT_PATHS = {
        "/health",
        "/health/live",
        "/health/ready",
        "/docs",
        "/redoc",
        "/openapi.json",
    }

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """Process request and validate API key if enabled."""
        settings = get_server_settings()

        # Skip if API key auth is disabled
        if not settings.api_key_enabled:
            return await call_next(request)

        # Skip exempt paths
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        # Extract API key
        api_key = self._extract_api_key(request)

        # Validate API key
        if not api_key or api_key not in settings.api_keys:
            logger.warning(
                f"Invalid API key for request: {request.method} {request.url.path}"
            )
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid or missing API key"},
            )

        return await call_next(request)

    def _extract_api_key(self, request: Request) -> Optional[str]:
        """
        Extract API key from request.

        Checks in order:
        1. X-API-Key header
        2. api_key query parameter
        3. Authorization: Bearer header
        """
        # Check X-API-Key header
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return api_key

        # Check query parameter
        api_key = request.query_params.get("api_key")
        if api_key:
            return api_key

        # Check Authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header[7:]

        return None
