"""
Request logging middleware.

Logs incoming requests and outgoing responses with timing information.
"""

import logging
import time
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging HTTP requests and responses.

    Logs:
    - Request method and path
    - Response status code
    - Request processing time
    """

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """Process request and log details."""
        start_time = time.time()

        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path}"
        )

        # Process request
        try:
            response = await call_next(request)
        except Exception as e:
            logger.error(
                f"Request failed: {request.method} {request.url.path} - {e}"
            )
            raise

        # Calculate processing time
        process_time = time.time() - start_time

        # Add timing header
        response.headers["X-Process-Time"] = f"{process_time:.4f}"

        # Log response
        logger.info(
            f"Response: {request.method} {request.url.path} "
            f"status={response.status_code} time={process_time:.4f}s"
        )

        return response
