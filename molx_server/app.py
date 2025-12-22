"""
FastAPI application factory and configuration.

Creates and configures the main FastAPI application with:
- CORS middleware
- Custom middleware (logging, auth)
- API routes
- OpenAPI documentation
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from molx_server import __version__
from molx_server.config import get_server_settings
from molx_server.middleware import APIKeyMiddleware, RequestLoggingMiddleware
from molx_server.routes import (
    agent_router,
    files_router,
    health_router,
    sar_router,
    session_router,
)
from molx_server.session import get_session_manager


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting MolX Server...")

    # Initialize session manager with storage backend
    session_manager = get_session_manager()
    await session_manager.initialize()
    await session_manager.start_cleanup_task()

    logger.info(f"MolX Server v{__version__} started successfully")

    yield

    # Shutdown
    logger.info("Shutting down MolX Server...")

    # Stop session cleanup and close storage
    await session_manager.stop_cleanup_task()
    await session_manager.close()

    logger.info("MolX Server shutdown complete")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    settings = get_server_settings()

    # Create FastAPI app
    app = FastAPI(
        title=settings.api_title,
        description=settings.api_description,
        version=settings.api_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )

    # Add custom middleware
    app.add_middleware(RequestLoggingMiddleware)
    # TODO: Close Auth for now
    # app.add_middleware(APIKeyMiddleware)

    # Register routes
    # Health routes at root level
    app.include_router(health_router)

    # API routes with version prefix
    app.include_router(agent_router, prefix=settings.api_prefix)
    app.include_router(files_router, prefix=settings.api_prefix)
    app.include_router(sar_router, prefix=settings.api_prefix)
    app.include_router(session_router, prefix=settings.api_prefix)

    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root() -> dict:
        """Root endpoint with API information."""
        return {
            "name": settings.api_title,
            "version": settings.api_version,
            "description": settings.api_description,
            "docs": "/docs",
            "health": "/health",
        }

    return app


# Create application instance
app = create_app()
