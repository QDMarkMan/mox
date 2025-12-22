"""
CLI entry point for molx-server.

Provides command-line interface for starting the server.
"""

import logging
import sys

import typer
import uvicorn

from molx_server.config import get_server_settings


cli = typer.Typer(
    name="molx-server",
    help="Molx Agent API Server",
    add_completion=False,
)


def setup_logging(verbose: bool = False, debug: bool = False) -> None:
    """Configure logging based on verbosity level."""
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@cli.command()
def run(
    host: str = typer.Option(None, "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(None, "--port", "-p", help="Port to bind to"),
    workers: int = typer.Option(None, "--workers", "-w", help="Number of workers"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Debug mode"),
) -> None:
    """Start the Molx Agent API server."""
    settings = get_server_settings()

    # Setup logging
    setup_logging(verbose, debug or settings.debug)

    # Use settings defaults if not specified
    final_host = host or settings.host
    final_port = port or settings.port
    final_workers = workers or settings.workers
    final_reload = reload or settings.reload

    typer.echo(f"Starting MolX Server on {final_host}:{final_port}")
    typer.echo(f"API docs: http://{final_host}:{final_port}/docs")

    uvicorn.run(
        "molx_server.app:app",
        host=final_host,
        port=final_port,
        workers=final_workers if not final_reload else 1,
        reload=final_reload,
        log_level="debug" if debug else "info" if verbose else "warning",
    )


@cli.command()
def version() -> None:
    """Show server version."""
    from molx_server import __version__

    typer.echo(f"molx-server version {__version__}")


@cli.command()
def config() -> None:
    """Show current configuration."""
    settings = get_server_settings()

    typer.echo("Current Configuration:")
    typer.echo("-" * 40)
    typer.echo(f"Host: {settings.host}")
    typer.echo(f"Port: {settings.port}")
    typer.echo(f"Workers: {settings.workers}")
    typer.echo(f"Debug: {settings.debug}")
    typer.echo(f"API Prefix: {settings.api_prefix}")
    typer.echo(f"CORS Origins: {settings.cors_origins}")
    typer.echo(f"Rate Limit Enabled: {settings.rate_limit_enabled}")
    typer.echo(f"API Key Enabled: {settings.api_key_enabled}")
    typer.echo(f"Session TTL: {settings.session_ttl_seconds}s")
    typer.echo(f"Max Sessions: {settings.max_sessions}")


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
