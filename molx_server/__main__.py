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
    """Configure colorful logging using rich library.
    
    Args:
        verbose: Enable INFO level logging
        debug: Enable DEBUG level logging
    """
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    # Try to use rich for colorful logging
    try:
        from rich.logging import RichHandler
        from rich.console import Console
        
        console = Console(force_terminal=True, color_system="auto")
        
        # Configure rich handler with colors
        handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
            tracebacks_show_locals=debug,
            markup=True,
        )
        handler.setLevel(level)
        
        # Configure root logger
        logging.basicConfig(
            level=level,
            format="%(message)s",
            datefmt="[%Y-%m-%d %H:%M:%S]",
            handlers=[handler],
            force=True,
        )
        
        # Also set uvicorn loggers to use the same handler
        for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
            uvicorn_logger = logging.getLogger(logger_name)
            uvicorn_logger.handlers = [handler]
            
    except ImportError:
        # Fallback to standard logging with ANSI colors
        class ColorFormatter(logging.Formatter):
            """Formatter with ANSI color codes."""
            
            COLORS = {
                'DEBUG': '\033[36m',     # Cyan
                'INFO': '\033[32m',      # Green
                'WARNING': '\033[33m',   # Yellow
                'ERROR': '\033[31m',     # Red
                'CRITICAL': '\033[35m',  # Magenta
            }
            RESET = '\033[0m'
            
            def format(self, record):
                color = self.COLORS.get(record.levelname, '')
                record.levelname = f"{color}{record.levelname:8}{self.RESET}"
                record.name = f"\033[34m{record.name}\033[0m"  # Blue for name
                return super().format(record)
        
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(ColorFormatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        
        logging.basicConfig(
            level=level,
            handlers=[handler],
            force=True,
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
    
    # Get core settings for session config
    from molx_core.config import get_core_settings
    core_settings = get_core_settings()

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
    typer.echo("-" * 40)
    typer.echo("Session Settings (from molx_core):")
    typer.echo(f"  TTL: {core_settings.session_ttl}s")
    typer.echo(f"  Cleanup Enabled: {core_settings.session_cleanup_enabled}")
    typer.echo(f"  Cleanup Interval: {core_settings.session_cleanup_interval}s")
    typer.echo(f"  Max Count: {core_settings.session_max_count}")


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
