# type: ignore[attr-defined]
"""CLI entry point for molx-agent."""

import logging

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from molx_agent import version

app = typer.Typer(
    name="molx-agent",
    help="Drug design agent - SAR analysis powered by AI",
    add_completion=False,
)
console = Console()


def version_callback(print_version: bool) -> None:
    """Print the version of the package."""
    if print_version:
        console.print(f"[yellow]molx-agent[/] version: [bold blue]{version}[/]")
        raise typer.Exit()


@app.callback()
def main_callback(
    print_version: bool = typer.Option(
        None,
        "-v",
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Print the version of molx-agent.",
    ),
) -> None:
    """Drug design agent - SAR analysis powered by AI."""
    pass


@app.command()
def sar(
    query: str = typer.Argument(..., help="SAR analysis query"),
    verbose: bool = typer.Option(
        False, "-V", "--verbose", help="Enable verbose output"
    ),
    output_json: bool = typer.Option(
        False, "--json", help="Output structured results as JSON"
    ),
) -> None:
    """Run SAR analysis agent.

    Example:
        molx-agent sar "Analyze SAR of aspirin derivatives for COX-2 selectivity"
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)

    console.print(
        Panel(
            f"[bold]Query:[/] {query}",
            title="[cyan]SAR Analysis[/]",
            border_style="cyan",
        )
    )

    with console.status("[bold green]Running SAR analysis..."):
        try:
            from molx_agent.agents import run_sar_agent

            state = run_sar_agent(query)
        except Exception as e:
            console.print(f"[red]Error:[/] {e}")
            raise typer.Exit(1)

    report = state.get("final_response", "Analysis complete.")

    if output_json:
        import json

        console.print(json.dumps(state, indent=2))
    else:
        console.print()
        console.print(Markdown(report))
        console.print()
        console.print(
            Panel(
                "[green]Analysis complete![/]",
                border_style="green",
            )
        )


@app.command()
def chat(
    verbose: bool = typer.Option(
        False, "-V", "--verbose", help="Enable verbose output"
    ),
) -> None:
    """Start interactive chat session with the SAR agent.

    Type your questions and get responses. Use 'exit', 'quit', or Ctrl+C to end.

    Example:
        molx-agent chat
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)

    console.print(
        Panel(
            "[bold]Interactive Drug Design Agent[/]\n\n"
            "Type your questions to analyze molecules.\n"
            "Commands: [cyan]exit[/], [cyan]quit[/], [cyan]clear[/] (history)",
            title="[cyan]MolxAgent Chat[/]",
            border_style="cyan",
        )
    )

    try:
        from molx_agent.agents.molx import ChatSession

        session = ChatSession()

        while True:
            try:
                user_input = Prompt.ask("\n[bold cyan]You[/]")

                if user_input.lower() in ("exit", "quit", "q"):
                    console.print("[yellow]Goodbye![/]")
                    break
                if user_input.lower() == "clear":
                    session.clear()
                    console.print("[green]Conversation history cleared.[/]")
                    continue
                if user_input.lower() == "history":
                    history = session.get_history()
                    if history:
                        for msg in history:
                            role = msg.get("role", "user")
                            content = msg.get("content", "")[:100]
                            console.print(f"  [{role}]: {content}...")
                    else:
                        console.print("[dim]No history yet.[/]")
                    continue
                if not user_input.strip():
                    continue

                with console.status("[bold green]Thinking..."):
                    response = session.send(user_input)

                console.print()
                console.print(Panel(Markdown(response), title="[bold green]Agent[/]"))

            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted. Goodbye![/]")
                break

    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1)


@app.command()
def hello(
    name: str = typer.Argument(..., help="Name to greet"),
) -> None:
    """Print a greeting (demo command)."""
    from molx_agent.example import hello as hello_fn

    console.print(f"[bold green]{hello_fn(name)}[/]")


if __name__ == "__main__":
    app()
