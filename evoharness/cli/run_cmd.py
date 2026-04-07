from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from evoharness import __version__
from evoharness.core.config import load_config

console = Console()


def _load_dotenv(project_dir: Path) -> None:
    """Load .env from the project directory into os.environ (setdefault)."""
    env_file = project_dir / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def run(
    config_path: Path = typer.Option(
        Path("evo.toml"),
        "--config",
        "-c",
        exists=False,
        file_okay=True,
        dir_okay=False,
        writable=False,
        resolve_path=True,
        help="Path to evo.toml",
    ),
    iterations: Optional[int] = typer.Option(
        None,
        "--iterations",
        "-n",
        help="Override max_iterations",
    ),
    budget: Optional[float] = typer.Option(
        None,
        "--budget",
        "-b",
        help="Override max_cost_usd",
    ),
    proposer: Optional[str] = typer.Option(
        None,
        "--proposer",
        "-p",
        help="Override proposer (e.g. anthropic:claude-sonnet-4-20250514)",
    ),
    steer: Optional[str] = typer.Option(
        None,
        "--steer",
        "-s",
        help="Steering instruction for the proposer",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume using existing .evo state (skip baseline re-init when present)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Verbose logging",
    ),
) -> None:
    """Start the evolutionary search loop."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not config_path.is_file():
        console.print(f"[red]Config not found:[/red] {config_path}")
        raise typer.Exit(code=1)

    _load_dotenv(config_path.parent.resolve())

    try:
        config = load_config(config_path)
    except Exception as e:
        console.print(f"[red]Error loading config:[/red] {e}")
        raise typer.Exit(code=1) from e

    if iterations is not None:
        config.search.max_iterations = iterations
    if budget is not None:
        config.search.budget.max_cost_usd = budget
    if proposer is not None:
        config.search.proposer = proposer

    console.print(
        Panel.fit(
            f"[bold]EvoHarness[/bold] v{__version__} — optimizing [cyan]{config.project.name}[/cyan]\n"
            f"Max iterations: {config.search.max_iterations} · "
            f"Budget: [yellow]${config.search.budget.max_cost_usd:.2f}[/yellow] · "
            f"Proposer: [dim]{config.search.proposer}[/dim]",
            title="[bold green]Starting search[/bold green]",
            border_style="green",
        )
    )

    if resume:
        console.print(
            "[dim]Resume:[/dim] using existing [cyan].evo[/cyan] state where possible."
        )

    proposer_instance = _create_proposer(config)

    from evoharness.core.loop import SearchLoop

    project_dir = config_path.parent.resolve()
    search_loop = SearchLoop(
        config=config,
        project_dir=project_dir,
        proposer=proposer_instance,
    )

    def on_iteration(iteration: int, state: Any) -> None:
        used = getattr(state, "total_cost_usd", 0.0)
        cap = config.search.budget.max_cost_usd
        console.print(
            f"\n[bold]Iteration[/bold] {iteration}/{config.search.max_iterations}  "
            f"·  [yellow]Budget[/yellow] ${used:.2f} / ${cap:.2f}"
        )

    summary: dict[str, Any]
    try:
        with console.status("[bold green]Running evolutionary search…[/bold green]", spinner="dots"):
            summary = asyncio.run(
                search_loop.run(
                    steering=steer,
                    on_iteration=on_iteration,
                    resume=resume,
                ),
            )
    except KeyboardInterrupt:
        console.print("\n[yellow]Search interrupted by user.[/yellow]")
        summary = search_loop._build_summary()
    except Exception as e:
        console.print(f"\n[red]Search failed:[/red] {e}")
        logging.getLogger(__name__).exception("search loop failed")
        raise typer.Exit(code=1) from e

    _print_summary(summary, config)


def _create_proposer(config: Any) -> Any:
    proposer_str = (config.search.proposer or "").strip()

    if proposer_str.startswith("anthropic:"):
        model = proposer_str.split(":", 1)[1].strip() or "claude-sonnet-4-20250514"
        from evoharness.proposers.anthropic_api import AnthropicAPIProposer

        return AnthropicAPIProposer(model=model)

    if proposer_str == "claude-code":
        console.print(
            "[yellow]Claude Code proposer is not implemented yet; "
            "using Anthropic API with default model.[/yellow]"
        )
        from evoharness.proposers.anthropic_api import AnthropicAPIProposer

        return AnthropicAPIProposer()

    from evoharness.proposers.anthropic_api import AnthropicAPIProposer

    return AnthropicAPIProposer(model=proposer_str)


def _print_summary(summary: dict[str, Any], config: Any) -> None:
    console.print()
    console.print(
        Panel.fit(
            f"[bold green]Search complete[/bold green] — [dim]{config.project.name}[/dim]",
            title="Summary",
            border_style="green",
        )
    )

    table = Table(title="Search summary", show_header=True, header_style="bold")
    table.add_column("Metric", style="bold", no_wrap=True)
    table.add_column("Value")

    table.add_row("Iterations", str(summary.get("iterations", 0)))
    table.add_row("Total cost", f"${float(summary.get('total_cost_usd', 0)):.2f}")
    table.add_row("Elapsed time", f"{float(summary.get('elapsed_seconds', 0)):.0f}s")
    table.add_row("Candidates evaluated", str(summary.get("candidates_evaluated", 0)))
    table.add_row("Stop reason", str(summary.get("stop_reason", "unknown")))

    best = summary.get("best_scores") or {}
    if isinstance(best, dict):
        for metric, value in best.items():
            try:
                table.add_row(f"Best ({metric})", f"{float(value):.4f}")
            except (TypeError, ValueError):
                table.add_row(f"Best ({metric})", str(value))

    console.print(table)

    frontier_data = summary.get("frontier") or []
    if isinstance(frontier_data, list) and frontier_data:
        ft = Table(title="Pareto frontier", show_header=True, header_style="bold")
        ft.add_column("Candidate", style="cyan", no_wrap=True)
        first = frontier_data[0]
        score_keys: list[str] = []
        if isinstance(first, dict):
            scores = first.get("scores") or {}
            if isinstance(scores, dict):
                score_keys = list(scores.keys())
        for key in score_keys:
            ft.add_column(key, justify="right")

        for point in frontier_data:
            if not isinstance(point, dict):
                continue
            cid = str(point.get("candidate_id", ""))
            row: list[str] = [cid]
            scores = point.get("scores") or {}
            if isinstance(scores, dict):
                for key in score_keys:
                    raw = scores.get(key)
                    try:
                        row.append(f"{float(raw):.4f}" if raw is not None else "—")
                    except (TypeError, ValueError):
                        row.append(str(raw) if raw is not None else "—")
            ft.add_row(*row)

        console.print(ft)
