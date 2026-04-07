from __future__ import annotations

from datetime import datetime
from difflib import unified_diff
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from evoharness import __version__
from evoharness.cli.run_cmd import run
from evoharness.core.config import EvoConfig, load_config, save_config
from evoharness.core.cost_tracker import CostTracker
from evoharness.core.frontier import ParetoFrontier
from evoharness.core.history import HistoryStore

app = typer.Typer(
    name="evo",
    help=(
        "EvoHarness — automatically improve LLM agent harnesses through evolutionary search."
    ),
    rich_markup_mode="rich",
)

console = Console()


@app.callback(invoke_without_command=True)
def _main(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit(0)

_CONFIG_OPTION = typer.Option(
    Path("evo.toml"),
    "--config",
    "-c",
    exists=False,
    file_okay=True,
    dir_okay=False,
    resolve_path=True,
    help="Path to evo.toml (project root is its parent directory)",
)

app.command("run")(run)


@app.command("version")
def version_cmd() -> None:
    """Print the installed evo version."""
    console.print(f"evo {__version__}")


@app.command("init")
def init_cmd(
    directory: Path = typer.Argument(Path("."), help="Directory to create the project in."),
    template: str = typer.Option(
        "basic",
        "--template",
        "-t",
        help="Project template (only basic is bundled).",
    ),
    name: Optional[str] = typer.Option(
        None,
        "--name",
        "-n",
        help="Project name (default: directory name).",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite evo.toml if it already exists.",
    ),
) -> None:
    """Scaffold a new EvoHarness project with evo.toml, harness, and evals."""
    root = directory.expanduser().resolve()
    if template != "basic":
        console.print(
            f"[yellow]Template {template!r} is not bundled; using [bold]basic[/bold].[/yellow]"
        )

    root.mkdir(parents=True, exist_ok=True)
    evo_toml = root / "evo.toml"
    if evo_toml.is_file() and not force:
        console.print(f"[red]Already exists:[/red] {evo_toml}  (use [cyan]--force[/cyan] to replace)")
        raise typer.Exit(code=1)

    project_name = name or root.name or "my-agent"
    cfg = EvoConfig()
    cfg.project.name = project_name
    save_config(cfg, evo_toml)

    harness_dir = root / "harness"
    harness_dir.mkdir(parents=True, exist_ok=True)
    (harness_dir / "__init__.py").write_text("", encoding="utf-8")
    (harness_dir / "agent.py").write_text(
        '''"""Example harness entrypoint. Replace with your agent logic."""


def run(input_data, trace_callback=None):
    """Return a string answer for the eval task (see evals/eval_suite.py)."""
    if trace_callback:
        trace_callback({"type": "info", "message": "stub harness"})
    return str(input_data) if input_data is not None else ""
''',
        encoding="utf-8",
    )

    evals_dir = root / "evals"
    evals_dir.mkdir(parents=True, exist_ok=True)
    (evals_dir / "eval_suite.py").write_text(
        '''"""Minimal evaluation suite for a fresh EvoHarness project."""

from __future__ import annotations

from typing import Any, Callable


def get_tasks(split: str) -> list[dict[str, Any]]:
    if split == "search":
        return [
            {
                "task_id": "t1",
                "description": "Echo input",
                "input_data": "hello",
                "expected": "hello",
            },
            {
                "task_id": "t2",
                "description": "Echo input",
                "input_data": "world",
                "expected": "world",
            },
        ]
    return []


def evaluate(
    harness_module: Any,
    task: dict[str, Any],
    trace_callback: Callable[[dict], None],
) -> dict[str, Any]:
    agent = __import__(f"{harness_module.__name__}.agent", fromlist=["agent"])
    out = agent.run(task.get("input_data"), trace_callback)
    expected = task.get("expected")
    ok = out == expected
    return {
        "task_id": task["task_id"],
        "scores": {"accuracy": 1.0 if ok else 0.0},
        "output": out,
    }
''',
        encoding="utf-8",
    )

    console.print(
        Panel.fit(
            f"[bold green]Created[/bold green] EvoHarness project [cyan]{project_name}[/cyan]\n"
            f"{root}",
            title="evo init",
            border_style="green",
        )
    )
    console.print("Next: [bold]cd[/bold] into the project, then [bold]evo run[/bold].")


def _project_paths(config_path: Path) -> tuple[EvoConfig, Path]:
    if not config_path.is_file():
        console.print(f"[red]Config not found:[/red] {config_path}")
        raise typer.Exit(code=1)
    try:
        cfg = load_config(config_path)
    except Exception as e:
        console.print(f"[red]Error loading config:[/red] {e}")
        raise typer.Exit(code=1) from e
    return cfg, config_path.parent.resolve()


def _objectives_from_config(cfg: EvoConfig) -> list[dict[str, str]]:
    objs: list[dict[str, str]] = [
        {"name": cfg.scoring.primary_metric, "direction": cfg.scoring.direction}
    ]
    for sec in cfg.scoring.secondary:
        objs.append({"name": sec.name, "direction": sec.direction})
    return objs


def _harness_unified_diff(
    left_root: Path,
    right_root: Path,
    *,
    left_label: str,
    right_label: str,
) -> str:
    def py_relpaths(root: Path) -> list[str]:
        if not root.is_dir():
            return []
        return sorted({p.relative_to(root).as_posix() for p in root.rglob("*.py")})

    rels = sorted(set(py_relpaths(left_root)) | set(py_relpaths(right_root)))
    chunks: list[str] = []
    for rel in rels:
        p_a = left_root / rel
        p_b = right_root / rel
        a_lines = (
            p_a.read_text(encoding="utf-8").splitlines(keepends=True) if p_a.is_file() else []
        )
        b_lines = (
            p_b.read_text(encoding="utf-8").splitlines(keepends=True) if p_b.is_file() else []
        )
        if a_lines == b_lines:
            continue
        diff_iter = unified_diff(
            a_lines,
            b_lines,
            fromfile=f"a/{left_label}/{rel}",
            tofile=f"b/{right_label}/{rel}",
        )
        chunk = "".join(diff_iter)
        if chunk:
            chunks.append(chunk)
    return "\n".join(chunks)


@app.command("status")
def status_cmd(
    config_path: Path = _CONFIG_OPTION,
) -> None:
    """Show budget usage, frontier, and leaderboard snapshot."""
    cfg, project_dir = _project_paths(config_path)
    evo_dir = project_dir / ".evo"
    history = HistoryStore(project_dir)

    if not evo_dir.is_dir():
        console.print(
            Panel.fit(
                "[yellow]No .evo directory yet.[/yellow] Run [bold]evo run[/bold] to start a search.",
                title=cfg.project.name,
            )
        )
        raise typer.Exit(code=0)

    objectives = _objectives_from_config(cfg)
    frontier_path = evo_dir / "frontier.json"
    if frontier_path.is_file():
        frontier = ParetoFrontier.load(frontier_path, objectives)
        frontier_rows = frontier.to_json()
    else:
        frontier_rows = []

    cost_path = evo_dir / "cost_tracker.json"
    tracker = CostTracker(cost_path)
    cost_summary = tracker.summary()

    candidates = history.list_candidates()
    last_iter = 0
    for c in candidates:
        last_iter = max(last_iter, int(c.metadata.iteration or 0))

    console.print(
        Panel.fit(
            f"[bold]{cfg.project.name}[/bold]\n"
            f"Candidates on disk: [cyan]{len(candidates)}[/cyan]\n"
            f"Last recorded iteration: [cyan]{last_iter}[/cyan]\n"
            f"Cost: [yellow]${cost_summary['total_cost_usd']:.2f}[/yellow] / "
            f"[dim]${cfg.search.budget.max_cost_usd:.2f}[/dim] cap",
            title="[bold]Search status[/bold]",
            border_style="cyan",
        )
    )

    if frontier_rows:
        ft = Table(title="Pareto frontier", header_style="bold")
        ft.add_column("Candidate", style="cyan", no_wrap=True)
        keys = list(frontier_rows[0].get("scores", {}).keys()) if frontier_rows else []
        for k in keys:
            ft.add_column(k, justify="right")
        for row in frontier_rows:
            cid = str(row.get("candidate_id", ""))
            scores = row.get("scores") or {}
            line = [cid]
            for k in keys:
                v = scores.get(k)
                try:
                    line.append(f"{float(v):.4f}" if v is not None else "—")
                except (TypeError, ValueError):
                    line.append(str(v) if v is not None else "—")
            ft.add_row(*line)
        console.print(ft)
    else:
        console.print("[dim]Frontier not initialized yet.[/dim]")

    board = history.get_leaderboard(cfg.scoring.primary_metric, cfg.scoring.direction)
    if board:
        lt = Table(title=f"Leaderboard ({cfg.scoring.primary_metric})", header_style="bold")
        lt.add_column("Rank", justify="right")
        lt.add_column("ID", style="cyan", no_wrap=True)
        lt.add_column("Score", justify="right")
        lt.add_column("Strategy", style="dim")
        for i, row in enumerate(board[:15], start=1):
            agg = (row.get("scores") or {}).get("aggregate") or {}
            raw = agg.get(cfg.scoring.primary_metric)
            try:
                score_txt = f"{float(raw):.4f}" if raw is not None else "—"
            except (TypeError, ValueError):
                score_txt = str(raw) if raw is not None else "—"
            lt.add_row(
                str(i),
                str(row.get("candidate_id", "")),
                score_txt,
                str(row.get("strategy_tag") or "—"),
            )
        console.print(lt)


@app.command("inspect")
def inspect_cmd(
    candidate_id: str = typer.Argument(help="Candidate directory id (e.g. 000, 001)."),
    config_path: Path = _CONFIG_OPTION,
    show_diff: bool = typer.Option(
        False,
        "--diff",
        "-d",
        help="Include unified diff from parent (if any).",
    ),
) -> None:
    """Show metadata, scores, and summary for one candidate."""
    cfg, project_dir = _project_paths(config_path)
    history = HistoryStore(project_dir)
    try:
        cand = history.get_candidate(candidate_id)
    except FileNotFoundError:
        console.print(f"[red]Unknown candidate:[/red] {candidate_id}")
        raise typer.Exit(code=1) from None

    cdir = history.candidate_dir(candidate_id)
    created = cand.metadata.created_at
    if isinstance(created, datetime):
        created_s = created.isoformat()
    else:
        created_s = str(created)

    meta_tbl = Table(show_header=False, box=None, padding=(0, 2))
    meta_tbl.add_column("Key", style="bold dim")
    meta_tbl.add_column("Value")
    meta_tbl.add_row("ID", cand.metadata.candidate_id)
    meta_tbl.add_row("Created", created_s)
    meta_tbl.add_row("Iteration", str(cand.metadata.iteration))
    parents = cand.primary_parents
    meta_tbl.add_row("Parents", ", ".join(parents) if parents else "—")
    meta_tbl.add_row("Strategy", str(cand.metadata.strategy_tag or "—"))
    meta_tbl.add_row("Proposer model", str(cand.metadata.proposer_model or "—"))
    meta_tbl.add_row(
        "Proposer cost",
        f"${float(cand.metadata.proposer_cost_usd):.4f} ({cand.metadata.proposer_tokens_used} tok)",
    )

    console.print(
        Panel(
            meta_tbl,
            title=f"[bold cyan]Candidate {candidate_id}[/bold cyan]",
            border_style="cyan",
        )
    )

    st = Table(title="Aggregate scores", header_style="bold")
    st.add_column("Metric", style="bold")
    st.add_column("Value", justify="right")
    for k, v in sorted(cand.scores.aggregate.items()):
        try:
            st.add_row(k, f"{float(v):.4f}")
        except (TypeError, ValueError):
            st.add_row(k, str(v))
    console.print(st)

    summary_text = cand.summary.strip() or "(empty summary)"
    console.print(
        Panel(
            summary_text,
            title="Summary",
            border_style="dim",
            expand=False,
        )
    )

    if show_diff:
        patch_path = cdir / "diff_from_parent.patch"
        if patch_path.is_file():
            diff_body = patch_path.read_text(encoding="utf-8")
            console.print(
                Syntax(diff_body, "diff", theme="ansi_dark", line_numbers=True, word_wrap=True)
            )
        else:
            console.print("[dim]No diff_from_parent.patch for this candidate.[/dim]")


@app.command("compare")
def compare_cmd(
    candidate_a: str = typer.Argument(help="First candidate id."),
    candidate_b: str = typer.Argument(help="Second candidate id."),
    config_path: Path = _CONFIG_OPTION,
) -> None:
    """Show a unified diff between two candidates' harness trees."""
    _cfg, project_dir = _project_paths(config_path)
    history = HistoryStore(project_dir)
    for label, cid in (("A", candidate_a), ("B", candidate_b)):
        try:
            history.get_candidate(cid)
        except FileNotFoundError:
            console.print(f"[red]Unknown candidate {label}:[/red] {cid}")
            raise typer.Exit(code=1) from None

    left = history.candidate_dir(candidate_a) / "harness"
    right = history.candidate_dir(candidate_b) / "harness"
    diff_text = _harness_unified_diff(
        left, right, left_label=candidate_a, right_label=candidate_b
    )
    if not diff_text.strip():
        console.print(
            f"[dim]No Python file differences between {candidate_a} and {candidate_b}.[/dim]"
        )
        return
    console.print(
        Syntax(
            diff_text,
            "diff",
            theme="ansi_dark",
            line_numbers=False,
            word_wrap=True,
        )
    )


@app.command("report")
def report_cmd(
    config_path: Path = _CONFIG_OPTION,
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Write Markdown report to this file instead of printing.",
    ),
) -> None:
    """Emit a Markdown summary of the search (stdout or file)."""
    cfg, project_dir = _project_paths(config_path)
    history = HistoryStore(project_dir)
    evo_dir = project_dir / ".evo"

    lines: list[str] = [
        f"# EvoHarness report — {cfg.project.name}",
        "",
        f"- Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- Config: `{config_path}`",
        "",
        "## Settings",
        "",
        f"- Primary metric: **{cfg.scoring.primary_metric}** ({cfg.scoring.direction})",
        f"- Max iterations: {cfg.search.max_iterations}",
        f"- Budget cap: ${cfg.search.budget.max_cost_usd:.2f}",
        f"- Proposer: `{cfg.search.proposer}`",
        "",
    ]

    if evo_dir.is_dir():
        snap = evo_dir / "config.snapshot.json"
        if snap.is_file():
            lines.extend(["## Config snapshot", "", f"Stored at `{snap.relative_to(project_dir)}`.", ""])

    objectives = _objectives_from_config(cfg)
    frontier_path = evo_dir / "frontier.json"
    if frontier_path.is_file():
        frontier = ParetoFrontier.load(frontier_path, objectives)
        lines.extend(["## Pareto frontier", ""])
        for p in frontier.to_json():
            parts: list[str] = []
            for k, v in sorted(p["scores"].items()):
                try:
                    parts.append(f"{k}={float(v):.4f}")
                except (TypeError, ValueError):
                    parts.append(f"{k}={v}")
            lines.append(f"- **{p['candidate_id']}**: {', '.join(parts)}")
        lines.append("")
    else:
        lines.extend(["## Pareto frontier", "", "_Not initialized._", ""])

    cost_path = evo_dir / "cost_tracker.json"
    if cost_path.is_file():
        summary = CostTracker(cost_path).summary()
        lines.extend(
            [
                "## Costs",
                "",
                f"- Total: ${summary['total_cost_usd']:.2f}",
                f"- Proposer: ${summary['proposer_cost_usd']:.2f}",
                f"- Eval: ${summary['eval_cost_usd']:.2f}",
                "",
            ]
        )

    lines.extend(["## Candidates", ""])
    candidates = history.list_candidates()
    if not candidates:
        lines.append("_No candidates yet._")
    else:
        board = history.get_leaderboard(cfg.scoring.primary_metric, cfg.scoring.direction)
        primary = cfg.scoring.primary_metric
        for row in board:
            cid = row["candidate_id"]
            agg = (row.get("scores") or {}).get("aggregate") or {}
            try:
                score = float(agg.get(primary, 0.0))
                score_s = f"{score:.4f}"
            except (TypeError, ValueError):
                score_s = str(agg.get(primary, "—"))
            lines.append(f"- **{cid}** — {primary}={score_s} — _{row.get('strategy_tag') or ''}_")
    lines.append("")

    text = "\n".join(lines)
    if output is not None:
        out_path = output.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        console.print(f"[green]Wrote[/green] {out_path}")
    else:
        console.print(Markdown(text))


if __name__ == "__main__":
    app()
