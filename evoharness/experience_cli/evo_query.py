from __future__ import annotations

import json
import re
from difflib import unified_diff
from pathlib import Path
from typing import Any, Optional

import typer

app = typer.Typer(help="Query the EvoHarness experience store. Designed for proposer agents.")

_MINIMIZE_METRICS = frozenset(
    {
        "context_tokens",
        "latency_ms",
        "cost_usd",
        "tokens",
        "eval_cost_usd",
        "proposer_tokens_used",
    }
)


def _find_evo_dir() -> Path:
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        evo = p / ".evo"
        if evo.is_dir():
            return evo
    typer.echo("error: no .evo directory found (walked up from cwd)", err=True)
    raise typer.Exit(code=1)


def _frontier_objectives(project_dir: Path, frontier_path: Path) -> list[dict[str, str]]:
    cfg_path = project_dir / "evo.toml"
    if cfg_path.is_file():
        from evoharness.core.config import load_config

        cfg = load_config(cfg_path)
        objs: list[dict[str, str]] = [
            {"name": cfg.scoring.primary_metric, "direction": cfg.scoring.direction}
        ]
        for sec in cfg.scoring.secondary:
            objs.append({"name": sec.name, "direction": sec.direction})
        return objs
    raw = json.loads(frontier_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return [{"name": "accuracy", "direction": "maximize"}]
    keys: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        scores = item.get("scores")
        if isinstance(scores, dict):
            keys.update(str(k) for k in scores.keys())
    if not keys:
        return [{"name": "accuracy", "direction": "maximize"}]
    return [
        {
            "name": k,
            "direction": "minimize" if k in _MINIMIZE_METRICS else "maximize",
        }
        for k in sorted(keys)
    ]


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows and not headers:
        return ""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))
    sep = " | "

    def fmt_row(cells: list[str]) -> str:
        parts = [cells[i].ljust(widths[i]) for i in range(len(widths))]
        return sep.join(parts)

    lines = [fmt_row(headers), fmt_row(["-" * w for w in widths])]
    lines.extend(fmt_row(row) for row in rows)
    return "\n".join(lines) + "\n"


def _score_cell(agg: dict, metric: str) -> str:
    v = agg.get(metric)
    if v is None:
        return ""
    try:
        return f"{float(v):.6g}"
    except (TypeError, ValueError):
        return str(v)


def _parent_display(row: dict, history: Any = None) -> str:
    pid = row.get("parent_id")
    if pid:
        return str(pid)
    if history is not None:
        try:
            c = history.get_candidate(str(row.get("candidate_id", "")))
            if c.metadata.parent_ids:
                return c.metadata.parent_ids[0]
        except FileNotFoundError:
            pass
    return ""


def query_leaderboard(
    evo_dir: Path,
    metric: str = "accuracy",
    top: int = 10,
    direction: str = "maximize",
) -> str:
    from evoharness.core.history import HistoryStore

    history = HistoryStore(evo_dir.parent)
    rows = history.get_leaderboard(metric, direction=direction)[: max(0, top)]
    headers = ["ID", "Parent", "Score", "Strategy"]
    table_rows: list[list[str]] = []
    for row in rows:
        agg = row.get("scores") or {}
        agg = agg.get("aggregate") or {}
        table_rows.append(
            [
                str(row.get("candidate_id", "")),
                _parent_display(row, history),
                _score_cell(agg, metric),
                str(row.get("strategy_tag") or ""),
            ]
        )
    return _format_table(headers, table_rows)


def query_frontier(evo_dir: Path) -> str:
    path = evo_dir / "frontier.json"
    if not path.is_file():
        return "error: frontier.json not found\n"
    try:
        from evoharness.core.frontier import ParetoFrontier

        objectives = _frontier_objectives(evo_dir.parent, path)
        pf = ParetoFrontier.load(path, objectives)
    except (json.JSONDecodeError, OSError, ValueError) as e:
        return f"error: failed to load frontier: {e}\n"
    lines: list[str] = []
    for p in pf.frontier:
        score_parts = [f"{k}={v:.6g}" for k, v in sorted(p.scores.items())]
        lines.append(f"{p.candidate_id}  " + "  ".join(score_parts))
    return "\n".join(lines) + ("\n" if lines else "no frontier points\n")


def _harness_py_relpaths(harness_root: Path) -> list[str]:
    if not harness_root.is_dir():
        return []
    return sorted({p.relative_to(harness_root).as_posix() for p in harness_root.rglob("*.py")})


def query_diff(evo_dir: Path, candidate_a: str, candidate_b: str) -> str:
    from evoharness.core.history import HistoryStore

    history = HistoryStore(evo_dir.parent)
    root_a = history.candidate_dir(candidate_a) / "harness"
    root_b = history.candidate_dir(candidate_b) / "harness"
    rels = sorted(set(_harness_py_relpaths(root_a)) | set(_harness_py_relpaths(root_b)))
    if not rels:
        return "no Python harness files in either candidate\n"
    chunks: list[str] = []
    for rel in rels:
        p_a = root_a / rel
        p_b = root_b / rel
        try:
            old_lines = (
                p_a.read_text(encoding="utf-8").splitlines(keepends=True) if p_a.is_file() else []
            )
            new_lines = (
                p_b.read_text(encoding="utf-8").splitlines(keepends=True) if p_b.is_file() else []
            )
        except OSError as e:
            return f"error reading harness files: {e}\n"
        if old_lines == new_lines:
            continue
        diff_iter = unified_diff(
            old_lines,
            new_lines,
            fromfile=f"{candidate_a}/{rel}",
            tofile=f"{candidate_b}/{rel}",
        )
        chunk = "".join(diff_iter)
        if chunk:
            chunks.append(chunk)
    if not chunks:
        return "no differences in harness Python files\n"
    return "".join(chunks)


def query_failures(
    evo_dir: Path,
    candidate_id: str,
    metric: str = "accuracy",
    top: int = 5,
) -> str:
    from evoharness.core.history import HistoryStore

    history = HistoryStore(evo_dir.parent)
    try:
        failed = history.get_failures(candidate_id, metric=metric, top=top)
    except FileNotFoundError as e:
        return f"error: {e}\n"
    if not failed:
        return f"no failures under threshold for candidate {candidate_id!r} (metric={metric})\n"
    headers = ["task_id", "score", "error_type", "error_message"]
    rows: list[list[str]] = []
    for r in failed:
        et = r.get("error_type") or ""
        em = r.get("error_message") or ""
        em_one = em.replace("\n", " ").strip()
        if len(em_one) > 200:
            em_one = em_one[:197] + "..."
        sc = r.get("score")
        sc_s = f"{float(sc):.6g}" if sc is not None else ""
        rows.append(
            [
                str(r.get("task_id", "")),
                sc_s,
                str(et),
                em_one,
            ]
        )
    return _format_table(headers, rows)


def _grep_file(
    path: Path,
    pattern: re.Pattern[str],
    context: int,
) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    lines = text.splitlines()
    match_idxs = [i for i, ln in enumerate(lines) if pattern.search(ln)]
    if not match_idxs:
        return []
    out: list[str] = []
    seen: set[int] = set()
    for i in match_idxs:
        lo, hi = max(0, i - context), min(len(lines), i + context + 1)
        for j in range(lo, hi):
            if j in seen:
                continue
            seen.add(j)
            mark = "M" if pattern.search(lines[j]) else " "
            out.append(f"{path.as_posix()}:{j + 1}:{mark} {lines[j]}")
    return out


def query_grep(
    evo_dir: Path,
    pattern: str,
    scope: str = "all",
    candidates: str | None = None,
) -> str:
    from evoharness.core.history import HistoryStore

    if scope not in ("traces", "code", "all"):
        return f"error: scope must be traces, code, or all (got {scope!r})\n"
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return f"error: invalid regex: {e}\n"

    history = HistoryStore(evo_dir.parent)
    if candidates is None:
        ids = [c.metadata.candidate_id for c in history.list_candidates()]
    else:
        ids = [x.strip() for x in candidates.split(",") if x.strip()]

    want_traces = scope in ("traces", "all")
    want_code = scope in ("code", "all")
    ctx = 2
    all_lines: list[str] = []
    for cid in ids:
        cdir = history.candidate_dir(cid)
        if not cdir.is_dir():
            all_lines.append(f"# skip missing candidate {cid}")
            continue
        paths: list[Path] = []
        if want_traces:
            td = cdir / "traces"
            if td.is_dir():
                paths.extend(sorted(td.glob("*.jsonl")))
        if want_code:
            harness = cdir / "harness"
            if harness.is_dir():
                paths.extend(sorted(harness.rglob("*.py")))
        for p in paths:
            all_lines.extend(_grep_file(p, rx, ctx))
    if not all_lines:
        return "no matches\n"
    return "\n".join(all_lines) + "\n"


def query_lineage(evo_dir: Path, candidate_id: str) -> str:
    path = evo_dir / "lineage.json"
    if not path.is_file():
        return "error: lineage.json not found\n"
    try:
        from evoharness.core.lineage import LineageGraph

        g = LineageGraph.load(path)
    except (json.JSONDecodeError, OSError, ValueError) as e:
        return f"error: failed to load lineage: {e}\n"
    return g.format_lineage(candidate_id) + "\n"


def query_task_matrix(evo_dir: Path, metric: str = "accuracy") -> str:
    from evoharness.core.history import HistoryStore

    history = HistoryStore(evo_dir.parent)
    data = history.get_task_matrix(metric)
    cids = data["candidate_ids"]
    tids = data["task_ids"]
    matrix: list[list[float | None]] = data["matrix"]
    if not cids:
        return "no candidates\n"
    headers = ["ID"] + list(tids)

    def cell(v: float | None) -> str:
        if v is None:
            return ""
        return f"{float(v):.6g}"

    rows: list[list[str]] = []
    for i, cid in enumerate(cids):
        row = [cid] + [cell(matrix[i][j]) for j in range(len(tids))]
        rows.append(row)
    return _format_table(headers, rows)


@app.command()
def leaderboard(
    metric: str = typer.Option("accuracy", "--metric", "-m"),
    top: int = typer.Option(10, "--top", "-n"),
    direction: str = typer.Option("maximize", "--direction", "-d"),
) -> None:
    """Print ranked leaderboard of candidates."""
    typer.echo(query_leaderboard(_find_evo_dir(), metric=metric, top=top, direction=direction), nl=False)


@app.command()
def frontier() -> None:
    """Print current Pareto frontier candidates with all metrics."""
    typer.echo(query_frontier(_find_evo_dir()), nl=False)


@app.command()
def diff(
    candidate_a: str = typer.Argument(..., metavar="A"),
    candidate_b: str = typer.Argument(..., metavar="B"),
) -> None:
    """Print unified diff of harness code between two candidates."""
    typer.echo(query_diff(_find_evo_dir(), candidate_a, candidate_b), nl=False)


@app.command()
def failures(
    candidate_id: str = typer.Argument(...),
    metric: str = typer.Option("accuracy", "--metric", "-m"),
    top: int = typer.Option(5, "--top", "-n"),
) -> None:
    """List failed tasks with error summaries."""
    typer.echo(query_failures(_find_evo_dir(), candidate_id, metric=metric, top=top), nl=False)


@app.command()
def grep(
    pattern: str = typer.Argument(...),
    scope: str = typer.Option("all", "--scope", "-s"),
    candidates: Optional[str] = typer.Option(
        None,
        "--candidates",
        "-c",
        help="Comma-separated candidate IDs; default all",
    ),
) -> None:
    """Search across traces/code for a regex pattern."""
    typer.echo(query_grep(_find_evo_dir(), pattern, scope=scope, candidates=candidates), nl=False)


@app.command()
def lineage(candidate_id: str = typer.Argument(...)) -> None:
    """Show ancestry of a candidate."""
    typer.echo(query_lineage(_find_evo_dir(), candidate_id), nl=False)


@app.command(name="task-matrix")
def task_matrix_cmd(
    metric: str = typer.Option("accuracy", "--metric", "-m"),
) -> None:
    """Print per-task score matrix. Rows=candidates, cols=tasks."""
    typer.echo(query_task_matrix(_find_evo_dir(), metric=metric), nl=False)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
