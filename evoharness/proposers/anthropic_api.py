from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
import re
from pathlib import Path
from typing import Any, Mapping

import anthropic

from evoharness.proposers.base import BaseProposer, ProposerResult

logger = logging.getLogger(__name__)

PROPOSER_SYSTEM_PROMPT = """You are an expert harness engineer. Your job is to improve LLM agent harness
code -- the code that wraps a language model and controls what context it sees,
what tools it has, how it manages state, and how its outputs are processed.

## Your Goal

Maximize {primary_metric} on the evaluation tasks, while considering
secondary objectives: {secondary_metrics}.

## What You Have Access To

You have read access to a directory of ALL previously evaluated harness
candidates at: {history_dir}

Each candidate directory (e.g., candidates/007/) contains:
  - harness/           Full harness source code
  - scores.json        Evaluation scores
  - summary.txt        What this candidate attempted
  - diff_from_parent.patch  Changes from parent candidate
  - metadata.json      Timing, cost, parent info
  - traces/            One JSONL file per eval task

You also have query tools. Use them to save time:
  - evo_query_leaderboard   Ranked candidates
  - evo_query_frontier      Pareto frontier
  - evo_query_diff          Diff between candidates
  - evo_query_failures      Failed tasks for a candidate
  - evo_query_task_matrix   Per-task score matrix
  - evo_query_lineage       Ancestry graph
  - evo_query_grep          Search across traces and code

## Current State

{leaderboard}

Pareto frontier:
{frontier}

Iteration: {iteration} / {max_iterations}
Budget used: ${cost_used:.2f} / ${max_cost:.2f}

## Evaluation Tasks

The harness is being evaluated on:
{task_descriptions}

## Your Process -- FOLLOW THIS CAREFULLY

### Step 1: DIAGNOSE (spend at least 40% of your effort here)
Do NOT jump to proposing changes. First build understanding:
a) Use evo_query_leaderboard and evo_query_task_matrix to identify top and failing candidates.
b) Use evo_query_failures for top candidates. Understand WHICH TASKS they fail on.
c) Read execution traces for failed tasks. Understand WHY they failed.
d) Compare traces between passed and failed tasks.
e) Use evo_query_diff between candidates. Understand what changes helped or hurt.
f) Check if regressions share a common factor (confound).

### Step 2: HYPOTHESIZE (be specific and causal)
Form a SPECIFIC, FALSIFIABLE hypothesis. Write it in your reasoning.

### Step 3: DESIGN CHANGES (follow safety principles)
a) PREFER ADDITIVE over subtractive changes.
b) ISOLATE your changes. Don't bundle independent fixes.
c) SMALL, TARGETED edits beat large rewrites.
d) After multiple regressions from a component, STOP modifying it. Pivot.
e) Consider COMPOSING successful changes from different lineages.

### Step 4: IMPLEMENT
Write new harness code using write_harness_file.
You can modify files matching: {mutable_files}
You CANNOT modify: {readonly_files}

### Step 5: VALIDATE
Check for syntax errors, missing imports, interface preservation.

### Step 6: SUBMIT
Call submit_candidate with reasoning, parent_id, and strategy_tag.

## Rules
1. Read traces from at least 3 prior candidates before proposing (unless fewer exist).
2. State your causal hypothesis in the reasoning.
3. Don't repeat strategies that already failed.
4. Consider both accuracy AND efficiency.
5. If generating multiple candidates, make them DIVERSE.

{steering}"""


class AnthropicAPIProposer(BaseProposer):
    """Uses Anthropic Messages API with tool use for filesystem access + evo-query."""

    TOOLS: list[dict[str, Any]] = [
        {
            "name": "read_file",
            "description": "Read a file from the candidates directory.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path from .evo/ (e.g., 'candidates/001/scores.json')",
                    }
                },
                "required": ["path"],
            },
        },
        {
            "name": "list_directory",
            "description": "List contents of a directory.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path from .evo/",
                    },
                    "pattern": {"type": "string", "description": "Optional glob pattern"},
                },
                "required": ["path"],
            },
        },
        {
            "name": "search_files",
            "description": "Search for a regex pattern across files (like grep).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {
                        "type": "string",
                        "description": "Directory to search in (relative from .evo/)",
                    },
                    "file_pattern": {
                        "type": "string",
                        "description": "File glob, e.g. '*.py'",
                    },
                },
                "required": ["pattern"],
            },
        },
        {
            "name": "write_harness_file",
            "description": "Write a file for the new harness candidate.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path within harness (e.g., 'agent.py')",
                    },
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
        {
            "name": "submit_candidate",
            "description": "Submit the new harness candidate for evaluation.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string"},
                    "parent_id": {"type": "string"},
                    "strategy_tag": {"type": "string"},
                },
                "required": ["reasoning", "parent_id"],
            },
        },
        {
            "name": "evo_query_leaderboard",
            "description": "Get ranked leaderboard of all candidates.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string"},
                    "top": {"type": "integer"},
                },
            },
        },
        {
            "name": "evo_query_frontier",
            "description": "Get current Pareto frontier.",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "evo_query_failures",
            "description": "List failed tasks for a candidate.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "string"},
                    "metric": {"type": "string"},
                    "top": {"type": "integer"},
                },
                "required": ["candidate_id"],
            },
        },
        {
            "name": "evo_query_task_matrix",
            "description": "Per-task score matrix across candidates.",
            "input_schema": {
                "type": "object",
                "properties": {"metric": {"type": "string"}},
            },
        },
        {
            "name": "evo_query_diff",
            "description": "Diff harness code between two candidates.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "candidate_a": {"type": "string"},
                    "candidate_b": {"type": "string"},
                },
                "required": ["candidate_a", "candidate_b"],
            },
        },
        {
            "name": "evo_query_lineage",
            "description": "Show ancestry of a candidate.",
            "input_schema": {
                "type": "object",
                "properties": {"candidate_id": {"type": "string"}},
                "required": ["candidate_id"],
            },
        },
        {
            "name": "evo_query_grep",
            "description": "Search across traces and code for a pattern.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "scope": {"type": "string"},
                    "candidates": {"type": "string"},
                },
                "required": ["pattern"],
            },
        },
    ]

    _MAX_SEARCH_MATCHES = 200
    _MAX_SEARCH_OUTPUT_CHARS = 256_000
    _MAX_FILE_READ_SEARCH = 512_000

    def __init__(self, model: str = "claude-sonnet-4-20250514", max_turns: int = 50) -> None:
        super().__init__()
        self.model = model
        self.max_turns = max_turns
        self.client = anthropic.Anthropic()

    def _evo_root(self, path: Path, evo_dir: Path) -> tuple[Path | None, str | None]:
        resolved = path.resolve()
        base = evo_dir.resolve()
        try:
            resolved.relative_to(base)
        except ValueError:
            return None, "Error: path traversal not allowed"
        return resolved, None

    async def propose(
        self,
        history_dir: Path,
        frontier: list[dict],
        leaderboard: list[dict],
        task_descriptions: list[str],
        config: dict,
        iteration: int = 0,
        max_iterations: int = 50,
        cost_used: float = 0.0,
        max_cost: float = 50.0,
        steering: str | None = None,
        candidates_per_iteration: int = 1,
    ) -> list[ProposerResult]:
        self._reset_access_log()
        evo_dir = history_dir.parent
        staging: dict[str, str] = {}
        results: list[ProposerResult] = []

        system = self._build_system_prompt(
            config=config,
            leaderboard=leaderboard,
            frontier=frontier,
            task_descriptions=task_descriptions,
            history_dir=history_dir,
            iteration=iteration,
            max_iterations=max_iterations,
            cost_used=cost_used,
            max_cost=max_cost,
            steering=steering,
        )

        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": (
                    f"Please propose {candidates_per_iteration} new harness candidate(s). "
                    "Follow your process carefully."
                ),
            }
        ]
        total_input_tokens = 0
        total_output_tokens = 0

        for _ in range(self.max_turns):
            try:
                response = await asyncio.to_thread(
                    lambda: self.client.messages.create(
                        model=self.model,
                        max_tokens=16384,
                        system=system,
                        tools=self.TOOLS,
                        messages=messages,
                    )
                )
            except anthropic.APIError as e:
                logger.warning("Anthropic API error: %s", e)
                break

            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens

            has_tool_use = any(
                getattr(block, "type", None) == "tool_use" for block in response.content
            )
            if not has_tool_use:
                break

            tool_results: list[dict[str, Any]] = []
            for block in response.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
                name = block.name
                raw_input = block.input
                input_data: dict[str, Any] = dict(raw_input) if isinstance(raw_input, Mapping) else {}
                result = self._handle_tool(name, input_data, evo_dir, history_dir, staging)

                if name == "submit_candidate":
                    results.append(
                        ProposerResult(
                            harness_files=dict(staging),
                            reasoning=input_data.get("reasoning", ""),
                            parent_id=input_data.get("parent_id"),
                            strategy_tag=input_data.get("strategy_tag"),
                            tokens_used=total_input_tokens + total_output_tokens,
                            cost_usd=self._estimate_cost(total_input_tokens, total_output_tokens),
                            access_log=self._reset_access_log(),
                        )
                    )
                    staging = {}

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    }
                )

            if not tool_results:
                break

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

            if len(results) >= candidates_per_iteration:
                break

        if staging and len(results) < candidates_per_iteration:
            results.append(
                ProposerResult(
                    harness_files=staging,
                    reasoning=(
                        "Auto-submitted: proposer wrote files but did not call submit_candidate"
                    ),
                    tokens_used=total_input_tokens + total_output_tokens,
                    cost_usd=self._estimate_cost(total_input_tokens, total_output_tokens),
                    access_log=self._reset_access_log(),
                )
            )

        return results

    def _handle_tool(
        self,
        name: str,
        input_data: dict[str, Any],
        evo_dir: Path,
        _history_dir: Path,
        staging: dict[str, str],
    ) -> str:
        self._log_access(name, args=input_data)

        try:
            if name == "read_file":
                return self._tool_read_file(evo_dir, str(input_data["path"]))
            if name == "list_directory":
                return self._tool_list_directory(
                    evo_dir,
                    str(input_data["path"]),
                    input_data.get("pattern"),
                )
            if name == "search_files":
                return self._tool_search_files(evo_dir, input_data)
            if name == "write_harness_file":
                path = str(input_data["path"])
                content = str(input_data["content"])
                staging[path] = content
                return f"Written: {path} ({len(content.encode('utf-8'))} bytes)"
            if name == "submit_candidate":
                return "Candidate submitted for evaluation."
            if name.startswith("evo_query_"):
                return self._tool_evo_query(name, input_data, evo_dir)
            return f"Unknown tool: {name}"
        except KeyError as e:
            return f"Error: missing required field {e}"
        except Exception as e:
            logger.exception("Tool %s failed", name)
            return f"Error: {e}"

    def _tool_read_file(self, evo_dir: Path, rel_path: str) -> str:
        full, err = self._evo_root(evo_dir / rel_path, evo_dir)
        if err:
            return err
        assert full is not None
        if not full.exists():
            return f"Error: file not found: {rel_path}"
        if full.is_dir():
            return f"Error: path is a directory: {rel_path}"
        try:
            size = full.stat().st_size
        except OSError as e:
            return f"Error: cannot stat file: {e}"
        if size > 1_000_000:
            text = full.read_text(errors="replace")
            return text[:500_000] + "\n...[truncated]..."
        return full.read_text(errors="replace")

    def _tool_list_directory(
        self,
        evo_dir: Path,
        rel_path: str,
        pattern: str | None,
    ) -> str:
        full, err = self._evo_root(evo_dir / rel_path, evo_dir)
        if err:
            return err
        assert full is not None
        if not full.exists():
            return f"Error: path not found: {rel_path}"
        if not full.is_dir():
            return f"Error: not a directory: {rel_path}"
        try:
            children = sorted(full.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError as e:
            return f"Error: cannot list directory: {e}"
        lines: list[str] = []
        for p in children:
            if pattern and not fnmatch.fnmatch(p.name, pattern):
                continue
            kind = "d" if p.is_dir() else "f"
            lines.append(f"{kind}\t{p.name}")
        return "\n".join(lines) if lines else "(empty)"

    def _tool_search_files(self, evo_dir: Path, input_data: Mapping[str, Any]) -> str:
        raw_pattern = input_data.get("pattern")
        if not raw_pattern or not isinstance(raw_pattern, str):
            return "Error: pattern is required and must be a string"
        try:
            regex = re.compile(raw_pattern)
        except re.error as e:
            return f"Error: invalid regex: {e}"

        rel_root = input_data.get("path")
        if rel_root is None or rel_root == "":
            rel_root = "."
        elif not isinstance(rel_root, str):
            return "Error: path must be a string"

        file_pattern = input_data.get("file_pattern")
        if file_pattern is not None and not isinstance(file_pattern, str):
            return "Error: file_pattern must be a string"
        if not file_pattern:
            file_pattern = "*"

        root, err = self._evo_root(evo_dir / rel_root, evo_dir)
        if err:
            return err
        assert root is not None
        if not root.exists():
            return f"Error: path not found: {rel_root}"
        if not root.is_dir():
            return f"Error: not a directory: {rel_root}"

        matches: list[str] = []
        out_len = 0
        truncated = False

        try:
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                if not fnmatch.fnmatch(path.name, file_pattern):
                    continue
                try:
                    st = path.stat()
                except OSError:
                    continue
                if st.st_size > 2_000_000:
                    continue
                try:
                    text = path.read_text(errors="replace")
                except OSError:
                    continue
                if len(text) > self._MAX_FILE_READ_SEARCH:
                    text = text[: self._MAX_FILE_READ_SEARCH] + "\n...[truncated for search]..."

                for i, line in enumerate(text.splitlines(), start=1):
                    if regex.search(line):
                        rel_evo = path.resolve().relative_to(evo_dir.resolve())
                        piece = f"{rel_evo}:{i}:{line}"
                        if out_len + len(piece) + 1 > self._MAX_SEARCH_OUTPUT_CHARS:
                            truncated = True
                            break
                        matches.append(piece)
                        out_len += len(piece) + 1
                        if len(matches) >= self._MAX_SEARCH_MATCHES:
                            truncated = True
                            break
                    if truncated:
                        break
                if truncated:
                    break
        except OSError as e:
            return f"Error: search failed: {e}"

        if not matches:
            return "No matches."
        footer = ""
        if truncated:
            footer = (
                f"\n...[truncated: max {self._MAX_SEARCH_MATCHES} matches or "
                f"{self._MAX_SEARCH_OUTPUT_CHARS} chars]"
            )
        return "\n".join(matches) + footer

    def _tool_evo_query(self, tool_name: str, input_data: dict[str, Any], evo_dir: Path) -> str:
        try:
            from evoharness.experience_cli.evo_query import (
                query_diff,
                query_failures,
                query_frontier,
                query_grep,
                query_leaderboard,
                query_lineage,
                query_task_matrix,
            )
        except ImportError as e:
            return f"Error: evo_query module not available: {e}"

        try:
            if tool_name == "evo_query_leaderboard":
                metric = input_data.get("metric")
                top = input_data.get("top")
                return query_leaderboard(
                    evo_dir,
                    **{k: v for k, v in (("metric", metric), ("top", top)) if v is not None},
                )
            if tool_name == "evo_query_frontier":
                return query_frontier(evo_dir)
            if tool_name == "evo_query_failures":
                cid = input_data.get("candidate_id")
                if not cid:
                    return "Error: candidate_id is required"
                metric = input_data.get("metric")
                top = input_data.get("top")
                kwargs: dict[str, Any] = {}
                if metric is not None:
                    kwargs["metric"] = metric
                if top is not None:
                    kwargs["top"] = top
                return query_failures(evo_dir, str(cid), **kwargs)
            if tool_name == "evo_query_task_matrix":
                metric = input_data.get("metric")
                if metric is not None:
                    return query_task_matrix(evo_dir, metric=metric)
                return query_task_matrix(evo_dir)
            if tool_name == "evo_query_diff":
                a = input_data.get("candidate_a")
                b = input_data.get("candidate_b")
                if not a or not b:
                    return "Error: candidate_a and candidate_b are required"
                return query_diff(evo_dir, str(a), str(b))
            if tool_name == "evo_query_lineage":
                cid = input_data.get("candidate_id")
                if not cid:
                    return "Error: candidate_id is required"
                return query_lineage(evo_dir, str(cid))
            if tool_name == "evo_query_grep":
                pat = input_data.get("pattern")
                if not pat:
                    return "Error: pattern is required"
                scope = input_data.get("scope")
                candidates = input_data.get("candidates")
                kwargs = {}
                if scope is not None:
                    kwargs["scope"] = scope
                if candidates is not None:
                    kwargs["candidates"] = candidates
                return query_grep(evo_dir, str(pat), **kwargs)
        except TypeError as e:
            return f"Error: invalid arguments for {tool_name}: {e}"
        except Exception as e:
            logger.exception("evo_query %s failed", tool_name)
            return f"Error: {e}"

        return f"Error: unhandled evo query tool: {tool_name}"

    def _build_system_prompt(
        self,
        *,
        config: Any,
        leaderboard: list[dict],
        frontier: list[dict],
        task_descriptions: list[str],
        history_dir: Path,
        iteration: int,
        max_iterations: int,
        cost_used: float,
        max_cost: float,
        steering: str | None,
    ) -> str:
        cfg = self._config_as_dict(config)
        scoring = cfg.get("scoring") or {}
        primary_metric = scoring.get("primary_metric", "accuracy")
        secondary = scoring.get("secondary") or []
        if isinstance(secondary, list):
            secondary_metrics = ", ".join(
                f"{s.get('name', '?')} ({s.get('direction', 'minimize')})"
                if isinstance(s, dict)
                else str(s)
                for s in secondary
            )
        else:
            secondary_metrics = str(secondary)
        if not secondary_metrics:
            secondary_metrics = "(none configured)"

        harness = cfg.get("harness") or {}
        mutable = harness.get("mutable_files") or ["harness/*.py"]
        readonly = harness.get("readonly_files") or []
        mutable_files = ", ".join(str(x) for x in mutable) if mutable else "(none)"
        readonly_files = ", ".join(str(x) for x in readonly) if readonly else "(none)"

        leaderboard_text = json.dumps(leaderboard, indent=2, default=str)
        frontier_text = json.dumps(frontier, indent=2, default=str)
        tasks_text = "\n".join(f"- {d}" for d in task_descriptions) if task_descriptions else "(none)"

        steering_block = steering.strip() if steering else ""

        return PROPOSER_SYSTEM_PROMPT.format(
            primary_metric=primary_metric,
            secondary_metrics=secondary_metrics,
            history_dir=str(history_dir.resolve()),
            leaderboard=leaderboard_text,
            frontier=frontier_text,
            iteration=iteration,
            max_iterations=max_iterations,
            cost_used=cost_used,
            max_cost=max_cost,
            task_descriptions=tasks_text,
            mutable_files=mutable_files,
            readonly_files=readonly_files,
            steering=steering_block,
        )

    def _config_as_dict(self, config: Any) -> dict[str, Any]:
        if config is None:
            return {}
        if isinstance(config, Mapping):
            return dict(config)
        dump = getattr(config, "model_dump", None)
        if callable(dump):
            return dump(mode="python")
        return {}

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * 3 + output_tokens * 15) / 1_000_000
