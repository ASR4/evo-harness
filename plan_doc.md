# EvoHarness — Complete Implementation Specification

## Project Overview

EvoHarness is an open-source platform that automatically improves LLM agent harnesses (the code surrounding a language model that determines what information to store, retrieve, and present to the model) through evolutionary search. It is inspired by the Meta-Harness paper (arxiv 2603.28052) but aims to be a general-purpose, developer-friendly tool rather than a research prototype.

The core idea: given an agent's code and an evaluation suite, EvoHarness uses a coding agent (the "proposer") to iteratively propose, evaluate, and refine harness code. The proposer has full access to the history of all prior candidates — their source code, evaluation scores, and complete execution traces — via a filesystem. This rich feedback enables the proposer to perform causal reasoning about failures rather than optimizing from compressed summaries.

### Key Design Principles

1. **Filesystem-first feedback**: All history is stored as flat files. The proposer accesses them via standard tools (grep, cat, find, diff) rather than ingesting everything into a single prompt. This scales to millions of tokens of diagnostic information.
2. **Minimal scaffolding**: The outer loop is deliberately simple. Diagnosis and proposal logic are delegated entirely to the proposer agent. As coding agents improve, the platform improves automatically.
3. **Pluggable everything**: Users bring their own agent template, evaluation tasks, scoring functions, and proposer model. The platform orchestrates the loop and provides the UI.
4. **Developer experience matters**: A polished CLI, real-time dashboard, trace viewer, and diff inspector differentiate this from a research script.

---

## Architecture Overview

```
evoharness/
├── cli/                     # CLI entry point and commands
│   ├── __init__.py
│   ├── main.py              # Click/Typer CLI app
│   ├── init_cmd.py          # `evo init` — scaffold a new project
│   ├── run_cmd.py           # `evo run` — start the search loop
│   ├── status_cmd.py        # `evo status` — print current best + frontier
│   ├── inspect_cmd.py       # `evo inspect <candidate_id>` — print details
│   └── serve_cmd.py         # `evo serve` — launch the web dashboard
│
├── core/                    # Core search loop engine
│   ├── __init__.py
│   ├── loop.py              # Main search loop orchestrator
│   ├── proposer.py          # Proposer agent interface + implementations
│   ├── evaluator.py         # Evaluation runner (sandboxed execution)
│   ├── history.py           # History store (filesystem read/write)
│   ├── frontier.py          # Pareto frontier tracker
│   ├── candidate.py         # Candidate data model
│   └── config.py            # Project configuration schema
│
├── sandbox/                 # Sandboxed execution environments
│   ├── __init__.py
│   ├── docker_sandbox.py    # Docker-based sandbox (primary)
│   ├── subprocess_sandbox.py # Subprocess-based sandbox (lightweight)
│   └── base.py              # Abstract sandbox interface
│
├── proposers/               # Proposer agent implementations
│   ├── __init__.py
│   ├── claude_code.py       # Claude Code as proposer (via subprocess)
│   ├── anthropic_agent.py   # Direct Anthropic API with tool use
│   ├── openai_agent.py      # OpenAI API with tool use
│   └── base.py              # Abstract proposer interface
│
├── dashboard/               # Web dashboard (React + FastAPI)
│   ├── api/
│   │   ├── __init__.py
│   │   ├── server.py        # FastAPI server
│   │   ├── routes.py        # API routes
│   │   └── websocket.py     # WebSocket for real-time updates
│   └── frontend/            # React app (Vite)
│       ├── src/
│       │   ├── App.tsx
│       │   ├── components/
│       │   │   ├── ParetoPlot.tsx
│       │   │   ├── CandidateList.tsx
│       │   │   ├── DiffViewer.tsx
│       │   │   ├── TraceInspector.tsx
│       │   │   ├── SearchTimeline.tsx
│       │   │   ├── CostTracker.tsx
│       │   │   └── SteeringPanel.tsx
│       │   ├── hooks/
│       │   └── utils/
│       ├── index.html
│       ├── package.json
│       └── vite.config.ts
│
├── templates/               # Project templates for `evo init`
│   ├── basic/               # Minimal template
│   ├── coding_agent/        # SWE-bench style coding agent
│   ├── rag_agent/           # Retrieval-augmented generation agent
│   └── classifier/          # Online text classification
│
├── pyproject.toml
├── README.md
└── LICENSE                  # Apache 2.0
```

---

## Data Model

### Project Configuration (`evo.toml`)

Every EvoHarness project has an `evo.toml` at its root. This is created by `evo init` and edited by the user.

```toml
[project]
name = "my-coding-agent"
description = "Optimizing a coding agent for SWE-bench"

[harness]
# Path to the agent template file. This is the code that gets evolved.
template = "harness/agent.py"

# Sections of the template marked with `# --- EVO:MUTABLE START ---` and
# `# --- EVO:MUTABLE END ---` are the parts the proposer is allowed to modify.
# If no markers exist, the entire file is mutable.
# Additional files the proposer can create/modify (glob patterns):
mutable_files = ["harness/*.py", "harness/prompts/*.txt"]

# Files the proposer can read but NOT modify (context for understanding):
readonly_files = ["harness/utils.py", "harness/types.py"]

[eval]
# Path to the evaluation script. Must expose a function matching the EvalSuite protocol.
script = "evals/eval_suite.py"

# Name of the evaluation function in the script
function = "evaluate"

# Number of eval tasks to run per candidate during search
search_tasks = 20

# Number of eval tasks for final evaluation of Pareto frontier
test_tasks = 100

# Timeout per individual task (seconds)
task_timeout = 300

# Maximum parallel eval tasks
max_parallel = 4

[scoring]
# Primary metric to optimize (must be a key in the score dict returned by eval)
primary_metric = "accuracy"

# Direction: "maximize" or "minimize"
direction = "maximize"

# Secondary metrics for Pareto frontier (optional)
# Each entry is {name, direction}
[[scoring.secondary]]
name = "context_tokens"
direction = "minimize"

[[scoring.secondary]]
name = "latency_ms"
direction = "minimize"

[[scoring.secondary]]
name = "cost_usd"
direction = "minimize"

[search]
# Maximum number of candidates to evaluate
max_iterations = 50

# Proposer model to use
proposer = "claude-code"  # Options: "claude-code", "anthropic:claude-sonnet-4-20250514", "openai:gpt-4o"

# How many top candidates to keep in the active frontier
frontier_size = 10

# Stop early if no improvement after N iterations
patience = 10

# Temperature for proposer (if applicable)
proposer_temperature = 1.0

[search.budget]
# Maximum total cost for the search (USD). Includes both proposer and eval costs.
max_cost_usd = 50.0

# Maximum cost per single proposer call (USD)
max_proposer_cost_usd = 5.0

[tracing]
# What to capture in execution traces
capture_prompts = true
capture_responses = true
capture_tool_calls = true
capture_state_updates = true

# Maximum trace size per task (MB). Truncate if exceeded.
max_trace_size_mb = 10

[dashboard]
port = 8420
host = "localhost"
```

### Candidate Directory Structure

Each evaluated candidate gets a directory under `.evo/candidates/`:

```
.evo/
├── candidates/
│   ├── 001/
│   │   ├── metadata.json        # Timestamp, parent candidate, proposer reasoning summary
│   │   ├── harness/             # Complete copy of the harness code for this candidate
│   │   │   ├── agent.py
│   │   │   └── prompts/
│   │   │       └── system.txt
│   │   ├── scores.json          # { "accuracy": 0.75, "context_tokens": 12000, ... }
│   │   ├── summary.txt          # One-paragraph description of what this candidate tries
│   │   ├── diff_from_parent.patch  # Unified diff from parent candidate
│   │   └── traces/              # One file per eval task
│   │       ├── task_001.jsonl   # JSON Lines: each line is a trace event
│   │       ├── task_002.jsonl
│   │       └── ...
│   ├── 002/
│   │   └── ...
│   └── ...
│
├── frontier.json               # Current Pareto frontier candidate IDs + scores
├── search_log.jsonl            # Append-only log of all search events
├── cost_tracker.json           # Running cost totals
└── config_snapshot.toml        # Copy of evo.toml at search start (immutable during search)
```

### Trace Event Format (`traces/task_XXX.jsonl`)

Each line is a JSON object representing one event in the agent's execution:

```jsonc
// Types of trace events:
{"type": "prompt", "timestamp": "...", "role": "system", "content": "You are a coding agent...", "token_count": 450}
{"type": "prompt", "timestamp": "...", "role": "user", "content": "Fix the bug in auth.py...", "token_count": 1200}
{"type": "response", "timestamp": "...", "content": "I'll start by examining...", "token_count": 800, "model": "claude-sonnet-4-20250514"}
{"type": "tool_call", "timestamp": "...", "tool": "bash", "input": "cat auth.py", "output": "...", "token_count": 2000}
{"type": "tool_call", "timestamp": "...", "tool": "write_file", "input": {"path": "auth.py", "content": "..."}, "output": "File written", "token_count": 500}
{"type": "state_update", "timestamp": "...", "key": "memory", "before": "...", "after": "...", "token_count": 300}
{"type": "error", "timestamp": "...", "error_type": "timeout", "message": "Task exceeded 300s limit"}
{"type": "result", "timestamp": "...", "score": {"accuracy": 1.0, "context_tokens": 8500}, "task_id": "task_001"}
```

### Metadata Format (`metadata.json`)

```jsonc
{
  "candidate_id": "007",
  "created_at": "2026-04-06T14:23:00Z",
  "parent_id": "004",           // null for the initial candidate
  "proposer_model": "claude-sonnet-4-20250514",
  "proposer_reasoning": "Candidate 004 failed on tasks 3, 7, and 12 due to context overflow...",
  "proposer_tokens_used": 45000,
  "proposer_cost_usd": 0.45,
  "eval_cost_usd": 1.20,
  "eval_duration_seconds": 180,
  "strategy_tag": "context_compression"  // Optional user-visible tag
}
```

---

## Core Components — Detailed Specifications

### 1. The Search Loop (`core/loop.py`)

This is the main orchestrator. It runs the following cycle:

```
INITIALIZE:
  1. Load evo.toml configuration
  2. Validate harness template exists and is parseable
  3. Validate eval script exists and exposes the expected function
  4. Create .evo/ directory structure
  5. Copy initial harness as candidate 000 (the baseline)
  6. Evaluate candidate 000 on search tasks
  7. Initialize Pareto frontier with candidate 000

LOOP (until stopping condition):
  8.  Build proposer context:
      - List of all candidate directories with their scores (a simple leaderboard)
      - Path to the .evo/candidates/ directory
      - The current Pareto frontier
      - The eval task descriptions (what the agent is being tested on)
      - Instructions for the proposer (see PROPOSER SYSTEM PROMPT below)

  9.  Call the proposer agent:
      - The proposer is a coding agent with filesystem access to .evo/candidates/
      - It reads whatever prior candidates, traces, and diffs it wants
      - It writes new harness code to a staging directory

  10. Validate the proposed harness:
      - Syntax check (parse the Python files)
      - Import check (can the harness module be imported without errors)
      - Basic smoke test (run on 1-2 tasks with a short timeout)
      - If validation fails: log the failure, increment attempt counter, retry (max 3 retries per iteration)

  11. Evaluate the proposed harness:
      - Run on all search tasks in sandboxed environment
      - Capture full execution traces
      - Compute scores

  12. Store results:
      - Create new candidate directory with all artifacts
      - Generate diff from parent candidate
      - Update Pareto frontier
      - Update cost tracker
      - Append to search log
      - Emit WebSocket event for dashboard

  13. Check stopping conditions:
      - max_iterations reached
      - max_cost_usd exceeded
      - patience exhausted (no frontier improvement in N iterations)
      - User sent stop signal (via CLI or dashboard)

FINALIZE:
  14. Run final evaluation of Pareto frontier candidates on the full test set
  15. Generate search report (see SEARCH REPORT section)
  16. Print summary to console
```

#### Stopping Conditions Logic

```python
def should_stop(state: SearchState, config: Config) -> tuple[bool, str]:
    if state.iteration >= config.search.max_iterations:
        return True, "max_iterations reached"
    if state.total_cost_usd >= config.search.budget.max_cost_usd:
        return True, f"budget exhausted (${state.total_cost_usd:.2f})"
    if state.iterations_since_frontier_update >= config.search.patience:
        return True, f"patience exhausted ({config.search.patience} iterations without improvement)"
    if state.stop_requested:
        return True, "user requested stop"
    return False, ""
```

### 2. The Proposer Agent (`core/proposer.py` + `proposers/`)

The proposer is the brain of the system. It's a coding agent that inspects past results and writes new harness code.

#### Abstract Interface

```python
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass

@dataclass
class ProposerResult:
    harness_files: dict[str, str]  # {relative_path: file_content}
    reasoning: str                  # Free-text explanation of what this candidate tries
    parent_id: str | None          # Which candidate this is based on
    strategy_tag: str | None       # Optional category tag
    tokens_used: int
    cost_usd: float

class BaseProposer(ABC):
    @abstractmethod
    async def propose(
        self,
        history_dir: Path,          # .evo/candidates/
        frontier: list[dict],       # Current Pareto frontier
        leaderboard: list[dict],    # All candidates sorted by primary metric
        task_descriptions: list[str], # What the eval tasks look like
        config: dict,               # Relevant config
        steering: str | None = None  # Optional user steering instruction
    ) -> ProposerResult:
        ...
```

#### Claude Code Proposer Implementation

When using Claude Code as the proposer, we invoke it as a subprocess with a carefully constructed prompt and give it access to the `.evo/candidates/` directory.

```python
class ClaudeCodeProposer(BaseProposer):
    async def propose(self, history_dir, frontier, leaderboard, task_descriptions, config, steering=None):
        # Build the system prompt (see PROPOSER SYSTEM PROMPT below)
        prompt = self._build_prompt(frontier, leaderboard, task_descriptions, config, steering)

        # Invoke Claude Code with filesystem access
        result = await self._invoke_claude_code(
            prompt=prompt,
            allowed_dirs=[str(history_dir), staging_dir],
            max_tokens=config.get("max_proposer_tokens", 100000),
        )

        # Parse the output: Claude Code writes files to the staging directory
        harness_files = self._read_staging_dir(staging_dir)

        return ProposerResult(
            harness_files=harness_files,
            reasoning=result.reasoning,
            parent_id=result.parent_id,
            strategy_tag=result.strategy_tag,
            tokens_used=result.tokens_used,
            cost_usd=result.cost_usd,
        )
```

#### Anthropic API Proposer Implementation

For users who want direct API control, provide a tool-use based proposer.

```python
class AnthropicAgentProposer(BaseProposer):
    """Uses the Anthropic messages API with tool use for filesystem access."""

    TOOLS = [
        {
            "name": "read_file",
            "description": "Read a file from the history directory.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path from .evo/candidates/"}
                },
                "required": ["path"]
            }
        },
        {
            "name": "list_directory",
            "description": "List contents of a directory in the history.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "pattern": {"type": "string", "description": "Optional glob pattern"}
                },
                "required": ["path"]
            }
        },
        {
            "name": "search_files",
            "description": "Search for a pattern across files in the history (like grep).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                    "path": {"type": "string", "description": "Directory to search in"},
                    "file_pattern": {"type": "string", "description": "File glob, e.g. '*.py' or 'scores.json'"}
                },
                "required": ["pattern"]
            }
        },
        {
            "name": "diff_files",
            "description": "Show unified diff between two files.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_a": {"type": "string"},
                    "file_b": {"type": "string"}
                },
                "required": ["file_a", "file_b"]
            }
        },
        {
            "name": "write_harness_file",
            "description": "Write a file for the new harness candidate. Can only write to the staging directory.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path within the harness (e.g., 'agent.py', 'prompts/system.txt')"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"]
            }
        },
        {
            "name": "submit_candidate",
            "description": "Finalize and submit the new harness candidate for evaluation.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string", "description": "Explanation of what this candidate changes and why"},
                    "parent_id": {"type": "string", "description": "ID of the candidate this is based on"},
                    "strategy_tag": {"type": "string", "description": "Short tag categorizing the approach"}
                },
                "required": ["reasoning", "parent_id"]
            }
        }
    ]
```

#### PROPOSER SYSTEM PROMPT

This is the most critical prompt in the entire system. It tells the proposer how to behave.

```
You are an expert AI engineer tasked with improving an LLM agent's harness code.

## Your Goal

You are optimizing harness code — the code that wraps a language model and controls what context it sees, what tools it has, and how its outputs are processed. Your goal is to find harness configurations that maximize the evaluation score.

## What You Have Access To

You have read access to a directory of all previously evaluated harness candidates. Each candidate directory contains:
- `harness/` — the complete harness source code for that candidate
- `scores.json` — evaluation scores (the metrics being optimized)
- `summary.txt` — a description of what that candidate attempted
- `diff_from_parent.patch` — what changed from the parent candidate
- `metadata.json` — timing, cost, parent info
- `traces/` — detailed execution traces from evaluation, one JSONL file per task. Each line is a trace event (prompts, responses, tool calls, state updates, errors, results).

## Current State

{LEADERBOARD — formatted table of all candidates with their scores, sorted by primary metric}

Pareto frontier (best candidates considering all metrics):
{FRONTIER — candidate IDs and their score vectors}

## Evaluation Tasks

The harness is being evaluated on these types of tasks:
{TASK_DESCRIPTIONS}

## Your Process

1. DIAGNOSE: Start by understanding what has been tried and what failed. Read the scores of top candidates. Read traces from failed tasks to understand WHY they failed. Use grep to search for patterns across multiple traces. Compare diffs between successful and unsuccessful candidates.

2. HYPOTHESIZE: Form a specific hypothesis about what change will improve performance. Be precise — "improve the prompt" is too vague. "Add a verification step where the model re-reads its solution before submitting" is specific.

3. IMPLEMENT: Write the new harness code. You can modify any file in the harness directory. Make focused changes — don't rewrite everything unless you have strong evidence the current approach is fundamentally broken.

4. VALIDATE: Before submitting, re-read your code and check for syntax errors, missing imports, and logic bugs. Think about edge cases.

5. SUBMIT: Call submit_candidate with a clear explanation of your changes and reasoning.

## Important Rules

- You MUST inspect at least 3 prior candidates' traces before proposing a new one (unless this is one of the first 3 iterations).
- Focus on HIGH-LEVERAGE changes. A single well-chosen change to context management, prompt structure, or retrieval logic can have an outsized effect.
- Don't just try random variations. Use the traces to build a causal understanding of what drives performance.
- If the last several candidates have been minor variations, consider a more radical departure.
- Track what strategies have been tried (read the summary.txt files) to avoid repeating failed approaches.
- Consider BOTH performance AND efficiency. A candidate that scores 2% higher but uses 10x more tokens may not be worthwhile.

{OPTIONAL: USER STEERING INSTRUCTION}
```

### 3. The Evaluator (`core/evaluator.py`)

The evaluator runs a candidate harness against eval tasks in a sandboxed environment and captures traces.

#### Evaluation Suite Protocol

Users implement this interface:

```python
# evals/eval_suite.py — User implements this

from dataclasses import dataclass
from typing import Any

@dataclass
class EvalTask:
    task_id: str
    description: str          # Human-readable description
    input_data: Any           # Whatever the harness needs (problem statement, files, etc.)
    expected: Any             # Ground truth for scoring (optional if scorer doesn't need it)
    metadata: dict | None = None

@dataclass
class EvalResult:
    task_id: str
    scores: dict[str, float]  # {"accuracy": 1.0, "context_tokens": 8500, ...}
    output: Any               # Whatever the harness produced
    metadata: dict | None = None

def get_tasks(split: str = "search") -> list[EvalTask]:
    """Return evaluation tasks. split is 'search' (used during evolution) or 'test' (final eval)."""
    ...

def evaluate(harness_module, task: EvalTask, trace_callback=None) -> EvalResult:
    """
    Run a single evaluation task.

    Args:
        harness_module: The imported harness module. Must expose a `run(input_data, **kwargs)` function.
        task: The evaluation task.
        trace_callback: Optional callable that accepts trace events (dicts).
                        Call this to emit trace events during execution:
                        trace_callback({"type": "prompt", "role": "user", "content": "...", ...})

    Returns:
        EvalResult with scores dict.
    """
    ...
```

#### Harness Template Protocol

The harness being evolved must expose a standard interface:

```python
# harness/agent.py — User provides initial version, EvoHarness evolves it

def run(input_data, trace_callback=None, **kwargs):
    """
    Execute the agent on a single task.

    Args:
        input_data: Task input (problem statement, files, whatever).
        trace_callback: Optional callable to emit trace events.
        **kwargs: Additional config passed from eval suite.

    Returns:
        The agent's output (answer, solution, etc.)
    """
    # --- EVO:MUTABLE START ---

    # Everything between these markers can be modified by the proposer.
    # This includes prompt construction, context management, tool use,
    # retrieval logic, state management, etc.

    # --- EVO:MUTABLE END ---
    pass
```

#### Sandboxed Execution

```python
class DockerSandbox:
    """Runs harness evaluation in isolated Docker containers."""

    def __init__(self, image: str = "evoharness/sandbox:latest", timeout: int = 300):
        self.image = image
        self.timeout = timeout

    async def run_evaluation(
        self,
        candidate_dir: Path,      # Directory with harness code
        eval_script: Path,        # Path to eval suite
        task: EvalTask,
        trace_output: Path,       # Where to write the trace JSONL
    ) -> EvalResult:
        """
        1. Create a temporary container
        2. Mount the candidate harness code (read-only) and eval script
        3. Mount a writable volume for traces and output
        4. Run the evaluation with timeout
        5. Collect results and traces
        6. Destroy container
        """
        ...

class SubprocessSandbox:
    """Lightweight alternative: runs in a subprocess with resource limits."""
    # For local development / simple cases where Docker is overkill
    ...
```

### 4. The History Store (`core/history.py`)

```python
class HistoryStore:
    """Manages the .evo/candidates/ directory and provides query interfaces."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir / ".evo" / "candidates"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def next_candidate_id(self) -> str:
        """Return the next sequential candidate ID (zero-padded 3 digits)."""
        ...

    def store_candidate(
        self,
        candidate_id: str,
        harness_files: dict[str, str],
        scores: dict[str, float],
        metadata: dict,
        summary: str,
        parent_id: str | None,
        traces: dict[str, list[dict]],  # {task_id: [trace_events]}
    ) -> Path:
        """
        Write all candidate artifacts to the filesystem.
        Generate diff_from_parent.patch if parent_id is provided.
        Return the candidate directory path.
        """
        ...

    def get_candidate(self, candidate_id: str) -> Candidate:
        """Load a candidate's metadata and scores (not full traces — those are read on demand)."""
        ...

    def get_leaderboard(self, metric: str, direction: str = "maximize") -> list[dict]:
        """Return all candidates sorted by the given metric."""
        ...

    def get_all_scores(self) -> list[dict]:
        """Return {candidate_id, scores} for all candidates. Used for frontier computation."""
        ...
```

### 5. Pareto Frontier Tracker (`core/frontier.py`)

```python
from dataclasses import dataclass

@dataclass
class FrontierPoint:
    candidate_id: str
    scores: dict[str, float]

class ParetoFrontier:
    """Tracks the Pareto frontier across multiple objectives."""

    def __init__(self, objectives: list[dict]):
        """
        objectives: [{"name": "accuracy", "direction": "maximize"},
                      {"name": "context_tokens", "direction": "minimize"}]
        """
        self.objectives = objectives
        self.frontier: list[FrontierPoint] = []

    def update(self, candidate_id: str, scores: dict[str, float]) -> bool:
        """
        Add a new candidate. Returns True if it entered the frontier.
        Remove any existing frontier points that are now dominated.
        """
        ...

    def dominates(self, scores_a: dict, scores_b: dict) -> bool:
        """Return True if scores_a dominates scores_b (better on all objectives)."""
        ...

    def to_json(self) -> list[dict]:
        ...

    @classmethod
    def from_json(cls, data: list[dict], objectives: list[dict]) -> "ParetoFrontier":
        ...
```

---

## CLI Specifications

Use **Typer** for the CLI framework. The CLI binary is called `evo`.

### `evo init`

```
Usage: evo init [OPTIONS] [DIRECTORY]

Scaffold a new EvoHarness project.

Arguments:
  DIRECTORY    Project directory [default: .]

Options:
  --template   Template to use [basic|coding_agent|rag_agent|classifier]
  --name       Project name
```

**Behavior:**
1. Create directory structure
2. Copy template files
3. Generate `evo.toml` with sensible defaults
4. Print next steps

### `evo run`

```
Usage: evo run [OPTIONS]

Start the evolutionary search loop.

Options:
  --iterations    Override max_iterations from config
  --budget        Override max_cost_usd from config
  --resume        Resume a previously stopped search
  --proposer      Override proposer model
  --steer TEXT    Steering instruction for the proposer (e.g., "focus on reducing context length")
  --no-dashboard  Don't auto-open the web dashboard
```

**Behavior:**
1. Load and validate config
2. If `--resume`, load existing state; otherwise initialize fresh
3. Launch dashboard server in background (unless `--no-dashboard`)
4. Run the search loop (as described above)
5. Print final results when done

**Console Output During Search:**
```
EvoHarness v0.1.0 — Optimizing "my-coding-agent"
Dashboard: http://localhost:8420

Iteration 1/50 | Budget: $0.00/$50.00
  Proposer: Analyzing baseline candidate 000...
  Proposer: Reading traces from 5 failed tasks...
  Proposer: Generated candidate 001 (strategy: "add verification step")
  Evaluating: ████████████████████ 20/20 tasks
  Scores: accuracy=0.80 (+0.05) | tokens=9200 (-800) | cost=$0.12
  ✓ New Pareto frontier member!

Iteration 2/50 | Budget: $1.82/$50.00
  Proposer: Comparing candidates 000 and 001...
  ...
```

### `evo status`

Print the current search state: iteration count, budget used, Pareto frontier, best candidate scores.

### `evo inspect <candidate_id>`

Print detailed info about a candidate: its metadata, scores, summary, diff from parent, and optionally trace excerpts.

### `evo serve`

Launch only the web dashboard without running a search (useful for inspecting completed searches).

---

## Web Dashboard Specifications

The dashboard is a React + TypeScript app served by a FastAPI backend. It communicates via REST for initial load and WebSocket for real-time updates.

### API Routes

```
GET  /api/config              → Project config
GET  /api/status              → Current search state (iteration, budget, running/stopped)
GET  /api/candidates          → List all candidates with scores and metadata
GET  /api/candidates/:id      → Full candidate details
GET  /api/candidates/:id/diff → Unified diff from parent
GET  /api/candidates/:id/traces          → List of trace files
GET  /api/candidates/:id/traces/:task_id → Full trace for one task
GET  /api/frontier            → Current Pareto frontier
GET  /api/search-log          → Full search event log
GET  /api/costs               → Cost breakdown

POST /api/steer               → Send steering instruction {"instruction": "..."}
POST /api/stop                → Stop the search gracefully
POST /api/resume              → Resume a stopped search

WS   /ws/events               → Real-time search events
```

### Dashboard Pages/Components

#### 1. Overview Page (default)
- **Search progress bar**: iteration X/N, budget $X/$Y, time elapsed
- **Score chart**: Line chart showing primary metric over iterations. X-axis = candidate ID, Y-axis = score. Highlight Pareto frontier members.
- **Pareto plot**: If 2+ objectives, scatter plot with Pareto frontier line highlighted. Each point is a candidate, clickable.
- **Current best**: Card showing the best candidate's scores and summary.
- **Cost tracker**: Breakdown of proposer vs eval costs, projected total.

#### 2. Candidates Page
- **Sortable table**: All candidates with columns for ID, parent, scores (all metrics), strategy tag, timestamp, cost.
- **Click a row** → opens candidate detail panel.
- **Multi-select** → compare mode (see diffs between any two candidates).

#### 3. Candidate Detail Page
- **Scores card**: All metrics, comparison to parent, comparison to baseline.
- **Summary**: The proposer's reasoning for this candidate.
- **Code viewer**: Syntax-highlighted harness code with diff overlay from parent.
- **Trace inspector**: Select a task → view the full execution trace as a timeline. Each event (prompt, response, tool call, etc.) is a card you can expand. Failed tasks are highlighted.

#### 4. Diff Viewer
- Side-by-side diff between any two candidates' harness code.
- Syntax highlighting.
- Dropdown to select which two candidates to compare.

#### 5. Trace Inspector
- Timeline view of a single evaluation trace.
- Events displayed as cards: system prompt → user prompt → model response → tool call → tool result → ...
- Token counts shown per event.
- Expandable/collapsible.
- Search/filter within trace.
- "Why did this fail?" link that sends the trace to Claude for analysis (optional feature).

#### 6. Steering Panel
- Text input to send steering instructions to the proposer.
- "Pin" a candidate (tell proposer to base next iteration on this one).
- "Focus on failures" button (tell proposer to prioritize tasks where recent candidates scored worst).
- "Go radical" button (tell proposer to try a fundamentally different approach).
- History of steering instructions sent.

---

## Templates

### Basic Template (`templates/basic/`)

```python
# harness/agent.py
import anthropic

client = anthropic.Anthropic()

def run(input_data: str, trace_callback=None, **kwargs) -> str:
    # --- EVO:MUTABLE START ---

    system_prompt = "You are a helpful assistant."

    messages = [{"role": "user", "content": input_data}]

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=system_prompt,
        messages=messages,
    )

    if trace_callback:
        trace_callback({"type": "prompt", "role": "system", "content": system_prompt})
        trace_callback({"type": "prompt", "role": "user", "content": input_data})
        trace_callback({"type": "response", "content": response.content[0].text})

    return response.content[0].text

    # --- EVO:MUTABLE END ---
```

```python
# evals/eval_suite.py
from dataclasses import dataclass
from typing import Any
import json

@dataclass
class EvalTask:
    task_id: str
    description: str
    input_data: Any
    expected: Any
    metadata: dict | None = None

@dataclass
class EvalResult:
    task_id: str
    scores: dict[str, float]
    output: Any
    metadata: dict | None = None

def get_tasks(split: str = "search") -> list[EvalTask]:
    """Load your evaluation tasks here."""
    # Example: load from a JSON file
    with open(f"evals/tasks_{split}.json") as f:
        raw = json.load(f)
    return [
        EvalTask(
            task_id=t["id"],
            description=t["description"],
            input_data=t["input"],
            expected=t["expected"],
        )
        for t in raw
    ]

def evaluate(harness_module, task: EvalTask, trace_callback=None) -> EvalResult:
    """Run the harness on one task and score it."""
    output = harness_module.run(task.input_data, trace_callback=trace_callback)

    # Score it (customize this for your use case)
    correct = output.strip().lower() == str(task.expected).strip().lower()

    return EvalResult(
        task_id=task.task_id,
        scores={"accuracy": 1.0 if correct else 0.0},
        output=output,
    )
```

---

## Error Handling and Edge Cases

### Proposer Failures
- If the proposer produces invalid Python: log the error, retry up to 3 times with the error message appended to the prompt.
- If the proposer exceeds its token/cost budget: terminate the call, log as failed iteration, continue.
- If the proposer produces code identical to an existing candidate: log as duplicate, skip evaluation, ask proposer to try something different.

### Evaluation Failures
- If a harness crashes on a task: score that task as 0, log the traceback in the trace file.
- If a harness times out: kill it, score as 0, log timeout event.
- If ALL tasks fail (score = 0 on everything): still store the candidate (traces are valuable for diagnosis).

### Filesystem Safety
- The proposer has READ-ONLY access to `.evo/candidates/`. It writes to a staging directory.
- Harness code is validated before evaluation (syntax check, import check).
- Trace files are size-limited per config to prevent disk exhaustion.
- All paths are resolved and validated to prevent path traversal attacks.

### Resume Safety
- The search state is reconstructed from the filesystem on resume. No separate state file that can get out of sync.
- If the search was interrupted mid-evaluation, the incomplete candidate directory is cleaned up on resume.
- The config snapshot (`config_snapshot.toml`) prevents config drift between search sessions.

---

## Search Report (Generated at End of Search)

When the search completes, generate `.evo/report.md`:

```markdown
# EvoHarness Search Report
Project: my-coding-agent
Date: 2026-04-06
Duration: 2h 34m
Total cost: $23.45

## Summary
- Iterations completed: 42/50
- Stopped because: patience exhausted (10 iterations without improvement)
- Baseline score: accuracy=0.75
- Best score: accuracy=0.89 (candidate 028)
- Improvement: +14 points (+18.7%)

## Pareto Frontier
| Candidate | Accuracy | Context Tokens | Cost/Task |
|-----------|----------|----------------|-----------|
| 028       | 0.89     | 8,200          | $0.12     |
| 031       | 0.87     | 4,100          | $0.06     |
| 019       | 0.83     | 2,800          | $0.04     |

## Key Strategies Discovered
1. **Draft verification** (candidates 015-028): Adding a step where the model re-reads and verifies its solution before submitting. First appeared in candidate 015, refined through 028.
2. **Context compression** (candidates 019-031): Replacing full conversation history with structured summaries. Reduces token usage by 60% with minimal accuracy loss.
3. **Failure recovery** (candidates 022-028): Detecting when the model is stuck in a loop and resetting its approach.

## Cost Breakdown
- Proposer calls: $8.20 (35%)
- Evaluation runs: $14.50 (62%)
- Final test evaluation: $0.75 (3%)

## Search Trajectory
[Chart: primary metric over iterations with frontier members highlighted]
```

---

## Implementation Priorities

Build in this order:

### Phase 1: Core Loop (Week 1)
1. `core/config.py` — Load and validate `evo.toml`
2. `core/candidate.py` — Data model
3. `core/history.py` — Filesystem read/write
4. `core/frontier.py` — Pareto frontier
5. `core/evaluator.py` — Subprocess sandbox (Docker later)
6. `proposers/anthropic_agent.py` — API-based proposer with tool use
7. `core/loop.py` — Main search loop
8. `cli/main.py` + `cli/run_cmd.py` — Basic `evo run`
9. End-to-end test: optimize a trivial harness (prompt optimization for a classification task)

### Phase 2: CLI Polish (Week 2)
1. `cli/init_cmd.py` — Project scaffolding with templates
2. `cli/status_cmd.py` and `cli/inspect_cmd.py`
3. `templates/basic/` — Starter template
4. `proposers/claude_code.py` — Claude Code integration
5. Better console output (rich/textual for progress bars, tables)
6. Resume logic
7. Cost tracking

### Phase 3: Dashboard (Week 3)
1. `dashboard/api/server.py` — FastAPI server
2. `dashboard/api/routes.py` — REST endpoints
3. `dashboard/api/websocket.py` — Real-time events
4. Frontend: Overview page with score chart + Pareto plot
5. Frontend: Candidate list + detail view
6. Frontend: Diff viewer
7. Frontend: Trace inspector
8. `cli/serve_cmd.py`

### Phase 4: Production Hardening (Week 4)
1. Docker sandbox implementation
2. Proper error handling for all edge cases
3. OpenAI proposer implementation
4. More templates (coding_agent, rag_agent)
5. Search report generation
6. Steering panel in dashboard
7. Documentation and README
8. PyPI packaging

---

## Dependencies

```toml
[project]
name = "evoharness"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "typer>=0.12",
    "rich>=13.0",
    "anthropic>=0.52",
    "openai>=1.30",
    "fastapi>=0.115",
    "uvicorn>=0.30",
    "websockets>=12.0",
    "tomli>=2.0; python_version < '3.12'",  # For TOML parsing
    "tomli-w>=1.0",                          # For TOML writing
    "pydantic>=2.5",
    "httpx>=0.27",
    "watchfiles>=0.21",                      # For filesystem watching
]

[project.optional-dependencies]
docker = ["docker>=7.0"]
dev = ["pytest>=8.0", "pytest-asyncio>=0.23", "ruff>=0.4"]

[project.scripts]
evo = "evoharness.cli.main:app"
```

---

## Testing Strategy

### Unit Tests
- `test_frontier.py` — Pareto dominance logic, frontier updates
- `test_history.py` — Filesystem read/write, leaderboard sorting
- `test_config.py` — Config loading, validation, defaults

### Integration Tests
- `test_loop.py` — Full search loop with a mock proposer and trivial eval
- `test_evaluator.py` — Sandbox execution, trace capture, timeout handling
- `test_proposer.py` — Proposer tool use with a mock filesystem

### End-to-End Tests
- Optimize a simple prompt for a 10-task classification eval
- Verify that scores improve over 5 iterations
- Verify that traces are correctly captured
- Verify that the dashboard API returns correct data

---

## Key Design Decisions and Rationale

1. **Flat files over database**: The paper's key insight is that coding agents work well with filesystem access. A database would add complexity and require a separate query interface. Flat files let the proposer use grep, cat, and diff natively.

2. **JSONL for traces**: Line-delimited JSON allows streaming writes during evaluation and efficient line-by-line reading. The proposer can grep for specific event types without parsing the entire file.

3. **Subprocess sandbox as default, Docker as upgrade**: Most users developing locally don't want to set up Docker. Subprocess with resource limits (via `resource` module on Unix) is good enough for development. Docker is there for production safety and reproducibility.

4. **Proposer as pluggable interface**: Different users will want different proposer models. Some will want Claude Code for maximum capability. Others will want a cheaper model for the outer loop and save budget for evaluation. The interface makes this a config choice.

5. **Mutable markers in code**: The `EVO:MUTABLE` markers let users protect critical code (API clients, type definitions, utilities) while allowing the proposer to modify the parts that matter (prompts, context management, retrieval logic). This prevents the proposer from breaking infrastructure code.

6. **Steering over automation**: The dashboard steering panel acknowledges that human insight is valuable. A user who notices "the agent keeps failing on task 7 because it doesn't read the file first" can tell the proposer to focus on that, saving many iterations.

7. **Apache 2.0 license**: Permissive enough for commercial use, encouraging adoption and contribution.