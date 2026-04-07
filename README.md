# EvoHarness

**Automatically improve LLM agent harnesses through evolutionary search.**

EvoHarness is an open-source platform that evolves the *harness* code surrounding a language model — the prompts, context management, retrieval logic, and tool-use patterns — by running an automated search loop. Given an agent's code and an evaluation suite, EvoHarness uses a coding agent (the "proposer") to iteratively propose, evaluate, and refine harness code until it finds configurations that maximize your chosen metrics.

## Inspiration

EvoHarness is inspired by the [Meta-Harness](https://arxiv.org/abs/2603.28052) paper (Lee et al., 2026), which demonstrated that evolutionary search over agent harness code can yield significant performance improvements. While the paper focused on a research prototype, EvoHarness aims to be a general-purpose, developer-friendly tool that anyone can use to optimize their LLM agents.

The core insight from the paper: the code *around* a language model often matters as much as the model itself. Small changes to prompt structure, context windowing, or retrieval strategies can produce outsized performance gains — but the search space is too large for manual tuning. EvoHarness automates that search.

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│                    Search Loop                          │
│                                                         │
│  1. Proposer agent reads history of all past attempts   │
│     (code, scores, execution traces, diffs)             │
│                                                         │
│  2. Proposer diagnoses failures, forms a hypothesis,    │
│     and writes new harness code                         │
│                                                         │
│  3. Validator checks syntax, imports, and interface     │
│                                                         │
│  4. Evaluator runs the new harness against eval tasks   │
│     in a sandboxed environment, capturing full traces   │
│                                                         │
│  5. Results are stored to the filesystem and the        │
│     Pareto frontier is updated                          │
│                                                         │
│  6. Repeat until budget, patience, or iteration limit   │
│     is reached                                          │
└─────────────────────────────────────────────────────────┘
```

### Key Design Principles

- **Filesystem-first feedback** — All history is stored as flat files. The proposer accesses them via standard tools (grep, cat, diff) rather than compressing everything into a single prompt. This scales to millions of tokens of diagnostic information.
- **Minimal scaffolding** — The outer loop is deliberately simple. Diagnosis and proposal logic are delegated entirely to the proposer agent. As coding agents improve, the platform improves automatically.
- **Pluggable everything** — Bring your own agent template, evaluation tasks, scoring functions, and proposer model. The platform orchestrates the loop.
- **Multi-objective optimization** — Track accuracy, token usage, latency, cost, or any custom metric. EvoHarness maintains a Pareto frontier across all objectives.

## Installation

### Prerequisites

- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com/) (for the default proposer)

### Install from source

```bash
git clone https://github.com/ASR4/evo-harness.git
cd evo-harness
pip install -e ".[dev]"
```

This installs the `evo` and `evo-query` CLI tools.

### Verify installation

```bash
evo version
```

## Quick Start

### 1. Initialize a project

```bash
mkdir my-agent && cd my-agent
evo init --template basic --name "my-agent"
```

This scaffolds:
- `evo.toml` — project configuration (metrics, budget, proposer settings)
- `harness/agent.py` — the agent code that will be evolved (with `EVO:MUTABLE` markers for editable sections)
- `evals/eval_suite.py` — your evaluation suite skeleton

### 2. Configure your evaluation

Edit `evo.toml` to set your metrics, budget, and search parameters:

```toml
[project]
name = "my-agent"

[harness]
template = "harness/agent.py"

[eval]
script = "evals/eval_suite.py"
function = "evaluate"
search_tasks = 20
task_timeout = 300

[scoring]
primary_metric = "accuracy"
direction = "maximize"

[search]
max_iterations = 50
proposer = "anthropic:claude-sonnet-4-20250514"
patience = 10

[search.budget]
max_cost_usd = 50.0
```

Edit `evals/eval_suite.py` to define your evaluation tasks and scoring logic. Your eval suite must expose:
- `get_tasks(split)` — returns a list of `EvalTask` objects
- `evaluate(harness_module, task, trace_callback)` — runs one task and returns an `EvalResult`

Your harness must expose a `run(input_data, trace_callback, **kwargs)` function.

### 3. Set your API key

```bash
export ANTHROPIC_API_KEY="your-key-here"
```

### 4. Run the search

```bash
evo run
```

Options:
- `--iterations N` — override max iterations
- `--budget N` — override max cost in USD
- `--proposer MODEL` — override the proposer model
- `--steer "focus on reducing context length"` — give the proposer a specific direction
- `--resume` — resume a previously stopped search
- `--verbose` — enable detailed logging

### 5. Monitor progress

While the search runs, use these commands in another terminal:

```bash
evo status              # Budget, frontier, leaderboard
evo inspect 005         # Details for candidate 005
evo inspect 005 --diff  # Show diff from parent candidate
evo compare 003 007     # Side-by-side diff of two candidates
evo report              # Generate a markdown summary
evo report -o report.md # Save report to file
```

## CLI Reference

### `evo` — Main CLI

| Command | Description |
|---------|-------------|
| `evo init` | Scaffold a new EvoHarness project |
| `evo run` | Start the evolutionary search loop |
| `evo status` | Show current search state (budget, frontier, leaderboard) |
| `evo inspect <id>` | Show details for a specific candidate |
| `evo compare <a> <b>` | Diff two candidates' harness code |
| `evo report` | Generate a search summary report |
| `evo version` | Print version |

### `evo-query` — Experience Store Query CLI

A separate tool for querying the search history, useful for interactive exploration:

| Command | Description |
|---------|-------------|
| `evo-query leaderboard` | Ranked list of candidates by metric |
| `evo-query frontier` | Current Pareto frontier |
| `evo-query diff <a> <b>` | Unified diff between two candidates |
| `evo-query failures <id>` | Show failed tasks with trace error hints |
| `evo-query grep <pattern>` | Search across traces or code |
| `evo-query lineage <id>` | Show the ancestry chain of a candidate |
| `evo-query task-matrix` | Per-task score grid across candidates |

## Project Structure

After running a search, your project will look like this:

```
my-agent/
├── evo.toml                    # Project configuration
├── harness/
│   └── agent.py                # The harness code being evolved
├── evals/
│   └── eval_suite.py           # Your evaluation suite
└── .evo/                       # Search artifacts (auto-generated)
    ├── candidates/
    │   ├── 000/                 # Baseline candidate
    │   │   ├── metadata.json
    │   │   ├── harness/         # Snapshot of harness code
    │   │   ├── scores.json
    │   │   ├── summary.txt
    │   │   └── traces/          # Execution traces (JSONL)
    │   ├── 001/
    │   │   ├── diff_from_parent.patch
    │   │   └── ...
    │   └── ...
    ├── frontier.json            # Current Pareto frontier
    ├── lineage.json             # Parent-child relationships
    ├── cost_tracker.json        # Running cost totals
    └── search_log.jsonl         # Append-only event log
```

## Architecture

```
evoharness/
├── cli/                  # Typer CLI (evo init, run, status, inspect, compare, report)
├── core/
│   ├── loop.py           # Main search loop orchestrator
│   ├── evaluator.py      # Sandboxed evaluation runner
│   ├── frontier.py       # Pareto frontier tracker
│   ├── history.py        # Filesystem-based history store
│   ├── candidate.py      # Candidate data model (Pydantic)
│   ├── config.py         # evo.toml schema and loader
│   ├── cost_tracker.py   # Budget tracking
│   ├── lineage.py        # Candidate ancestry graph
│   └── validator.py      # Syntax, import, and interface validation
├── proposers/
│   ├── base.py           # Abstract proposer interface
│   └── anthropic_api.py  # Anthropic Messages API proposer with tool use
├── sandbox/
│   ├── base.py           # Abstract sandbox interface
│   └── subprocess_sandbox.py  # Subprocess-based sandboxed execution
├── experience_cli/
│   └── evo_query.py      # evo-query CLI for history exploration
└── templates/            # Project templates for evo init
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

Run with verbose output:

```bash
pytest -v
```

Run a specific test file:

```bash
pytest tests/test_frontier.py -v
```

The test suite covers:
- **Unit tests** — Pareto frontier logic, config loading/validation, candidate models, cost tracking, history store, lineage graph, validator checks
- **Integration tests** — Full search loop with mock proposer, evaluator with subprocess sandbox

## Writing Your Own Evaluation Suite

Your eval suite needs two functions:

```python
from dataclasses import dataclass
from typing import Any

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
    """Return evaluation tasks. split is 'search' or 'test'."""
    ...

def evaluate(harness_module, task: EvalTask, trace_callback=None) -> EvalResult:
    """Run the harness on one task and return scored results."""
    output = harness_module.run(task.input_data, trace_callback=trace_callback)
    score = your_scoring_logic(output, task.expected)
    return EvalResult(
        task_id=task.task_id,
        scores={"accuracy": score},
        output=output,
    )
```

## Writing Your Harness

Your harness must expose a `run` function:

```python
# harness/agent.py

def run(input_data, trace_callback=None, **kwargs):
    # --- EVO:MUTABLE START ---

    # Everything between these markers can be modified by the proposer.
    # Put your prompt construction, context management, tool use,
    # retrieval logic, and output processing here.

    # --- EVO:MUTABLE END ---
    pass
```

The `EVO:MUTABLE` markers tell the proposer which parts of your code it's allowed to change. Code outside the markers is protected from modification.

## Configuration Reference

See the [plan document](plan_doc.md) for a complete `evo.toml` reference with all available options including secondary metrics, tracing settings, and dashboard configuration.

## Roadmap

- [x] Core search loop with proposer/evaluator/frontier
- [x] Anthropic API proposer with tool use
- [x] Subprocess sandbox for evaluation
- [x] CLI: init, run, status, inspect, compare, report
- [x] Experience query CLI (evo-query)
- [x] Pareto frontier tracking (multi-objective)
- [x] Candidate lineage tracking
- [x] Cost tracking and budget enforcement
- [x] Search resume support
- [ ] Claude Code as proposer
- [ ] Docker sandbox
- [ ] Web dashboard with real-time updates
- [ ] OpenAI proposer
- [ ] Additional templates (coding_agent, rag_agent, classifier)
- [ ] Search report generation with charts
- [ ] PyPI package

## License

[Apache 2.0](LICENSE)
