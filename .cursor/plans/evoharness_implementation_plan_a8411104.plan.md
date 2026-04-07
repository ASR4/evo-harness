---
name: EvoHarness Implementation Plan
overview: A comprehensive, graded review and updated implementation specification for the EvoHarness open-source platform, incorporating all critical lessons from the Meta-Harness paper and addressing gaps in the original plan.
todos:
  - id: phase1-core
    content: "Phase 1: Build core engine -- config, candidate model, history store, frontier, lineage, validator, cost tracker, evaluator, experience CLI, proposer (with evo-query wiring for both backends), proposer access instrumentation, loop (with parallel eval), and evo run command"
    status: pending
  - id: phase1-e2e
    content: "Phase 1 validation: End-to-end test optimizing a trivial classifier over 5 iterations"
    status: pending
  - id: phase2-cli
    content: "Phase 2: CLI polish -- init, status, inspect, compare, report commands with rich terminal output"
    status: pending
  - id: phase2-templates
    content: "Phase 2: Templates and skills -- basic template, classifier template, Claude Code proposer, skill-writing guide"
    status: pending
  - id: phase3-dashboard
    content: "Phase 3 (deferred): Web dashboard with FastAPI backend and React frontend"
    status: pending
  - id: phase4-harden
    content: "Phase 4: Docker sandbox, OpenAI proposer, additional templates, warm-start, PyPI packaging, docs"
    status: pending
isProject: false
---

# EvoHarness -- Complete Implementation Specification (v2.1)

## Grade of Original Plan: B+

### What the original plan gets right

- Excellent project structure and architecture
- Solid data model (evo.toml, candidate directories, trace format, metadata)
- Good CLI specification with sensible commands
- Thoughtful error handling and edge cases
- Clear phased implementation priorities
- Well-reasoned design decisions (flat files, JSONL, pluggable proposer)

### Critical gaps the original plan misses

1. **The proposer system prompt is too generic.** The paper explicitly states that "iterating on the skill text had a larger effect on search quality than changing iteration count or population size." The current prompt doesn't encode the paper's key lessons: causal reasoning over confounds, preferring additive changes after regressions, cross-candidate trace comparison, explicit hypothesis formation. This is the single highest-leverage component.
2. **No experience store CLI for the proposer.** The paper (Appendix D) recommends building a small CLI that the proposer agent can call programmatically to query the experience store (list Pareto frontier, diff between runs, show top-k harnesses). This is distinct from the user-facing `evo` CLI and significantly reduces the tokens the proposer wastes on navigation.
3. **No multi-candidate per iteration.** The paper runs 2-3 candidates per iteration in some settings. The plan assumes 1 candidate per iteration.
4. **Missing proposer filesystem access instrumentation.** The paper tracks exactly what files the proposer reads (median 82 files, 41% source code, 40% traces, 6% scores). This instrumentation is critical for understanding whether the proposer is actually using the history.
5. **Dashboard is over-scoped for MVP.** The paper itself has no dashboard -- it's all filesystem + CLI. A full React + WebSocket dashboard in week 3 is ambitious and risks delaying the core value. The terminal-based rich output is more important.
6. **Missing warm-start from offline experience.** The paper (Appendix D) mentions converting offline experience (rollouts from other models, solved problem corpora) into the same directory structure to warm-start exploration.
7. **Search set vs test set boundary is unclear in the evaluation flow.** The paper is strict: the proposer NEVER sees test-set results. The plan mentions `search_tasks` and `test_tasks` in config but doesn't enforce separation in the evaluation flow.
8. **No guidance on writing domain-specific skills.** The paper emphasizes iterating on the skill text with 3-5 short debug runs before committing to a full search. The plan needs a skill-writing guide.
9. **Missing duplicate detection logic.** The plan mentions detecting duplicates but doesn't specify how.
10. **No structured lineage tracking.** The paper shows that the proposer merges lineages (e.g., combining the best geometry route from one lineage with the best combinatorics route from another). The plan needs explicit lineage/ancestry tracking.

---

## Post-Review Clarifications (v2.1)

Three clarifications incorporated after review approval:

### Clarification 1: How the proposer discovers `evo-query`

The `evo-query` tool must be surfaced differently depending on the proposer backend:

**Anthropic API proposer (`proposers/anthropic_api.py`):** Register each `evo-query` command as a native tool in the tool-use API. The proposer calls `evo_query_leaderboard`, `evo_query_failures`, etc. as structured tool calls. The tool handler executes the corresponding `evo-query` logic in-process (no subprocess needed -- import the functions directly from `experience_cli/`). This gives the proposer structured JSON responses it can parse reliably.

```python
# In anthropic_api.py, add these to the TOOLS list alongside read_file, search_files, etc.
{
    "name": "evo_query_leaderboard",
    "description": "Get ranked leaderboard of all candidates by primary metric.",
    "input_schema": {
        "type": "object",
        "properties": {
            "metric": {"type": "string", "description": "Metric to rank by (default: primary)"},
            "top": {"type": "integer", "description": "Number of top candidates to show (default: 10)"}
        }
    }
},
{
    "name": "evo_query_task_matrix",
    "description": "Get matrix of per-task scores across candidates. Rows=candidates, cols=tasks.",
    "input_schema": {
        "type": "object",
        "properties": {
            "metric": {"type": "string", "description": "Metric to show in cells"},
            "candidates": {"type": "string", "description": "Comma-separated candidate IDs (default: all)"}
        }
    }
},
# ... similar for frontier, diff, failures, grep, lineage
```

**Claude Code proposer (`proposers/claude_code.py`):** Ensure `evo-query` is on PATH in the subprocess environment. Add this line to the proposer system prompt:

```
You have `evo-query` on your PATH. Run `evo-query --help` to see all available
commands. These are much faster than manually navigating the candidates/ directory.
```

The Claude Code subprocess invocation must set `PATH` to include the virtualenv's bin directory where `evo-query` is installed:

```python
env = os.environ.copy()
env["PATH"] = f"{sys.prefix}/bin:{env.get('PATH', '')}"
```

### Clarification 2: Parallel evaluation of multi-candidate batches

When `candidates_per_iteration > 1`, the loop evaluates all candidates from one iteration concurrently using `asyncio.gather`, not sequentially. This is a significant wall-clock savings (2x with 2 candidates).

Updated loop pseudocode for step 11:

```python
# In core/loop.py, the evaluation step:
async def _evaluate_batch(self, candidates: list[Candidate], tasks: list[EvalTask]):
    """Evaluate multiple candidates in parallel."""
    results = await asyncio.gather(
        *[self._evaluate_single(c, tasks) for c in candidates],
        return_exceptions=True,
    )
    for candidate, result in zip(candidates, results):
        if isinstance(result, Exception):
            logger.error(f"Candidate {candidate.id} evaluation failed: {result}")
            self._store_failed_candidate(candidate, result)
        else:
            self._store_candidate(candidate, result)
```

Within `_evaluate_single`, individual tasks are ALSO parallelized up to `config.eval.max_parallel`:

```python
async def _evaluate_single(self, candidate: Candidate, tasks: list[EvalTask]):
    """Evaluate one candidate on all tasks, with bounded parallelism."""
    semaphore = asyncio.Semaphore(self.config.eval.max_parallel)
    async def _run_task(task):
        async with semaphore:
            return await self.sandbox.run_evaluation(candidate, task)
    return await asyncio.gather(*[_run_task(t) for t in tasks])
```

### Clarification 3: Proposer access instrumentation (bumped to Phase 1)

Proposer access instrumentation is trivial to implement and has enormous diagnostic value. Bumped from Phase 4 to Phase 1.

For every proposer invocation, log every file read and tool call to `proposer_access.jsonl` inside the candidate's directory:

```jsonc
// .evo/candidates/007/proposer_access.jsonl
{"timestamp": "...", "action": "read_file", "path": "candidates/003/scores.json", "bytes": 245}
{"timestamp": "...", "action": "read_file", "path": "candidates/003/traces/task_007.jsonl", "bytes": 18400}
{"timestamp": "...", "action": "search_files", "pattern": "timeout", "path": "candidates/", "matches": 12}
{"timestamp": "...", "action": "evo_query", "command": "leaderboard", "args": {"top": 10}}
{"timestamp": "...", "action": "evo_query", "command": "task_matrix", "args": {"metric": "accuracy"}}
{"timestamp": "...", "action": "write_file", "path": "staging/harness/agent.py", "bytes": 3200}
```

**Implementation for Anthropic API proposer:** Wrap every tool handler with a logging decorator:

```python
def _log_access(self, action: str, details: dict):
    self._access_log.append({
        "timestamp": datetime.utcnow().isoformat(),
        "action": action,
        **details,
    })

async def _handle_tool_call(self, tool_name, tool_input):
    self._log_access(tool_name, {"args": tool_input})
    result = await self._tool_handlers[tool_name](tool_input)
    return result
```

**Implementation for Claude Code proposer:** Claude Code doesn't expose individual file reads. Instead, use filesystem watching (`watchfiles`) on the candidates directory during the proposer's execution to detect reads. After the proposer finishes, parse its output/conversation to extract which files were accessed.

After the proposer returns, save the access log:

```python
# In ProposerResult, add:
access_log: list[dict]  # Every file/tool access made during proposal

# In history.py store_candidate(), write it:
(candidate_dir / "proposer_access.jsonl").write_text(
    "\n".join(json.dumps(e) for e in result.access_log)
)
```

This enables post-hoc analysis: "Is the proposer reading traces? How many prior candidates does it inspect? Does it use evo-query or raw file reads?" -- directly mirroring Table 8 from the paper (median 82 files, 41% code, 40% traces).

---

## Updated Architecture

```
evoharness/
├── cli/                     # User-facing CLI
│   ├── __init__.py
│   ├── main.py              # Typer app entry point
│   ├── init_cmd.py          # `evo init` -- scaffold a new project
│   ├── run_cmd.py           # `evo run` -- start the search loop
│   ├── status_cmd.py        # `evo status` -- print frontier + progress
│   ├── inspect_cmd.py       # `evo inspect <id>` -- candidate details
│   ├── compare_cmd.py       # `evo compare <id_a> <id_b>` -- diff two candidates
│   ├── report_cmd.py        # `evo report` -- generate search report
│   └── serve_cmd.py         # `evo serve` -- launch web dashboard
│
├── core/
│   ├── __init__.py
│   ├── loop.py              # Main search loop orchestrator (with parallel eval)
│   ├── proposer.py          # Proposer interface + dispatch
│   ├── evaluator.py         # Evaluation runner
│   ├── history.py           # Filesystem experience store
│   ├── frontier.py          # Pareto frontier tracker
│   ├── candidate.py         # Candidate data model
│   ├── config.py            # Config schema (evo.toml)
│   ├── validator.py         # Harness validation (syntax, import, smoke test)
│   ├── cost_tracker.py      # Budget tracking
│   └── lineage.py           # Ancestry/lineage graph
│
├── experience_cli/          # CLI tools for the PROPOSER to query the store
│   ├── __init__.py
│   ├── evo_query.py         # `evo-query` binary the proposer calls
│   ├── leaderboard.py       # `evo-query leaderboard` -- ranked candidates
│   ├── frontier_cmd.py      # `evo-query frontier` -- current Pareto frontier
│   ├── diff_cmd.py          # `evo-query diff <a> <b>` -- diff any two candidates
│   ├── failures.py          # `evo-query failures <id>` -- summarize failed tasks
│   ├── grep_traces.py       # `evo-query grep <pattern>` -- search across traces
│   ├── task_matrix.py       # `evo-query task-matrix` -- per-task score matrix
│   └── lineage_cmd.py       # `evo-query lineage <id>` -- show ancestry
│
├── sandbox/
│   ├── __init__.py
│   ├── base.py              # Abstract sandbox interface
│   ├── subprocess_sandbox.py # Default: subprocess with resource limits
│   └── docker_sandbox.py    # Optional: Docker-based isolation
│
├── proposers/
│   ├── __init__.py
│   ├── base.py              # Abstract proposer interface (includes access_log)
│   ├── claude_code.py       # Claude Code proposer (subprocess, evo-query on PATH)
│   ├── anthropic_api.py     # Anthropic Messages API (evo-query as native tools)
│   └── openai_api.py        # OpenAI API with tool use
│
├── dashboard/               # Phase 3 -- deferred
│   ├── api/
│   │   ├── server.py
│   │   ├── routes.py
│   │   └── websocket.py
│   └── frontend/
│       └── ...
│
├── templates/               # Project templates for `evo init`
│   ├── basic/
│   ├── classifier/
│   ├── coding_agent/
│   └── rag_agent/
│
├── skills/                  # Skill-writing guide + examples
│   ├── GUIDE.md
│   ├── text_classification.md
│   ├── coding_agent.md
│   └── rag_retrieval.md
│
├── pyproject.toml
├── README.md
└── LICENSE                  # Apache 2.0
```

### Key structural changes from original

- Added `experience_cli/` -- a separate CLI binary (`evo-query`) that the proposer agent calls during its filesystem exploration. Wired as native tools for API proposers and as a PATH binary for Claude Code.
- Added `core/validator.py` -- explicit validation pipeline
- Added `core/lineage.py` -- ancestry graph tracking
- Added `core/cost_tracker.py` -- separated from main loop
- Added `skills/GUIDE.md` -- how to write proposer skills
- Added `cli/compare_cmd.py` and `cli/report_cmd.py`
- Proposer access instrumentation built into `proposers/base.py` from day one

---

## Data Model (Unchanged from Original -- These Are Correct)

The original plan's data model is solid. Retain all of:

- `evo.toml` config schema (as specified in [plan_doc.md](plan_doc.md) lines 96-190)
- Candidate directory structure (lines 196-220)
- Trace event JSONL format (lines 226-236)
- Metadata format (lines 240-253)

### Addition: Multi-candidate per iteration support

Add to `evo.toml`:

```toml
[search]
candidates_per_iteration = 2  # Number of candidates the proposer generates per iteration
```

The proposer is instructed to produce N candidates per iteration. Each is independently evaluated (in parallel via asyncio.gather) and stored.

### Addition: Lineage graph (`.evo/lineage.json`)

```jsonc
{
  "nodes": {
    "000": {"parents": [], "children": ["001", "002"]},
    "001": {"parents": ["000"], "children": ["003"]},
    "002": {"parents": ["000"], "children": ["003"]},
    "003": {"parents": ["001", "002"], "children": []}
  }
}
```

### Addition: Proposer access log (`.evo/candidates/NNN/proposer_access.jsonl`)

Every file read, tool call, and evo-query invocation the proposer makes during its proposal is logged. See Clarification 3 above for format and implementation details.

---

## CRITICAL COMPONENT: The Proposer System Prompt

This is the most important artifact in the entire system. The original plan's prompt is too generic. Below is a thorough prompt that encodes the paper's empirical lessons.

```
You are an expert harness engineer. Your job is to improve LLM agent harness
code -- the code that wraps a language model and controls what context it sees,
what tools it has, how it manages state, and how its outputs are processed.

## Your Goal

Maximize {PRIMARY_METRIC} on the evaluation tasks, while considering
secondary objectives: {SECONDARY_METRICS}.

## What You Have Access To

You have read access to a directory of ALL previously evaluated harness
candidates at: {HISTORY_DIR}

Each candidate directory (e.g., candidates/007/) contains:
  - harness/           Full harness source code
  - scores.json        Evaluation scores
  - summary.txt        What this candidate attempted
  - diff_from_parent.patch  Changes from parent candidate
  - metadata.json      Timing, cost, parent info
  - traces/            One JSONL file per eval task (prompts, responses,
                       tool calls, state updates, errors)

You also have a query tool. Use it to save time:
  - `evo-query leaderboard`          Ranked candidates by {PRIMARY_METRIC}
  - `evo-query frontier`             Current Pareto frontier
  - `evo-query diff <id_a> <id_b>`   Diff code between any two candidates
  - `evo-query failures <id>`        List failed tasks with error summaries
  - `evo-query grep <pattern>`       Search across all traces and code
  - `evo-query lineage <id>`         Show ancestry graph
  - `evo-query task-matrix`          Per-task score matrix across candidates

You have `evo-query` on your PATH. Run `evo-query --help` to see all available
commands. These are much faster than manually navigating the candidates/ directory.

## Current State

{LEADERBOARD}

Pareto frontier:
{FRONTIER}

Iteration: {ITERATION} / {MAX_ITERATIONS}
Budget used: ${COST_USED} / ${MAX_COST}

## Evaluation Tasks

The harness is being evaluated on:
{TASK_DESCRIPTIONS}

## Your Process -- FOLLOW THIS CAREFULLY

### Step 1: DIAGNOSE (spend at least 40% of your effort here)

Do NOT jump to proposing changes. First build understanding:

a) Run `evo-query leaderboard` and `evo-query task-matrix`. Identify the top 3
   candidates and which specific tasks they fail on.
b) Run `evo-query failures <id>` for the top candidates. Understand WHICH TASKS
   they fail on and the error types.
c) Read execution traces for failed tasks. Understand WHY they failed.
   Look for: context overflow, wrong retrieval, missing information,
   hallucinations, tool misuse, state corruption, timeout.
d) Compare traces between a candidate that PASSED a task and one that FAILED
   the same task. What was different?
e) Run `evo-query diff` between recent candidates. Understand what changes were
   tried and whether they helped or hurt.
f) Check if recent regressions share a common factor (a confound). The paper
   shows that bundling structural fixes with prompt edits often causes
   regressions where the prompt change is the actual culprit.

### Step 2: HYPOTHESIZE (be specific and causal)

Form a SPECIFIC, FALSIFIABLE hypothesis. Bad: "improve the prompt."
Good: "Tasks 3, 7, and 12 fail because the context window overflows at step 8,
causing the model to lose the original instructions. Truncating tool output to
2000 chars should prevent overflow while preserving the key information."

Write your hypothesis in the summary. This is critical for future iterations
(including yours) to understand what was tried and why.

### Step 3: DESIGN CHANGES (follow safety principles)

CRITICAL LESSONS from prior harness search research:

a) PREFER ADDITIVE over subtractive changes. Adding a new capability
   (e.g., environment bootstrapping, verification step) is safer than
   removing or rewriting existing logic that may have non-obvious effects.

b) ISOLATE your changes. Do NOT bundle multiple independent fixes in one
   candidate. If you want to test a retrieval change AND a prompt change,
   make them separate candidates. Bundling creates confounds that make
   diagnosis impossible if the candidate regresses.

c) SMALL, TARGETED edits beat large rewrites. The search space is vast;
   a full rewrite has a low probability of improving on a strong baseline.
   Only do a full rewrite if you have strong evidence the current approach
   is fundamentally broken.

d) After multiple regressions from modifying a specific component (e.g.,
   the completion flow, the prompt template), STOP modifying that component.
   Pivot to a different part of the harness.

e) Consider COMPOSING successful changes. If candidate A improved retrieval
   and candidate B improved prompting, try merging both improvements.
   Use `evo-query lineage` to see what combinations have been tried.

### Step 4: IMPLEMENT

Write the new harness code. You can modify files matching these patterns:
{MUTABLE_FILES}

You CANNOT modify:
{READONLY_FILES}

### Step 5: VALIDATE

Before submitting, verify:
- No syntax errors (mentally trace the code)
- All imports are available
- The harness interface is preserved (the `run()` function signature)
- Edge cases are handled (empty input, timeout, API errors)

### Step 6: SUBMIT

Call submit_candidate with:
- reasoning: Your full diagnosis, hypothesis, and what you changed
- parent_id: Which candidate(s) this is based on
- strategy_tag: A short label (e.g., "context_compression", "verification_step")

## Rules

1. You MUST read traces from at least 3 prior candidates before proposing
   (unless fewer than 3 exist).
2. You MUST state your causal hypothesis in the reasoning field.
3. You MUST NOT modify files outside the mutable patterns.
4. Do NOT repeat a strategy that has already been tried and failed
   (read summary.txt files to check).
5. Consider both accuracy AND efficiency. A +2% accuracy gain that costs
   10x more tokens may not be worthwhile -- check the Pareto frontier.
6. If you are generating {CANDIDATES_PER_ITERATION} candidates this iteration,
   make them DIVERSE. Test different hypotheses, not minor variations of
   the same idea.

{OPTIONAL_STEERING}
```

### Why this prompt is better

- Encodes the paper's empirical lesson about confounds (Iterations 1-3 in Appendix A.2)
- Encodes the "prefer additive changes" lesson (Iteration 7 winning candidate)
- Encodes the "isolate changes" lesson (don't bundle structural + prompt edits)
- Encodes the "compose successful lineages" lesson (Iteration 8)
- Encodes the "pivot after repeated regressions" lesson (Iterations 4-6)
- Requires explicit causal hypotheses (not just "try something different")
- References the experience store CLI tools to reduce wasted navigation tokens
- Explicitly instructs the proposer to use `evo-query` first for structured queries
- Supports multi-candidate per iteration with diversity requirement
- Includes budget awareness

---

## Experience Store CLI (`experience_cli/evo_query.py`)

This is a NEW component not in the original plan. The paper (Appendix D) explicitly recommends it:

> "A short CLI that lists the Pareto frontier, shows top-k harnesses, and diffs code and results between pairs of runs can make the experience store much easier to use, and querying such CLIs is closely aligned with the workflows on which coding agents are trained."

This is a separate binary (`evo-query`) installed alongside `evo`, designed for the proposer agent to call. Commands:

```
evo-query leaderboard [--metric accuracy] [--top 10]
  Prints ranked table: ID | Parent | Accuracy | Tokens | Cost | Strategy

evo-query frontier
  Prints Pareto frontier candidates with all metrics

evo-query diff <candidate_a> <candidate_b>
  Prints unified diff of harness code between two candidates

evo-query failures <candidate_id> [--top 5]
  Lists failed tasks with: task_id, score, error type, first error line

evo-query grep <pattern> [--scope traces|code|all] [--candidates 005,006,007]
  Searches across traces/code for a regex pattern, returns matches with context

evo-query lineage <candidate_id>
  Shows ancestry: parent chain, children, sibling candidates

evo-query task-matrix [--metric accuracy]
  Prints matrix: rows=candidates, cols=tasks, cells=per-task scores
  Instantly shows which candidates pass/fail which tasks
```

Implementation: Each command reads from `.evo/candidates/` and `.evo/frontier.json`. Pure Python, no external dependencies beyond what's in core.

**Dual surfacing (per Clarification 1):**

- For Anthropic API proposer: imported in-process as native tools
- For Claude Code proposer: registered as console script on PATH

```toml
[project.scripts]
evo = "evoharness.cli.main:app"
evo-query = "evoharness.experience_cli.evo_query:app"
```

---

## Validation Pipeline (`core/validator.py`)

Detailed specification (the original plan was underspecified here):

```python
@dataclass
class ValidationResult:
    passed: bool
    stage: str           # "syntax", "import", "interface", "smoke"
    error: str | None
    duration_seconds: float

async def validate_candidate(
    harness_dir: Path,
    eval_module,
    smoke_tasks: list[EvalTask],  # 1-2 easy tasks
    timeout: int = 30,
) -> ValidationResult:
    """
    Four-stage validation before expensive full evaluation:

    1. SYNTAX: Parse all .py files with ast.parse(). Catches syntax errors
       in seconds.

    2. IMPORT: Import the harness module in a subprocess. Catches missing
       imports, circular dependencies, module-level exceptions.
       Timeout: 10 seconds.

    3. INTERFACE: Verify the harness module exposes the expected interface
       (e.g., a `run` function with the right signature). Uses inspect.

    4. SMOKE TEST: Run on 1-2 easy tasks with a short timeout. Catches
       runtime errors, infinite loops, and obviously broken logic.
       Timeout: 30 seconds per task.

    Any stage failure stops validation and returns immediately.
    """
```

---

## Duplicate Detection

Add to `core/history.py`:

```python
def is_duplicate(self, new_files: dict[str, str], threshold: float = 0.95) -> str | None:
    """
    Check if new harness code is too similar to any existing candidate.
    Uses normalized Levenshtein ratio on concatenated source files.
    Returns the candidate_id of the duplicate if found, None otherwise.
    """
```

Threshold of 0.95 means >95% identical code is flagged as duplicate. The proposer is informed and asked to try something different.

---

## Search Set / Test Set Enforcement

The paper is strict: the proposer NEVER sees test results. Enforce this:

```python
class Evaluator:
    def run_search_eval(self, candidate, tasks):
        """Run on search tasks. Results stored in .evo/candidates/. Proposer can see these."""

    def run_test_eval(self, candidate, tasks):
        """Run on test tasks. Results stored in .evo/test_results/ (NOT in candidates/).
        Proposer has NO access to this directory."""
```

In the sandbox configuration, `.evo/test_results/` is excluded from the proposer's filesystem access.

---

## Updated Implementation Phases

### Phase 1: Core Engine (Priority -- build this first)

Files to implement in order:

1. `pyproject.toml` -- Project metadata and dependencies
2. `core/config.py` -- Pydantic model for evo.toml
3. `core/candidate.py` -- Candidate dataclass
4. `core/history.py` -- Filesystem store (write candidates, read leaderboard, duplicate detection)
5. `core/frontier.py` -- Pareto dominance logic
6. `core/lineage.py` -- Ancestry graph
7. `core/validator.py` -- Four-stage validation pipeline
8. `core/cost_tracker.py` -- Budget tracking
9. `core/evaluator.py` -- Subprocess sandbox evaluation with parallel task execution
10. `experience_cli/evo_query.py` -- All 8 query commands (leaderboard, frontier, diff, failures, grep, lineage, task-matrix, help)
11. `proposers/base.py` -- Abstract proposer interface with access logging built in
12. `proposers/anthropic_api.py` -- Anthropic Messages API proposer with evo-query commands registered as native tools
13. `core/loop.py` -- Main search loop with parallel candidate evaluation (asyncio.gather)
14. `cli/main.py` + `cli/run_cmd.py` -- `evo run` command

End-to-end test: optimize a trivial text classification harness (prompt tuning) over 5 iterations.

### Phase 2: CLI + Templates + Claude Code

1. `cli/init_cmd.py` -- Project scaffolding
2. `cli/status_cmd.py` -- Print search state
3. `cli/inspect_cmd.py` -- Candidate details
4. `cli/compare_cmd.py` -- Diff two candidates
5. `cli/report_cmd.py` -- Generate markdown report
6. `templates/basic/` -- Minimal starter template
7. `templates/classifier/` -- Text classification template
8. `proposers/claude_code.py` -- Claude Code as proposer (with evo-query on PATH)
9. Rich console output (progress bars, tables, color)
10. Resume logic
11. `skills/GUIDE.md` -- How to write effective proposer skills

### Phase 3: Dashboard (Deferred -- only after core is solid)

The original plan's dashboard spec is good but should be deferred. A rich terminal UI (using textual or rich) provides 80% of the value at 20% of the effort. Build the full React dashboard only after the core loop is proven.

Interim: `evo status` and `evo inspect` with rich terminal output.

### Phase 4: Hardening + Community

1. Docker sandbox
2. OpenAI proposer
3. `templates/coding_agent/` and `templates/rag_agent/`
4. Warm-start from offline experience
5. PyPI packaging
6. Documentation site
7. Contributing guide

---

## Dependencies

```toml
[project]
name = "evoharness"
version = "0.1.0"
description = "Automatically improve LLM agent harnesses through evolutionary search"
requires-python = ">=3.11"
license = "Apache-2.0"
dependencies = [
    "typer>=0.12",
    "rich>=13.0",
    "anthropic>=0.52",
    "pydantic>=2.5",
    "tomli>=2.0; python_version < '3.12'",
    "tomli-w>=1.0",
]

[project.optional-dependencies]
openai = ["openai>=1.30"]
dashboard = ["fastapi>=0.115", "uvicorn>=0.30", "websockets>=12.0"]
docker = ["docker>=7.0"]
dev = ["pytest>=8.0", "pytest-asyncio>=0.23", "ruff>=0.4"]

[project.scripts]
evo = "evoharness.cli.main:app"
evo-query = "evoharness.experience_cli.evo_query:app"
```

Note: Moved openai, dashboard, and docker to optional dependencies. The core system only requires typer, rich, anthropic, pydantic, and TOML handling.

---

## Testing Strategy (Updated)

### Unit Tests

- `test_frontier.py` -- Pareto dominance, frontier updates, edge cases (single objective, all dominated)
- `test_history.py` -- Write/read candidates, leaderboard, duplicate detection
- `test_config.py` -- Config loading, validation, defaults, missing fields
- `test_validator.py` -- Each validation stage independently
- `test_lineage.py` -- Ancestry graph, multi-parent merge
- `test_cost_tracker.py` -- Budget enforcement

### Integration Tests

- `test_loop.py` -- Full loop with mock proposer, verify candidates are stored correctly, verify parallel evaluation
- `test_evaluator.py` -- Sandbox execution, trace capture, timeout, crash handling
- `test_experience_cli.py` -- All 8 evo-query commands against a fixture store
- `test_proposer_tools.py` -- Tool use flow with mock filesystem, verify access log is captured

### End-to-End Test

- Optimize a 10-question classification prompt over 5 iterations using a cheap model
- Verify: scores improve, traces captured, frontier updated, report generated, evo-query works, proposer_access.jsonl is populated

---

## Key Design Decisions (Updated)

All 7 original design decisions are retained. Adding:

1. **Experience store CLI as a separate binary**: The proposer agent needs a fast, token-efficient way to query the history. Raw filesystem access works but wastes tokens on navigation. The `evo-query` CLI gives the proposer structured summaries it can parse quickly, closely matching the workflows coding agents are trained on. Surfaced as native tools for API proposers and as a PATH binary for Claude Code.
2. **Multi-candidate per iteration with parallel evaluation**: Generating 2-3 diverse candidates per iteration is more search-efficient than sequential single candidates. All candidates from one iteration are evaluated concurrently via asyncio.gather, with individual tasks parallelized up to max_parallel.
3. **Strict search/test separation**: The proposer never sees test results. This prevents overfitting to the test set and ensures discovered harnesses generalize. Enforced at the filesystem level.
4. **Lineage graph over linear parent chain**: The paper shows that the proposer naturally merges successful strategies from different lineages. A lineage graph (vs. a simple parent pointer) captures this and enables the proposer to reason about which combination of ideas has/hasn't been tried.
5. **Proposer access instrumentation from day one**: Logging every file read and tool call the proposer makes is trivial to implement and provides enormous diagnostic value. Without it, you can't determine whether the proposer is actually reading traces, how many prior candidates it inspects, or whether it's using evo-query vs. raw file reads.

