from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import shutil
import signal
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from evoharness.core.config import EvoConfig
from evoharness.core.candidate import CandidateMetadata, CandidateScores
from evoharness.core.cost_tracker import CostTracker
from evoharness.core.evaluator import Evaluator
from evoharness.core.frontier import ParetoFrontier
from evoharness.core.history import HistoryStore
from evoharness.core.lineage import LineageGraph
from evoharness.core.validator import validate_candidate
from evoharness.proposers.base import BaseProposer, ProposerResult
from evoharness.sandbox.subprocess_sandbox import SubprocessSandbox

logger = logging.getLogger(__name__)


class SearchState:
    def __init__(self) -> None:
        self.iteration: int = 0
        self.total_cost_usd: float = 0.0
        self.iterations_since_frontier_update: int = 0
        self.stop_requested: bool = False
        self.start_time: float = 0.0


class SearchLoop:
    def __init__(
        self,
        config: EvoConfig,
        project_dir: Path,
        proposer: BaseProposer,
    ) -> None:
        self.config = config
        self.project_dir = project_dir.resolve()
        self.proposer = proposer
        self.state = SearchState()

        self.history = HistoryStore(self.project_dir)
        self.evo_dir = self.project_dir / ".evo"
        self.evo_dir.mkdir(parents=True, exist_ok=True)

        self.objectives = self._build_objectives()
        self.frontier = ParetoFrontier(self.objectives)
        frontier_path = self.evo_dir / "frontier.json"
        if frontier_path.is_file():
            try:
                self.frontier = ParetoFrontier.load(frontier_path, self.objectives)
            except Exception:
                logger.exception("failed to load frontier from %s; starting fresh", frontier_path)

        self.lineage = LineageGraph()
        lineage_path = self.evo_dir / "lineage.json"
        if lineage_path.is_file():
            try:
                self.lineage = LineageGraph.load(lineage_path)
            except Exception:
                logger.exception("failed to load lineage from %s; starting fresh", lineage_path)

        self.cost_tracker = CostTracker(self.evo_dir / "cost_tracker.json")
        self.state.total_cost_usd = self.cost_tracker.total_cost_usd

        sandbox = SubprocessSandbox()
        self.evaluator = Evaluator(
            sandbox=sandbox,
            eval_script=self.project_dir / config.eval.script,
            eval_function=config.eval.function,
            max_parallel=config.eval.max_parallel,
        )

        self._search_tasks_cache: list[dict[str, Any]] | None = None
        self._id_lock: asyncio.Lock | None = None
        self._stop_reason: str = ""

    def _build_objectives(self) -> list[dict[str, str]]:
        objs: list[dict[str, str]] = [
            {
                "name": self.config.scoring.primary_metric,
                "direction": self.config.scoring.direction,
            }
        ]
        for sec in self.config.scoring.secondary:
            objs.append({"name": sec.name, "direction": sec.direction})
        return objs

    def _harness_entry_name(self) -> str:
        return Path(self.config.harness.template).name

    def _read_harness_template_files(self) -> dict[str, str]:
        template_path = (self.project_dir / self.config.harness.template).resolve()
        harness_root = (self.project_dir / "harness").resolve()
        if not template_path.exists():
            msg = f"harness template not found: {template_path}"
            raise FileNotFoundError(msg)

        files: dict[str, str] = {}
        if template_path.is_file():
            try:
                rel = template_path.relative_to(harness_root).as_posix()
            except ValueError:
                rel = template_path.name
            files[rel] = template_path.read_text(encoding="utf-8")
        elif template_path.is_dir():
            for p in sorted(template_path.rglob("*")):
                if not p.is_file() or "__pycache__" in p.parts:
                    continue
                rel = p.relative_to(template_path).as_posix()
                files[rel] = p.read_text(encoding="utf-8")
        else:
            msg = f"harness template is not a file or directory: {template_path}"
            raise FileNotFoundError(msg)
        if not files:
            msg = f"no harness files under template path: {template_path}"
            raise ValueError(msg)
        return files

    def _ensure_aggregate_scores(self, scores: CandidateScores) -> CandidateScores:
        agg = dict(scores.aggregate)
        for obj in self.objectives:
            name = obj["name"]
            if name not in agg:
                agg[name] = 0.0
        return CandidateScores(aggregate=agg, per_task=dict(scores.per_task))

    def _task_to_dict(self, task: Any) -> dict[str, Any]:
        if hasattr(task, "model_dump") and callable(task.model_dump):
            raw = task.model_dump()
        elif is_dataclass(task):
            raw = asdict(task)
        elif isinstance(task, dict):
            raw = task
        else:
            raw = {
                "task_id": getattr(task, "task_id", None),
                "description": getattr(task, "description", None),
                "input_data": getattr(task, "input_data", None),
                "expected": getattr(task, "expected", None),
            }
        out: dict[str, Any] = {
            "task_id": raw.get("task_id", ""),
            "description": raw.get("description", ""),
            "input_data": raw.get("input_data"),
            "expected": raw.get("expected"),
        }
        if not out["task_id"]:
            out["task_id"] = f"task_{id(task)}"
        return out

    def _load_search_tasks(self) -> list[dict[str, Any]]:
        if self._search_tasks_cache is not None:
            return self._search_tasks_cache

        script_path = (self.project_dir / self.config.eval.script).resolve()
        if not script_path.is_file():
            msg = f"eval script not found: {script_path}"
            raise FileNotFoundError(msg)

        module_name = f"_evoharness_eval_{script_path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if spec is None or spec.loader is None:
            msg = f"cannot load eval module from {script_path}"
            raise ImportError(msg)

        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        get_tasks = getattr(mod, "get_tasks", None)
        if not callable(get_tasks):
            msg = f"eval script {script_path} has no callable get_tasks"
            raise AttributeError(msg)

        raw = get_tasks("search")
        if not isinstance(raw, list):
            raw = list(raw)

        limit = self.config.eval.search_tasks
        converted = [self._task_to_dict(t) for t in raw[:limit]]
        self._search_tasks_cache = converted
        return converted

    def _get_task_descriptions(self) -> list[str]:
        tasks = self._load_search_tasks()
        out: list[str] = []
        for t in tasks:
            desc = t.get("description")
            if isinstance(desc, str) and desc.strip():
                out.append(desc.strip())
            else:
                tid = t.get("task_id", "")
                out.append(str(tid) if tid else "(no description)")
        return out

    def _save_config_snapshot(self) -> None:
        path = self.evo_dir / "config.snapshot.json"
        path.write_text(
            json.dumps(self.config.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _make_sigint_handler(self) -> Callable[[int, Any], None]:
        loop = asyncio.get_running_loop()

        def _handler_threadsafe(_signum: int, _frame: Any | None) -> None:
            def _set_stop() -> None:
                self.state.stop_requested = True

            loop.call_soon_threadsafe(_set_stop)
            logger.warning("stop requested (SIGINT); finishing current work gracefully")

        return _handler_threadsafe

    async def run(
        self,
        steering: str | None = None,
        on_iteration: Callable[[int, SearchState], None] | None = None,
        *,
        resume: bool = False,
    ) -> dict[str, Any]:
        self.state.start_time = time.time()
        self._stop_reason = ""
        self._id_lock = asyncio.Lock()
        self._save_config_snapshot()

        prev_sigint = signal.signal(signal.SIGINT, self._make_sigint_handler())
        try:
            await self._initialize_baseline(resume=resume)

            while True:
                stop, reason = self._should_stop()
                if stop:
                    self._stop_reason = reason
                    break

                self.state.iteration += 1
                self.state.total_cost_usd = self.cost_tracker.total_cost_usd
                if on_iteration:
                    on_iteration(self.state.iteration, self.state)

                try:
                    await self._run_iteration(steering)
                except Exception:
                    logger.exception("iteration %s failed", self.state.iteration)

            return self._build_summary()
        finally:
            signal.signal(signal.SIGINT, prev_sigint)

    async def _initialize_baseline(self, *, resume: bool) -> None:
        cdir = self.history.candidate_dir("000")
        meta = cdir / "metadata.json"
        scores_path = cdir / "scores.json"
        complete = cdir.is_dir() and meta.is_file() and scores_path.is_file()

        if complete:
            if resume:
                logger.info("baseline candidate 000 present; resuming without re-initialization")
                cand = self.history.get_candidate("000")
                scores = self._ensure_aggregate_scores(cand.scores)
                if not self.frontier.frontier:
                    updated = self.frontier.update("000", scores.aggregate)
                    if updated:
                        self.frontier.save(self.evo_dir / "frontier.json")
                if "000" not in self.lineage.nodes:
                    self.lineage.add_candidate("000", [])
                    self.lineage.save(self.evo_dir / "lineage.json")
                return

            logger.info("baseline candidate 000 present; skipping re-initialization")
            if not self.frontier.frontier:
                cand = self.history.get_candidate("000")
                scores = self._ensure_aggregate_scores(cand.scores)
                self.frontier.update("000", scores.aggregate)
                self.frontier.save(self.evo_dir / "frontier.json")
            if "000" not in self.lineage.nodes:
                self.lineage.add_candidate("000", [])
                self.lineage.save(self.evo_dir / "lineage.json")
            return

        harness_files = self._read_harness_template_files()
        staging_root = self.evo_dir / "staging" / "000"
        staging_harness = staging_root / "harness"
        try:
            if staging_root.exists():
                shutil.rmtree(staging_root, ignore_errors=True)
            staging_harness.mkdir(parents=True, exist_ok=True)
            for rel_path, content in harness_files.items():
                out = staging_harness / rel_path
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(content, encoding="utf-8")

            validation = await validate_candidate(
                staging_harness,
                harness_entry=self._harness_entry_name(),
            )
            if not validation.passed:
                msg = f"baseline failed validation at {validation.stage}: {validation.error}"
                raise RuntimeError(msg)

            eval_tasks = self._load_search_tasks()
            scores, traces = await self.evaluator.evaluate_candidate(
                harness_dir=staging_harness,
                tasks=eval_tasks,
                timeout=self.config.eval.task_timeout,
            )
            scores = self._ensure_aggregate_scores(scores)

            now = datetime.now(timezone.utc)
            metadata = CandidateMetadata(
                candidate_id="000",
                created_at=now,
                parent_id=None,
                parent_ids=[],
                proposer_model=None,
                proposer_reasoning="baseline from template",
                proposer_tokens_used=0,
                proposer_cost_usd=0.0,
                eval_cost_usd=0.0,
                eval_duration_seconds=0.0,
                strategy_tag="baseline",
                iteration=0,
            )

            self.history.store_candidate(
                candidate_id="000",
                harness_files=harness_files,
                scores=scores,
                metadata=metadata,
                summary="Baseline harness copied from project template.",
                traces=traces,
                access_log=None,
            )

            self.frontier.update("000", scores.aggregate)
            self.frontier.save(self.evo_dir / "frontier.json")
            self.state.iterations_since_frontier_update = 0

            self.lineage.add_candidate("000", [])
            self.lineage.save(self.evo_dir / "lineage.json")
        finally:
            shutil.rmtree(staging_root, ignore_errors=True)

    async def _run_iteration(self, steering: str | None = None) -> None:
        leaderboard = self.history.get_leaderboard(
            self.config.scoring.primary_metric,
            self.config.scoring.direction,
        )
        frontier_data = self.frontier.to_json()

        proposals = await self.proposer.propose(
            history_dir=self.history.candidates_dir,
            frontier=frontier_data,
            leaderboard=leaderboard,
            task_descriptions=self._get_task_descriptions(),
            config=self.config.model_dump(),
            iteration=self.state.iteration,
            max_iterations=self.config.search.max_iterations,
            cost_used=self.cost_tracker.total_cost_usd,
            max_cost=self.config.search.budget.max_cost_usd,
            steering=steering,
            candidates_per_iteration=self.config.search.candidates_per_iteration,
        )

        if not proposals:
            logger.warning("proposer returned no candidates for iteration %s", self.state.iteration)
            return

        await self._evaluate_batch(proposals)

    async def _evaluate_batch(self, proposals: list[ProposerResult]) -> None:
        tasks = [self._process_proposal(p) for p in proposals]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                logger.error("proposal %s processing failed: %s", i, result)

    async def _process_proposal(self, proposal: ProposerResult) -> None:
        dup = self.history.is_duplicate(proposal.harness_files)
        if dup:
            logger.warning("duplicate harness skipped (matches candidate %s)", dup)
            return

        assert self._id_lock is not None
        async with self._id_lock:
            dup2 = self.history.is_duplicate(proposal.harness_files)
            if dup2:
                logger.warning("duplicate harness skipped (matches candidate %s)", dup2)
                return
            candidate_id = self.history.next_candidate_id()

        staging_root = self.evo_dir / "staging" / candidate_id
        staging_harness = staging_root / "harness"

        try:
            staging_harness.mkdir(parents=True, exist_ok=True)
            for rel_path, content in proposal.harness_files.items():
                file_path = staging_harness / rel_path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content, encoding="utf-8")

            validation = await validate_candidate(
                staging_harness,
                harness_entry=self._harness_entry_name(),
            )

            parent_ids: list[str] = []
            if proposal.parent_ids:
                parent_ids = list(proposal.parent_ids)
            elif proposal.parent_id:
                parent_ids = [proposal.parent_id]

            now = datetime.now(timezone.utc)
            proposer_model = getattr(self.proposer, "model", None)
            proposer_model_str = str(proposer_model) if proposer_model is not None else None

            if not validation.passed:
                logger.warning(
                    "candidate %s failed validation at %s: %s",
                    candidate_id,
                    validation.stage,
                    validation.error,
                )
                failed_agg = {obj["name"]: 0.0 for obj in self.objectives}
                failed_scores = CandidateScores(aggregate=failed_agg, per_task={})
                metadata = CandidateMetadata(
                    candidate_id=candidate_id,
                    created_at=now,
                    parent_id=proposal.parent_id,
                    parent_ids=parent_ids,
                    proposer_model=proposer_model_str,
                    proposer_reasoning=proposal.reasoning,
                    proposer_tokens_used=proposal.tokens_used,
                    proposer_cost_usd=proposal.cost_usd,
                    eval_cost_usd=0.0,
                    eval_duration_seconds=validation.duration_seconds,
                    strategy_tag=proposal.strategy_tag,
                    iteration=self.state.iteration,
                )
                self.history.store_candidate(
                    candidate_id=candidate_id,
                    harness_files=proposal.harness_files,
                    scores=failed_scores,
                    metadata=metadata,
                    summary=f"validation_failed [{validation.stage}]: {validation.error or ''}",
                    traces=None,
                    access_log=proposal.access_log,
                )
                self.lineage.add_candidate(candidate_id, parent_ids)
                self.lineage.save(self.evo_dir / "lineage.json")
                self.cost_tracker.add_proposer_cost(
                    candidate_id, proposal.cost_usd, proposal.tokens_used
                )
                return

            eval_tasks = self._load_search_tasks()
            scores, traces = await self.evaluator.evaluate_candidate(
                harness_dir=staging_harness,
                tasks=eval_tasks,
                timeout=self.config.eval.task_timeout,
            )
            scores = self._ensure_aggregate_scores(scores)

            metadata = CandidateMetadata(
                candidate_id=candidate_id,
                created_at=now,
                parent_id=proposal.parent_id,
                parent_ids=parent_ids,
                proposer_model=proposer_model_str,
                proposer_reasoning=proposal.reasoning,
                proposer_tokens_used=proposal.tokens_used,
                proposer_cost_usd=proposal.cost_usd,
                eval_cost_usd=0.0,
                eval_duration_seconds=0.0,
                strategy_tag=proposal.strategy_tag,
                iteration=self.state.iteration,
            )

            self.history.store_candidate(
                candidate_id=candidate_id,
                harness_files=proposal.harness_files,
                scores=scores,
                metadata=metadata,
                summary=proposal.reasoning,
                traces=traces,
                access_log=proposal.access_log,
            )

            frontier_updated = self.frontier.update(candidate_id, scores.aggregate)
            if frontier_updated:
                self.state.iterations_since_frontier_update = 0
                self.frontier.save(self.evo_dir / "frontier.json")
            else:
                self.state.iterations_since_frontier_update += 1

            self.lineage.add_candidate(candidate_id, parent_ids)
            self.lineage.save(self.evo_dir / "lineage.json")

            self.cost_tracker.add_proposer_cost(
                candidate_id, proposal.cost_usd, proposal.tokens_used
            )

            logger.info("candidate %s: %s", candidate_id, scores.aggregate)
        finally:
            shutil.rmtree(staging_root, ignore_errors=True)

    def _should_stop(self) -> tuple[bool, str]:
        if self.state.iteration >= self.config.search.max_iterations:
            return True, "max_iterations reached"
        if self.cost_tracker.is_over_budget(self.config.search.budget.max_cost_usd):
            return True, f"budget exhausted (${self.cost_tracker.total_cost_usd:.2f})"
        if self.state.iterations_since_frontier_update >= self.config.search.patience:
            return True, (
                f"patience exhausted ({self.config.search.patience} iterations "
                "without frontier improvement)"
            )
        if self.state.stop_requested:
            return True, "user requested stop"
        return False, ""

    def _best_aggregate_scores(self) -> dict[str, float]:
        rows = self.history.get_leaderboard(
            self.config.scoring.primary_metric,
            self.config.scoring.direction,
        )
        if not rows:
            return {}
        agg = rows[0].get("scores", {}).get("aggregate")
        if not isinstance(agg, dict):
            return {}
        out: dict[str, float] = {}
        for k, v in agg.items():
            try:
                out[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
        return out

    def _build_summary(self) -> dict[str, Any]:
        elapsed = max(0.0, time.time() - self.state.start_time)
        candidates = self.history.list_candidates()
        return {
            "iterations": self.state.iteration,
            "total_cost_usd": self.cost_tracker.total_cost_usd,
            "elapsed_seconds": elapsed,
            "candidates_evaluated": len(candidates),
            "best_scores": self._best_aggregate_scores(),
            "frontier": self.frontier.to_json(),
            "stop_reason": self._stop_reason or "completed",
        }
