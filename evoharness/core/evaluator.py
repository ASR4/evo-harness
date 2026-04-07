from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from evoharness.core.candidate import CandidateScores
from evoharness.sandbox.base import BaseSandbox, TaskResult

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(
        self,
        sandbox: BaseSandbox,
        eval_script: Path,
        eval_function: str = "evaluate",
        max_parallel: int = 4,
    ) -> None:
        self.sandbox = sandbox
        self.eval_script = eval_script
        self.eval_function = eval_function
        self.max_parallel = max_parallel

    async def evaluate_candidate(
        self,
        harness_dir: Path,
        tasks: list[dict],
        timeout: int = 300,
    ) -> tuple[CandidateScores, dict[str, list[dict]]]:
        sem = asyncio.Semaphore(max(self.max_parallel, 1))

        async def run_one(task_data: dict) -> TaskResult:
            tid = str(task_data.get("task_id", "unknown"))
            async with sem:
                try:
                    return await self.sandbox.run_task(
                        harness_dir,
                        self.eval_script,
                        self.eval_function,
                        task_data,
                        timeout=timeout,
                    )
                except Exception as e:
                    logger.exception("evaluate_candidate task failed: task_id=%s", tid)
                    msg = str(e)
                    return TaskResult(
                        task_id=tid,
                        scores={},
                        output=None,
                        trace_events=[{"type": "error", "message": msg}],
                        error=msg,
                        duration_seconds=0.0,
                    )

        results = await asyncio.gather(*(run_one(t) for t in tasks))
        scores = self._aggregate_scores(list(results))
        traces = {r.task_id: list(r.trace_events) for r in results}
        return scores, traces

    def _aggregate_scores(self, results: list[TaskResult]) -> CandidateScores:
        if not results:
            return CandidateScores()

        all_metrics: set[str] = set()
        for r in results:
            all_metrics.update(r.scores.keys())

        per_task: dict[str, dict[str, float]] = {}
        for r in results:
            if r.error:
                per_task[r.task_id] = {m: 0.0 for m in all_metrics}
            else:
                per_task[r.task_id] = {
                    m: float(r.scores[m]) if m in r.scores else 0.0
                    for m in all_metrics
                }

        n = len(results)
        aggregate = (
            {m: sum(per_task[tid][m] for tid in per_task) / n for m in all_metrics}
            if n
            else {}
        )

        return CandidateScores(aggregate=aggregate, per_task=per_task)
