from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TaskResult:
    task_id: str
    scores: dict[str, float]
    output: Any = None
    trace_events: list[dict] = field(default_factory=list)
    error: str | None = None
    duration_seconds: float = 0.0


class BaseSandbox(ABC):
    @abstractmethod
    async def run_task(
        self,
        harness_dir: Path,
        eval_script: Path,
        eval_function: str,
        task_data: dict,
        timeout: int = 300,
    ) -> TaskResult:
        ...
