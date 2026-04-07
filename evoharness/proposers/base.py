from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ProposerResult:
    harness_files: dict[str, str]
    reasoning: str
    parent_id: str | None = None
    parent_ids: list[str] = field(default_factory=list)
    strategy_tag: str | None = None
    tokens_used: int = 0
    cost_usd: float = 0.0
    access_log: list[dict] = field(default_factory=list)


class BaseProposer(ABC):
    def __init__(self) -> None:
        self._access_log: list[dict] = []

    def _log_access(self, action: str, **details: object) -> None:
        self._access_log.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": action,
                **details,
            }
        )

    def _reset_access_log(self) -> list[dict]:
        log = self._access_log.copy()
        self._access_log = []
        return log

    @abstractmethod
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
        """
        Propose one or more new harness candidates.
        Returns a list of ProposerResult (one per candidate).
        """
        ...
