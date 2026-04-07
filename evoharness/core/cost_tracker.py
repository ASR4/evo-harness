from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class CostEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    category: str
    candidate_id: str
    cost_usd: float
    tokens_used: int = 0
    description: str = ""


class CostTracker:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: list[CostEntry] = []
        if path.exists():
            self._load()

    def add(self, entry: CostEntry) -> None:
        self.entries.append(entry)
        self._save()

    def add_proposer_cost(
        self,
        candidate_id: str,
        cost_usd: float,
        tokens_used: int = 0,
        description: str = "",
    ) -> None:
        self.add(
            CostEntry(
                timestamp=datetime.now(timezone.utc),
                category="proposer",
                candidate_id=candidate_id,
                cost_usd=cost_usd,
                tokens_used=tokens_used,
                description=description,
            )
        )

    def add_eval_cost(
        self,
        candidate_id: str,
        cost_usd: float,
        description: str = "",
    ) -> None:
        self.add(
            CostEntry(
                timestamp=datetime.now(timezone.utc),
                category="eval",
                candidate_id=candidate_id,
                cost_usd=cost_usd,
                tokens_used=0,
                description=description,
            )
        )

    @property
    def total_cost_usd(self) -> float:
        return sum(e.cost_usd for e in self.entries)

    @property
    def proposer_cost_usd(self) -> float:
        return sum(e.cost_usd for e in self.entries if e.category == "proposer")

    @property
    def eval_cost_usd(self) -> float:
        return sum(e.cost_usd for e in self.entries if e.category == "eval")

    def is_over_budget(self, max_cost_usd: float) -> bool:
        return self.total_cost_usd >= max_cost_usd

    def summary(self) -> dict:
        by_cat: dict[str, float] = {}
        tokens_by_cat: dict[str, int] = {}
        for e in self.entries:
            by_cat[e.category] = by_cat.get(e.category, 0.0) + e.cost_usd
            tokens_by_cat[e.category] = tokens_by_cat.get(e.category, 0) + e.tokens_used

        last_ts: datetime | None = None
        if self.entries:
            last_ts = max(e.timestamp for e in self.entries)

        return {
            "path": str(self.path),
            "entry_count": len(self.entries),
            "total_cost_usd": self.total_cost_usd,
            "proposer_cost_usd": self.proposer_cost_usd,
            "eval_cost_usd": self.eval_cost_usd,
            "cost_by_category": by_cat,
            "tokens_by_category": tokens_by_cat,
            "total_tokens_used": sum(e.tokens_used for e in self.entries),
            "last_entry_at": last_ts.isoformat() if last_ts else None,
        }

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "entries": [e.model_dump(mode="json") for e in self.entries],
        }
        self.path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _load(self) -> None:
        raw = self.path.read_text(encoding="utf-8").strip()
        if not raw:
            self.entries = []
            return
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError(f"invalid cost tracker file (expected object): {self.path}")
        entries_raw = data.get("entries", [])
        if not isinstance(entries_raw, list):
            raise ValueError(f"invalid cost tracker file (entries must be list): {self.path}")
        self.entries = [CostEntry.model_validate(item) for item in entries_raw]
