from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class CandidateMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    created_at: datetime
    parent_id: str | None = None
    parent_ids: list[str] = Field(default_factory=list)
    proposer_model: str | None = None
    proposer_reasoning: str = ""
    proposer_tokens_used: int = 0
    proposer_cost_usd: float = 0.0
    eval_cost_usd: float = 0.0
    eval_duration_seconds: float = 0.0
    strategy_tag: str | None = None
    iteration: int = 0


class CandidateScores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    aggregate: dict[str, float] = Field(default_factory=dict)
    per_task: dict[str, dict[str, float]] = Field(default_factory=dict)


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata: CandidateMetadata
    scores: CandidateScores = Field(default_factory=CandidateScores)
    summary: str = ""

    @property
    def id(self) -> str:
        return self.metadata.candidate_id

    @property
    def primary_parents(self) -> list[str]:
        if self.metadata.parent_ids:
            return self.metadata.parent_ids
        if self.metadata.parent_id:
            return [self.metadata.parent_id]
        return []
