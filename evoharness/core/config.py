from __future__ import annotations

from pathlib import Path
from typing import Literal

import tomli_w
from pydantic import BaseModel, ConfigDict, Field, field_validator

try:
    import tomli
except ModuleNotFoundError:
    import tomllib as tomli

Direction = Literal["maximize", "minimize"]


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "my-agent"
    description: str = ""


class HarnessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    template: str = "harness/agent.py"
    mutable_files: list[str] = Field(default_factory=lambda: ["harness/*.py"])
    readonly_files: list[str] = Field(default_factory=list)


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    script: str = "evals/eval_suite.py"
    function: str = "evaluate"
    search_tasks: int = 20
    test_tasks: int = 100
    task_timeout: int = 300
    max_parallel: int = 4

    @field_validator("search_tasks", "test_tasks")
    @classmethod
    def _positive_tasks(cls, v: int) -> int:
        if v < 1:
            msg = "must be >= 1"
            raise ValueError(msg)
        return v

    @field_validator("task_timeout")
    @classmethod
    def _positive_timeout(cls, v: int) -> int:
        if v < 1:
            msg = "must be >= 1"
            raise ValueError(msg)
        return v

    @field_validator("max_parallel")
    @classmethod
    def _parallel_ge_one(cls, v: int) -> int:
        if v < 1:
            msg = "must be >= 1"
            raise ValueError(msg)
        return v


class ScoringSecondary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    direction: Direction = "minimize"


class ScoringConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary_metric: str = "accuracy"
    direction: Direction = "maximize"
    secondary: list[ScoringSecondary] = Field(default_factory=list)


class SearchBudget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_cost_usd: float = 50.0
    max_proposer_cost_usd: float = 5.0

    @field_validator("max_cost_usd", "max_proposer_cost_usd")
    @classmethod
    def _non_negative_cost(cls, v: float) -> float:
        if v < 0:
            msg = "must be >= 0"
            raise ValueError(msg)
        return v


class SearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_iterations: int = 50
    candidates_per_iteration: int = 2
    proposer: str = "anthropic:claude-sonnet-4-20250514"
    frontier_size: int = 10
    patience: int = 10
    proposer_temperature: float = 1.0
    budget: SearchBudget = Field(default_factory=SearchBudget)

    @field_validator("max_iterations", "candidates_per_iteration", "frontier_size", "patience")
    @classmethod
    def _positive_int_search(cls, v: int) -> int:
        if v < 1:
            msg = "must be >= 1"
            raise ValueError(msg)
        return v

    @field_validator("proposer_temperature")
    @classmethod
    def _temperature_non_negative(cls, v: float) -> float:
        if v < 0:
            msg = "must be >= 0"
            raise ValueError(msg)
        return v


class TracingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capture_prompts: bool = True
    capture_responses: bool = True
    capture_tool_calls: bool = True
    capture_state_updates: bool = True
    max_trace_size_mb: int = 10

    @field_validator("max_trace_size_mb")
    @classmethod
    def _trace_size_positive(cls, v: int) -> int:
        if v < 1:
            msg = "must be >= 1"
            raise ValueError(msg)
        return v


class DashboardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    port: int = 8420
    host: str = "localhost"

    @field_validator("port")
    @classmethod
    def _port_in_range(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            msg = "must be between 1 and 65535"
            raise ValueError(msg)
        return v


class EvoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    harness: HarnessConfig = Field(default_factory=HarnessConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)


def load_config(path: Path) -> EvoConfig:
    with path.open("rb") as f:
        data = tomli.load(f)
    return EvoConfig.model_validate(data)


def save_config(config: EvoConfig, path: Path) -> None:
    payload = config.model_dump(mode="python")
    with path.open("wb") as f:
        tomli_w.dump(payload, f)
