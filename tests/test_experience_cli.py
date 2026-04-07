import pytest
from datetime import datetime, timezone
from pathlib import Path

from evoharness.core.candidate import CandidateMetadata, CandidateScores
from evoharness.core.frontier import ParetoFrontier
from evoharness.core.history import HistoryStore
from evoharness.core.lineage import LineageGraph
from evoharness.experience_cli.evo_query import (
    query_diff,
    query_failures,
    query_frontier,
    query_grep,
    query_leaderboard,
    query_lineage,
    query_task_matrix,
)


@pytest.fixture
def populated_store(tmp_path: Path) -> Path:
    store = HistoryStore(tmp_path)
    evo_dir = tmp_path / ".evo"

    store.store_candidate(
        "000",
        {"agent.py": "def run(x, cb=None): return str(x)"},
        CandidateScores(
            aggregate={"accuracy": 0.5},
            per_task={"t1": {"accuracy": 1.0}, "t2": {"accuracy": 0.0}},
        ),
        CandidateMetadata(
            candidate_id="000",
            created_at=datetime.now(timezone.utc),
            strategy_tag="baseline",
        ),
        "Baseline candidate",
        traces={
            "t2": [
                {
                    "type": "error",
                    "error_type": "wrong_answer",
                    "message": "expected hello got nothing",
                }
            ]
        },
    )

    store.store_candidate(
        "001",
        {"agent.py": "def run(x, cb=None): return str(x).upper()"},
        CandidateScores(
            aggregate={"accuracy": 0.75},
            per_task={"t1": {"accuracy": 1.0}, "t2": {"accuracy": 0.5}},
        ),
        CandidateMetadata(
            candidate_id="001",
            created_at=datetime.now(timezone.utc),
            parent_id="000",
            parent_ids=["000"],
            strategy_tag="uppercase",
        ),
        "Try uppercase",
    )

    store.store_candidate(
        "002",
        {"agent.py": "def run(x, cb=None):\n    return x\n"},
        CandidateScores(
            aggregate={"accuracy": 1.0},
            per_task={"t1": {"accuracy": 1.0}, "t2": {"accuracy": 1.0}},
        ),
        CandidateMetadata(
            candidate_id="002",
            created_at=datetime.now(timezone.utc),
            parent_id="001",
            parent_ids=["001"],
            strategy_tag="passthrough",
        ),
        "Direct passthrough",
    )

    objectives = [{"name": "accuracy", "direction": "maximize"}]
    frontier = ParetoFrontier(objectives)
    frontier.update("000", {"accuracy": 0.5})
    frontier.update("001", {"accuracy": 0.75})
    frontier.update("002", {"accuracy": 1.0})
    frontier.save(evo_dir / "frontier.json")

    lineage = LineageGraph()
    lineage.add_candidate("000", [])
    lineage.add_candidate("001", ["000"])
    lineage.add_candidate("002", ["001"])
    lineage.save(evo_dir / "lineage.json")

    return evo_dir


class TestExperienceCLI:
    def test_leaderboard(self, populated_store: Path) -> None:
        result = query_leaderboard(populated_store, metric="accuracy", top=10)
        assert "002" in result
        assert "001" in result
        assert "000" in result
        lines = result.strip().split("\n")
        assert len(lines) >= 4
        assert "ID" in lines[0] and "Score" in lines[0]
        assert all(" | " in ln for ln in lines[:3])

    def test_leaderboard_top(self, populated_store: Path) -> None:
        result = query_leaderboard(populated_store, metric="accuracy", top=1)
        lines = [
            ln
            for ln in result.strip().split("\n")
            if ln.strip() and not ln.startswith("-")
        ]
        assert len(lines) == 2
        assert "002" in lines[1]

    def test_frontier(self, populated_store: Path) -> None:
        result = query_frontier(populated_store)
        assert "002" in result
        assert "accuracy" in result
        assert "accuracy=1" in result or "accuracy=1.0" in result

    def test_diff(self, populated_store: Path) -> None:
        result = query_diff(populated_store, "000", "001")
        assert "agent.py" in result
        assert "---" in result or "+++" in result

    def test_diff_no_difference(self, populated_store: Path) -> None:
        result = query_diff(populated_store, "000", "000")
        assert "no differences in harness python files" in result.lower()

    def test_failures(self, populated_store: Path) -> None:
        result = query_failures(populated_store, "000", metric="accuracy")
        assert "t2" in result
        assert "wrong_answer" in result
        assert "0" in result or "0.0" in result

    def test_failures_no_failures(self, populated_store: Path) -> None:
        result = query_failures(populated_store, "002", metric="accuracy")
        assert "no failures under threshold" in result.lower()

    def test_grep(self, populated_store: Path) -> None:
        result = query_grep(populated_store, r"def run", scope="code")
        assert "agent.py" in result
        assert "harness" in result

    def test_grep_traces(self, populated_store: Path) -> None:
        result = query_grep(populated_store, r"wrong_answer", scope="traces")
        assert "wrong_answer" in result
        assert "t2.jsonl" in result or "traces" in result

    def test_grep_no_match(self, populated_store: Path) -> None:
        result = query_grep(populated_store, r"zzz_nonexistent_zzz")
        assert result.strip() == "no matches"

    def test_lineage(self, populated_store: Path) -> None:
        result = query_lineage(populated_store, "002")
        assert "002" in result
        assert "001" in result
        assert "parents:" in result
        assert "ancestors:" in result

    def test_task_matrix(self, populated_store: Path) -> None:
        result = query_task_matrix(populated_store, metric="accuracy")
        assert "t1" in result
        assert "t2" in result
        assert "000" in result
        assert "1" in result

    def test_frontier_missing(self, tmp_path: Path) -> None:
        evo_dir = tmp_path / ".evo"
        evo_dir.mkdir(parents=True)
        result = query_frontier(evo_dir)
        assert result == "error: frontier.json not found\n"

    def test_lineage_missing(self, tmp_path: Path) -> None:
        evo_dir = tmp_path / ".evo"
        evo_dir.mkdir(parents=True)
        result = query_lineage(evo_dir, "000")
        assert result == "error: lineage.json not found\n"
