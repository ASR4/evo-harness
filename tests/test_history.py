import pytest
from datetime import datetime, timezone

from evoharness.core.candidate import CandidateMetadata, CandidateScores
from evoharness.core.history import HistoryStore


@pytest.fixture
def store(tmp_path):
    return HistoryStore(tmp_path)


def _make_metadata(cid, parent_id=None):
    return CandidateMetadata(
        candidate_id=cid,
        created_at=datetime.now(timezone.utc),
        parent_id=parent_id,
        parent_ids=[parent_id] if parent_id else [],
    )


def _make_scores(accuracy=0.5, per_task=None):
    return CandidateScores(
        aggregate={"accuracy": accuracy},
        per_task=per_task or {"t1": {"accuracy": accuracy}},
    )


class TestHistoryStore:
    def test_next_candidate_id_empty(self, store):
        assert store.next_candidate_id() == "000"

    def test_store_and_get(self, store):
        store.store_candidate(
            candidate_id="000",
            harness_files={"agent.py": "def run(): pass"},
            scores=_make_scores(0.8),
            metadata=_make_metadata("000"),
            summary="test baseline",
        )
        c = store.get_candidate("000")
        assert c.id == "000"
        assert c.scores.aggregate["accuracy"] == 0.8
        assert c.summary == "test baseline"

    def test_next_id_increments(self, store):
        store.store_candidate("000", {"a.py": ""}, _make_scores(), _make_metadata("000"), "")
        assert store.next_candidate_id() == "001"
        store.store_candidate("001", {"a.py": ""}, _make_scores(), _make_metadata("001"), "")
        assert store.next_candidate_id() == "002"

    def test_list_candidates(self, store):
        for i in range(3):
            cid = f"{i:03d}"
            store.store_candidate(cid, {"a.py": f"v{i}"}, _make_scores(0.5 + i * 0.1), _make_metadata(cid), f"c{i}")
        candidates = store.list_candidates()
        assert len(candidates) == 3

    def test_get_leaderboard(self, store):
        store.store_candidate("000", {"a.py": "v0"}, _make_scores(0.5), _make_metadata("000"), "")
        store.store_candidate("001", {"a.py": "v1"}, _make_scores(0.9), _make_metadata("001", "000"), "")
        store.store_candidate("002", {"a.py": "v2"}, _make_scores(0.7), _make_metadata("002", "000"), "")
        board = store.get_leaderboard("accuracy", "maximize")
        assert board[0]["candidate_id"] == "001"
        assert board[-1]["candidate_id"] == "000"

    def test_get_not_found(self, store):
        with pytest.raises(FileNotFoundError):
            store.get_candidate("999")

    def test_traces_written(self, store):
        traces = {"t1": [{"type": "prompt", "content": "hello"}]}
        store.store_candidate("000", {"a.py": ""}, _make_scores(), _make_metadata("000"), "", traces=traces)
        trace_dir = store.candidate_dir("000") / "traces"
        assert trace_dir.is_dir()
        files = list(trace_dir.iterdir())
        assert len(files) == 1

    def test_diff_generated(self, store):
        store.store_candidate("000", {"agent.py": "def run(): return 'a'"}, _make_scores(), _make_metadata("000"), "")
        store.store_candidate("001", {"agent.py": "def run(): return 'b'"}, _make_scores(), _make_metadata("001", "000"), "")
        patch = store.candidate_dir("001") / "diff_from_parent.patch"
        assert patch.is_file()
        content = patch.read_text()
        assert "a/agent.py" in content or "b/agent.py" in content

    def test_access_log_written(self, store):
        log = [{"action": "read_file", "path": "test.py"}]
        store.store_candidate("000", {"a.py": ""}, _make_scores(), _make_metadata("000"), "", access_log=log)
        alog = store.candidate_dir("000") / "proposer_access.jsonl"
        assert alog.is_file()

    def test_is_duplicate(self, store):
        store.store_candidate("000", {"agent.py": "def run(): pass"}, _make_scores(), _make_metadata("000"), "")
        dup = store.is_duplicate({"agent.py": "def run(): pass"})
        assert dup == "000"

    def test_not_duplicate(self, store):
        store.store_candidate("000", {"agent.py": "def run(): return 'a'"}, _make_scores(), _make_metadata("000"), "")
        dup = store.is_duplicate({"agent.py": "def totally_different(): return 'completely new code here'"})
        assert dup is None

    def test_get_task_matrix(self, store):
        scores0 = CandidateScores(
            aggregate={"accuracy": 0.5},
            per_task={"t1": {"accuracy": 1.0}, "t2": {"accuracy": 0.0}},
        )
        scores1 = CandidateScores(
            aggregate={"accuracy": 0.75},
            per_task={"t1": {"accuracy": 0.5}, "t2": {"accuracy": 1.0}},
        )
        store.store_candidate("000", {"a.py": "v0"}, scores0, _make_metadata("000"), "")
        store.store_candidate("001", {"a.py": "v1"}, scores1, _make_metadata("001"), "")
        matrix = store.get_task_matrix("accuracy")
        assert len(matrix["candidate_ids"]) == 2
        assert len(matrix["task_ids"]) == 2

    def test_get_failures(self, store):
        scores = CandidateScores(
            aggregate={"accuracy": 0.5},
            per_task={"t1": {"accuracy": 1.0}, "t2": {"accuracy": 0.0}},
        )
        traces = {
            "t2": [{"type": "error", "error_type": "timeout", "message": "timed out"}],
        }
        store.store_candidate("000", {"a.py": ""}, scores, _make_metadata("000"), "", traces=traces)
        failures = store.get_failures("000", metric="accuracy", threshold=0.5)
        assert len(failures) == 1
        assert failures[0]["task_id"] == "t2"
        assert failures[0]["score"] == 0.0
