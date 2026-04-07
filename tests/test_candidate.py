from datetime import datetime, timezone

from evoharness.core.candidate import Candidate, CandidateMetadata, CandidateScores


class TestCandidate:
    def test_create_metadata(self):
        m = CandidateMetadata(candidate_id="001", created_at=datetime.now(timezone.utc))
        assert m.candidate_id == "001"
        assert m.parent_id is None
        assert m.parent_ids == []

    def test_candidate_id_property(self):
        m = CandidateMetadata(candidate_id="005", created_at=datetime.now(timezone.utc))
        c = Candidate(metadata=m)
        assert c.id == "005"

    def test_primary_parents_from_parent_ids(self):
        m = CandidateMetadata(
            candidate_id="003",
            created_at=datetime.now(timezone.utc),
            parent_ids=["001", "002"],
        )
        c = Candidate(metadata=m)
        assert c.primary_parents == ["001", "002"]

    def test_primary_parents_from_parent_id(self):
        m = CandidateMetadata(
            candidate_id="002",
            created_at=datetime.now(timezone.utc),
            parent_id="001",
        )
        c = Candidate(metadata=m)
        assert c.primary_parents == ["001"]

    def test_primary_parents_empty(self):
        m = CandidateMetadata(candidate_id="000", created_at=datetime.now(timezone.utc))
        c = Candidate(metadata=m)
        assert c.primary_parents == []

    def test_scores_default(self):
        m = CandidateMetadata(candidate_id="000", created_at=datetime.now(timezone.utc))
        c = Candidate(metadata=m)
        assert c.scores.aggregate == {}
        assert c.scores.per_task == {}

    def test_serialization_round_trip(self):
        m = CandidateMetadata(
            candidate_id="007",
            created_at=datetime.now(timezone.utc),
            parent_id="004",
            strategy_tag="context_compression",
        )
        s = CandidateScores(
            aggregate={"accuracy": 0.85, "tokens": 9200},
            per_task={"t1": {"accuracy": 1.0}, "t2": {"accuracy": 0.7}},
        )
        c = Candidate(metadata=m, scores=s, summary="test summary")
        data = c.model_dump(mode="json")
        c2 = Candidate.model_validate(data)
        assert c2.id == "007"
        assert c2.scores.aggregate["accuracy"] == 0.85
