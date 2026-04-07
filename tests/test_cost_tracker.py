import pytest
from pathlib import Path

from evoharness.core.cost_tracker import CostTracker


class TestCostTracker:
    def test_empty(self, tmp_path):
        ct = CostTracker(tmp_path / "costs.json")
        assert ct.total_cost_usd == 0.0
        assert ct.proposer_cost_usd == 0.0
        assert ct.eval_cost_usd == 0.0

    def test_add_proposer_cost(self, tmp_path):
        ct = CostTracker(tmp_path / "costs.json")
        ct.add_proposer_cost("001", 1.50, tokens_used=5000)
        assert ct.total_cost_usd == 1.50
        assert ct.proposer_cost_usd == 1.50
        assert ct.eval_cost_usd == 0.0

    def test_add_eval_cost(self, tmp_path):
        ct = CostTracker(tmp_path / "costs.json")
        ct.add_eval_cost("001", 2.00)
        assert ct.eval_cost_usd == 2.00

    def test_is_over_budget(self, tmp_path):
        ct = CostTracker(tmp_path / "costs.json")
        ct.add_proposer_cost("001", 5.0)
        assert not ct.is_over_budget(10.0)
        ct.add_eval_cost("001", 5.0)
        assert ct.is_over_budget(10.0)

    def test_persistence(self, tmp_path):
        path = tmp_path / "costs.json"
        ct1 = CostTracker(path)
        ct1.add_proposer_cost("001", 1.0)
        ct1.add_eval_cost("001", 2.0)
        ct2 = CostTracker(path)
        assert ct2.total_cost_usd == 3.0
        assert len(ct2.entries) == 2

    def test_summary(self, tmp_path):
        ct = CostTracker(tmp_path / "costs.json")
        ct.add_proposer_cost("001", 1.0, tokens_used=1000)
        ct.add_eval_cost("001", 2.0)
        s = ct.summary()
        assert s["total_cost_usd"] == 3.0
        assert s["entry_count"] == 2
        assert s["total_tokens_used"] == 1000
