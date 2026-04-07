from evoharness.core.frontier import FrontierPoint, ParetoFrontier


class TestParetoFrontier:
    def test_single_objective_maximize(self):
        pf = ParetoFrontier([{"name": "accuracy", "direction": "maximize"}])
        assert pf.update("001", {"accuracy": 0.8})
        assert pf.update("002", {"accuracy": 0.9})
        assert len(pf.frontier) == 1
        assert isinstance(pf.frontier[0], FrontierPoint)
        assert pf.frontier[0].candidate_id == "002"

    def test_single_objective_minimize(self):
        pf = ParetoFrontier([{"name": "cost", "direction": "minimize"}])
        pf.update("001", {"cost": 10.0})
        pf.update("002", {"cost": 5.0})
        assert len(pf.frontier) == 1
        assert pf.frontier[0].candidate_id == "002"

    def test_dominated_not_added(self):
        pf = ParetoFrontier(
            [
                {"name": "accuracy", "direction": "maximize"},
                {"name": "cost", "direction": "minimize"},
            ]
        )
        pf.update("001", {"accuracy": 0.9, "cost": 5.0})
        result = pf.update("002", {"accuracy": 0.8, "cost": 6.0})
        assert result is False
        assert len(pf.frontier) == 1

    def test_non_dominated_kept(self):
        pf = ParetoFrontier(
            [
                {"name": "accuracy", "direction": "maximize"},
                {"name": "cost", "direction": "minimize"},
            ]
        )
        pf.update("001", {"accuracy": 0.9, "cost": 10.0})
        pf.update("002", {"accuracy": 0.8, "cost": 5.0})
        assert len(pf.frontier) == 2

    def test_dominates(self):
        pf = ParetoFrontier(
            [
                {"name": "accuracy", "direction": "maximize"},
                {"name": "cost", "direction": "minimize"},
            ]
        )
        assert pf.dominates({"accuracy": 0.9, "cost": 5.0}, {"accuracy": 0.8, "cost": 6.0})
        assert not pf.dominates({"accuracy": 0.9, "cost": 5.0}, {"accuracy": 0.95, "cost": 6.0})

    def test_get_best(self):
        pf = ParetoFrontier(
            [
                {"name": "accuracy", "direction": "maximize"},
                {"name": "cost", "direction": "minimize"},
            ]
        )
        pf.update("001", {"accuracy": 0.9, "cost": 10.0})
        pf.update("002", {"accuracy": 0.8, "cost": 5.0})
        assert pf.get_best("accuracy").candidate_id == "001"
        assert pf.get_best("cost").candidate_id == "002"

    def test_save_load(self, tmp_path):
        pf = ParetoFrontier([{"name": "accuracy", "direction": "maximize"}])
        pf.update("001", {"accuracy": 0.9})
        path = tmp_path / "frontier.json"
        pf.save(path)
        loaded = ParetoFrontier.load(path, [{"name": "accuracy", "direction": "maximize"}])
        assert len(loaded.frontier) == 1
        assert loaded.frontier[0].scores["accuracy"] == 0.9

    def test_empty_frontier(self):
        pf = ParetoFrontier([{"name": "accuracy", "direction": "maximize"}])
        assert pf.get_best("accuracy") is None
        assert pf.to_json() == []

    def test_replaces_same_id(self):
        pf = ParetoFrontier([{"name": "accuracy", "direction": "maximize"}])
        pf.update("001", {"accuracy": 0.8})
        pf.update("001", {"accuracy": 0.9})
        assert len(pf.frontier) == 1
        assert pf.frontier[0].scores["accuracy"] == 0.9
