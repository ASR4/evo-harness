import pytest
from evoharness.core.config import (
    DashboardConfig,
    EvalConfig,
    EvoConfig,
    HarnessConfig,
    ProjectConfig,
    ScoringConfig,
    ScoringSecondary,
    SearchBudget,
    SearchConfig,
    TracingConfig,
    load_config,
    save_config,
)


class TestEvoConfig:
    def test_default_config(self):
        """Default config should be valid."""
        config = EvoConfig()
        assert config.project.name == "my-agent"
        assert config.scoring.primary_metric == "accuracy"
        assert config.scoring.direction == "maximize"
        assert config.search.max_iterations == 50
        assert config.search.candidates_per_iteration == 2

    def test_round_trip(self, tmp_path):
        """Config should survive write -> read."""
        config = EvoConfig()
        config.project.name = "test-agent"
        p = tmp_path / "evo.toml"
        save_config(config, p)
        loaded = load_config(p)
        assert loaded.project.name == "test-agent"
        assert loaded.search.budget.max_cost_usd == 50.0

    def test_full_config(self, tmp_path):
        """Write a full config with all fields and read it back."""
        config = EvoConfig(
            project=ProjectConfig(name="full-agent", description="A complete test"),
            harness=HarnessConfig(template="my_harness/main.py"),
            eval=EvalConfig(search_tasks=30),
            scoring=ScoringConfig(
                secondary=[ScoringSecondary(name="tokens", direction="minimize")],
            ),
            search=SearchConfig(
                proposer="anthropic:claude-sonnet-4-20250514",
                budget=SearchBudget(max_cost_usd=100.0),
            ),
            tracing=TracingConfig(),
            dashboard=DashboardConfig(),
        )

        p = tmp_path / "evo.toml"
        save_config(config, p)
        loaded = load_config(p)
        assert loaded.project.name == "full-agent"
        assert loaded.eval.search_tasks == 30
        assert len(loaded.scoring.secondary) == 1
        assert loaded.scoring.secondary[0].name == "tokens"
        assert loaded.search.budget.max_cost_usd == 100.0

    def test_invalid_direction(self):
        """Invalid direction should raise ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ScoringConfig(direction="up")

    def test_negative_tasks(self):
        """Negative task count should raise."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            EvalConfig(search_tasks=-1)

    def test_zero_iterations(self):
        """Zero iterations should raise."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SearchConfig(max_iterations=0)

    def test_negative_budget(self):
        """Negative budget should raise."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SearchBudget(max_cost_usd=-1.0)

    def test_port_range(self):
        """Port outside 1-65535 should raise."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DashboardConfig(port=0)
        with pytest.raises(ValidationError):
            DashboardConfig(port=70000)

    def test_missing_file(self, tmp_path):
        """Loading a nonexistent file should raise."""
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nope.toml")
