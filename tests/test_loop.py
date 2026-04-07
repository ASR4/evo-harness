import pytest

from evoharness.core.config import EvoConfig
from evoharness.core.loop import SearchLoop
from evoharness.proposers.base import BaseProposer, ProposerResult


class MockProposer(BaseProposer):
    def __init__(self):
        super().__init__()
        self.call_count = 0

    async def propose(
        self,
        history_dir,
        frontier,
        leaderboard,
        task_descriptions,
        config,
        iteration=0,
        max_iterations=50,
        cost_used=0.0,
        max_cost=50.0,
        steering=None,
        candidates_per_iteration=1,
    ):
        self.call_count += 1
        pad = "\n".join(
            f"# proposer_noise_{self.call_count}_{i}" for i in range(40)
        )
        return [
            ProposerResult(
                harness_files={
                    "agent.py": (
                        f"{pad}\n"
                        "def run(input_data, trace_callback=None):\n"
                        "    if trace_callback:\n"
                        "        trace_callback({'type': 'info'})\n"
                        "    return str(input_data)\n"
                    ),
                },
                reasoning=f"Iteration {self.call_count} improvement",
                parent_id="000",
                strategy_tag=f"strategy_{self.call_count}",
                tokens_used=1000,
                cost_usd=0.01,
                access_log=[{"action": "read_file", "path": "test"}],
            )
        ]


@pytest.fixture
def project_dir(tmp_path):
    harness = tmp_path / "harness"
    harness.mkdir()
    (harness / "__init__.py").write_text("")
    (harness / "agent.py").write_text(
        "def run(input_data, trace_callback=None):\n"
        "    if trace_callback:\n"
        "        trace_callback({'type': 'info'})\n"
        "    return str(input_data)\n"
    )

    evals = tmp_path / "evals"
    evals.mkdir()
    (evals / "eval_suite.py").write_text(
        "def get_tasks(split):\n"
        "    return [\n"
        "        {'task_id': 't1', 'description': 'echo', 'input_data': 'hello', 'expected': 'hello'},\n"
        "        {'task_id': 't2', 'description': 'echo', 'input_data': 'world', 'expected': 'world'},\n"
        "    ]\n"
        "\n"
        "def evaluate(harness_module, task, trace_callback):\n"
        "    agent = __import__(f'{harness_module.__name__}.agent', fromlist=['agent'])\n"
        "    out = agent.run(task.get('input_data'), trace_callback)\n"
        "    ok = out == task.get('expected')\n"
        "    return {'task_id': task['task_id'], 'scores': {'accuracy': 1.0 if ok else 0.0}, 'output': out}\n"
    )

    return tmp_path


class TestSearchLoop:
    @pytest.mark.asyncio
    async def test_baseline_initialization(self, project_dir):
        config = EvoConfig()
        config.search.max_iterations = 1
        config.eval.search_tasks = 2
        proposer = MockProposer()
        loop = SearchLoop(config=config, project_dir=project_dir, proposer=proposer)
        summary = await loop.run()
        assert summary["iterations"] >= 1
        assert summary["candidates_evaluated"] >= 2

    @pytest.mark.asyncio
    async def test_loop_stops_at_max_iterations(self, project_dir):
        config = EvoConfig()
        config.search.max_iterations = 2
        config.eval.search_tasks = 2
        proposer = MockProposer()
        loop = SearchLoop(config=config, project_dir=project_dir, proposer=proposer)
        summary = await loop.run()
        assert summary["iterations"] <= 2
        assert summary["stop_reason"] == "max_iterations reached"

    @pytest.mark.asyncio
    async def test_frontier_updated(self, project_dir):
        config = EvoConfig()
        config.search.max_iterations = 1
        config.eval.search_tasks = 2
        proposer = MockProposer()
        loop = SearchLoop(config=config, project_dir=project_dir, proposer=proposer)
        await loop.run()
        assert len(loop.frontier.frontier) >= 1

    @pytest.mark.asyncio
    async def test_lineage_tracked(self, project_dir):
        config = EvoConfig()
        config.search.max_iterations = 1
        config.eval.search_tasks = 2
        proposer = MockProposer()
        loop = SearchLoop(config=config, project_dir=project_dir, proposer=proposer)
        await loop.run()
        assert "000" in loop.lineage.nodes

    @pytest.mark.asyncio
    async def test_cost_tracked(self, project_dir):
        config = EvoConfig()
        config.search.max_iterations = 1
        config.eval.search_tasks = 2
        proposer = MockProposer()
        loop = SearchLoop(config=config, project_dir=project_dir, proposer=proposer)
        await loop.run()
        assert loop.cost_tracker.total_cost_usd >= 0
