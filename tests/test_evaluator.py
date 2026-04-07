import pytest

from evoharness.core.candidate import CandidateScores
from evoharness.core.evaluator import Evaluator
from evoharness.sandbox.subprocess_sandbox import SubprocessSandbox


@pytest.fixture
def harness_dir(tmp_path):
    d = tmp_path / "harness"
    d.mkdir()
    (d / "__init__.py").write_text("")
    (d / "agent.py").write_text(
        "def run(input_data, trace_callback=None):\n"
        "    if trace_callback:\n"
        "        trace_callback({'type': 'info', 'message': 'running'})\n"
        "    return str(input_data)\n"
    )
    return d


@pytest.fixture
def eval_script(tmp_path):
    p = tmp_path / "eval_suite.py"
    p.write_text(
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
    return p


class TestSubprocessSandbox:
    @pytest.mark.asyncio
    async def test_run_task_success(self, harness_dir, eval_script):
        sandbox = SubprocessSandbox()
        result = await sandbox.run_task(
            harness_dir=harness_dir,
            eval_script=eval_script,
            eval_function="evaluate",
            task_data={
                "task_id": "t1",
                "input_data": "hello",
                "expected": "hello",
            },
            timeout=30,
        )
        assert result.task_id == "t1"
        assert result.scores.get("accuracy") == 1.0
        assert result.error is None

    @pytest.mark.asyncio
    async def test_run_task_failure(self, harness_dir, eval_script):
        sandbox = SubprocessSandbox()
        result = await sandbox.run_task(
            harness_dir=harness_dir,
            eval_script=eval_script,
            eval_function="evaluate",
            task_data={
                "task_id": "t1",
                "input_data": "hello",
                "expected": "wrong",
            },
            timeout=30,
        )
        assert result.scores.get("accuracy") == 0.0


class TestEvaluator:
    @pytest.mark.asyncio
    async def test_evaluate_candidate(self, harness_dir, eval_script):
        sandbox = SubprocessSandbox()
        evaluator = Evaluator(sandbox=sandbox, eval_script=eval_script, max_parallel=2)
        tasks = [
            {"task_id": "t1", "input_data": "hello", "expected": "hello"},
            {"task_id": "t2", "input_data": "world", "expected": "world"},
        ]
        scores, traces = await evaluator.evaluate_candidate(harness_dir, tasks, timeout=30)
        assert isinstance(scores, CandidateScores)
        assert scores.aggregate["accuracy"] == 1.0
        assert "t1" in scores.per_task
        assert "t2" in scores.per_task
        assert "t1" in traces
        assert "t2" in traces

    @pytest.mark.asyncio
    async def test_partial_failure(self, harness_dir, eval_script):
        sandbox = SubprocessSandbox()
        evaluator = Evaluator(sandbox=sandbox, eval_script=eval_script, max_parallel=2)
        tasks = [
            {"task_id": "t1", "input_data": "hello", "expected": "hello"},
            {"task_id": "t2", "input_data": "world", "expected": "wrong"},
        ]
        scores, traces = await evaluator.evaluate_candidate(harness_dir, tasks, timeout=30)
        assert scores.aggregate["accuracy"] == 0.5
        assert scores.per_task["t1"]["accuracy"] == 1.0
        assert scores.per_task["t2"]["accuracy"] == 0.0
