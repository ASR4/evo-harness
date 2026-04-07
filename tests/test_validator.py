import pytest
from evoharness.core.validator import validate_candidate


@pytest.fixture
def valid_harness(tmp_path):
    harness_dir = tmp_path / "harness"
    harness_dir.mkdir()
    (harness_dir / "agent.py").write_text(
        "def run(input_data, trace_callback=None): return str(input_data)"
    )
    return harness_dir


@pytest.fixture
def syntax_error_harness(tmp_path):
    harness_dir = tmp_path / "harness"
    harness_dir.mkdir()
    (harness_dir / "agent.py").write_text("def run(:\n  pass")
    return harness_dir


@pytest.fixture
def import_error_harness(tmp_path):
    harness_dir = tmp_path / "harness"
    harness_dir.mkdir()
    (harness_dir / "agent.py").write_text(
        "import nonexistent_module_xyz123\ndef run(): pass"
    )
    return harness_dir


@pytest.fixture
def no_run_harness(tmp_path):
    harness_dir = tmp_path / "harness"
    harness_dir.mkdir()
    (harness_dir / "agent.py").write_text("x = 42")
    return harness_dir


class TestValidator:
    @pytest.mark.asyncio
    async def test_valid_harness_passes(self, valid_harness):
        result = await validate_candidate(valid_harness)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_syntax_error(self, syntax_error_harness):
        result = await validate_candidate(syntax_error_harness)
        assert result.passed is False
        assert result.stage == "syntax"

    @pytest.mark.asyncio
    async def test_import_error(self, import_error_harness):
        result = await validate_candidate(import_error_harness)
        assert result.passed is False
        assert result.stage == "import"

    @pytest.mark.asyncio
    async def test_missing_function(self, no_run_harness):
        result = await validate_candidate(no_run_harness)
        assert result.passed is False
        assert result.stage == "interface"

    @pytest.mark.asyncio
    async def test_nonexistent_dir(self, tmp_path):
        result = await validate_candidate(tmp_path / "nope")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_missing_entry_file(self, tmp_path):
        harness = tmp_path / "h"
        harness.mkdir()
        result = await validate_candidate(harness)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_duration_tracked(self, valid_harness):
        result = await validate_candidate(valid_harness)
        assert result.duration_seconds > 0
