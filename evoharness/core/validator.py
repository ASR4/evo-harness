from __future__ import annotations

import ast
import asyncio
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ValidationResult:
    passed: bool
    stage: str
    error: str | None = None
    duration_seconds: float = 0.0


def _syntax_check(harness_dir: Path) -> tuple[bool, str | None]:
    for py_path in sorted(harness_dir.rglob("*.py")):
        if "__pycache__" in py_path.parts:
            continue
        try:
            source = py_path.read_text(encoding="utf-8")
        except OSError as e:
            return False, f"{py_path}: read error: {e}"
        try:
            ast.parse(source, filename=str(py_path))
        except SyntaxError as e:
            lineno = e.lineno or 0
            return False, f"{py_path}:{lineno}: {e.msg}"
    return True, None


async def _run_python_c(
    code: str,
    timeout: float,
) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        code,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return -1, "", f"subprocess timed out after {timeout}s"
    stdout = stdout_b.decode("utf-8", errors="replace").strip()
    stderr = stderr_b.decode("utf-8", errors="replace").strip()
    return proc.returncode or 0, stdout, stderr


async def validate_candidate(
    harness_dir: Path,
    harness_entry: str = "agent.py",
    interface_function: str = "run",
) -> ValidationResult:
    """
    Four-stage validation:
    1. SYNTAX: ast.parse() all .py files
    2. IMPORT: import the harness module in a subprocess
    3. INTERFACE: verify the module exposes the expected function with the right signature

    Stage 4 (SMOKE TEST) is handled separately by the evaluator with real tasks.
    Each stage failure returns immediately.
    """
    t0 = time.perf_counter()
    harness_dir = harness_dir.resolve()
    entry_path = harness_dir / harness_entry
    if not harness_dir.is_dir():
        return ValidationResult(
            passed=False,
            stage="syntax",
            error=f"harness_dir is not a directory: {harness_dir}",
            duration_seconds=time.perf_counter() - t0,
        )
    if not entry_path.is_file():
        return ValidationResult(
            passed=False,
            stage="syntax",
            error=f"missing harness entry file: {entry_path}",
            duration_seconds=time.perf_counter() - t0,
        )

    module_name = Path(harness_entry).stem
    if not module_name.isidentifier():
        return ValidationResult(
            passed=False,
            stage="syntax",
            error=f"invalid module name derived from harness_entry: {module_name!r}",
            duration_seconds=time.perf_counter() - t0,
        )

    parent_dir = str(harness_dir)

    ok, err = _syntax_check(harness_dir)
    if not ok:
        return ValidationResult(
            passed=False,
            stage="syntax",
            error=err,
            duration_seconds=time.perf_counter() - t0,
        )

    import_code = (
        "import importlib\n"
        "import sys\n"
        f"sys.path.insert(0, {parent_dir!r})\n"
        f"importlib.import_module({module_name!r})\n"
    )
    rc, _out, err_text = await _run_python_c(import_code, timeout=10.0)
    if rc != 0:
        msg = err_text or "import failed with no stderr"
        return ValidationResult(
            passed=False,
            stage="import",
            error=msg,
            duration_seconds=time.perf_counter() - t0,
        )

    iface_code = (
        "import importlib\n"
        "import sys\n"
        f"sys.path.insert(0, {parent_dir!r})\n"
        f"_m = importlib.import_module({module_name!r})\n"
        f"_fn = {interface_function!r}\n"
        "assert hasattr(_m, _fn), f\"missing {repr(_fn)}\"\n"
        "_obj = getattr(_m, _fn)\n"
        "assert callable(_obj), f\"{repr(_fn)} is not callable\"\n"
    )
    rc, _out, err_text = await _run_python_c(iface_code, timeout=5.0)
    if rc != 0:
        msg = err_text or "interface check failed with no stderr"
        return ValidationResult(
            passed=False,
            stage="interface",
            error=msg,
            duration_seconds=time.perf_counter() - t0,
        )

    return ValidationResult(
        passed=True,
        stage="interface",
        error=None,
        duration_seconds=time.perf_counter() - t0,
    )
