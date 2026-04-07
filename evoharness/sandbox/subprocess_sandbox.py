from __future__ import annotations

import asyncio
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

from evoharness.sandbox.base import BaseSandbox, TaskResult

_RUNNER_SOURCE = r'''from __future__ import annotations

import importlib
import json
import sys
import time
import traceback
from pathlib import Path


def _apply_result_payload(payload: dict, result: object) -> None:
    if result is None:
        return
    if isinstance(result, dict):
        if "task_id" in result:
            payload["task_id"] = result["task_id"]
        if "scores" in result:
            payload["scores"] = dict(result["scores"])
        if "output" in result:
            payload["output"] = result["output"]
        return
    if hasattr(result, "task_id"):
        payload["task_id"] = getattr(result, "task_id")
    if hasattr(result, "scores"):
        payload["scores"] = dict(getattr(result, "scores"))
    if hasattr(result, "output"):
        payload["output"] = getattr(result, "output")


def main() -> None:
    config_path = Path(sys.argv[1])
    with config_path.open(encoding="utf-8") as f:
        cfg = json.load(f)

    task_file = Path(cfg["task_file"])
    output_file = Path(cfg["output_file"])
    harness_parent = cfg["harness_parent"]
    harness_pkg = cfg["harness_pkg"]
    eval_parent = cfg["eval_parent"]
    eval_module = cfg["eval_module"]
    eval_function = cfg["eval_function"]

    sys.path.insert(0, harness_parent)
    sys.path.insert(0, eval_parent)

    trace_events: list = []

    def trace_callback(event: dict) -> None:
        trace_events.append(event)

    payload: dict = {
        "task_id": "",
        "scores": {},
        "output": None,
        "trace_events": trace_events,
        "error": None,
        "duration_seconds": 0.0,
    }

    try:
        with task_file.open(encoding="utf-8") as tf:
            task = json.load(tf)
        payload["task_id"] = str(task.get("task_id", ""))

        harness_module = importlib.import_module(harness_pkg)
        eval_mod = importlib.import_module(eval_module)
        fn = getattr(eval_mod, eval_function)

        t0 = time.perf_counter()
        result = fn(harness_module, task, trace_callback)
        payload["duration_seconds"] = time.perf_counter() - t0
        _apply_result_payload(payload, result)
    except Exception:
        payload["error"] = traceback.format_exc()
    finally:
        with output_file.open("w", encoding="utf-8") as out:
            json.dump(payload, out, indent=2, default=str)


if __name__ == "__main__":
    main()
'''


class SubprocessSandbox(BaseSandbox):
    async def run_task(
        self,
        harness_dir: Path,
        eval_script: Path,
        eval_function: str,
        task_data: dict,
        timeout: int = 300,
    ) -> TaskResult:
        harness_dir = harness_dir.resolve()
        eval_script = eval_script.resolve()
        task_id = str(task_data.get("task_id", ""))

        tmp = Path(tempfile.mkdtemp(prefix="evoharness_eval_"))
        task_path = tmp / "task.json"
        results_path = tmp / "results.json"
        config_path = tmp / "config.json"
        runner_path = tmp / "_runner.py"

        try:
            task_path.write_text(json.dumps(task_data, default=str), encoding="utf-8")
            config = {
                "task_file": str(task_path),
                "output_file": str(results_path),
                "harness_parent": str(harness_dir.parent),
                "harness_pkg": harness_dir.name,
                "eval_parent": str(eval_script.parent),
                "eval_module": eval_script.stem,
                "eval_function": eval_function,
            }
            config_path.write_text(json.dumps(config), encoding="utf-8")
            runner_path.write_text(_RUNNER_SOURCE, encoding="utf-8")

            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                str(runner_path),
                str(config_path),
                cwd=str(tmp),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            wall0 = time.perf_counter()
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=float(timeout),
                )
            except asyncio.TimeoutError:
                proc.kill()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=10.0)
                except asyncio.TimeoutError:
                    pass
                elapsed = time.perf_counter() - wall0
                err = f"Task timed out after {timeout} seconds"
                return TaskResult(
                    task_id=task_id or "unknown",
                    scores={},
                    output=None,
                    trace_events=[{"type": "error", "message": err}],
                    error=err,
                    duration_seconds=elapsed,
                )

            wall = time.perf_counter() - wall0
            stderr_text = (stderr or b"").decode(errors="replace").strip()
            stdout_text = (stdout or b"").decode(errors="replace").strip()

            if not results_path.is_file():
                parts = [f"subprocess exited with code {proc.returncode}"]
                if stderr_text:
                    parts.append(stderr_text)
                if stdout_text:
                    parts.append(stdout_text)
                return TaskResult(
                    task_id=task_id or "unknown",
                    scores={},
                    output=None,
                    trace_events=[
                        {
                            "type": "error",
                            "message": "No results file from evaluator subprocess",
                            "stderr": stderr_text,
                            "stdout": stdout_text,
                            "returncode": proc.returncode,
                        }
                    ],
                    error="\n".join(parts),
                    duration_seconds=wall,
                )

            try:
                data = json.loads(results_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                return TaskResult(
                    task_id=task_id or "unknown",
                    scores={},
                    output=None,
                    trace_events=[
                        {
                            "type": "error",
                            "message": "Invalid results JSON",
                            "detail": str(e),
                        }
                    ],
                    error=f"Invalid results JSON: {e}",
                    duration_seconds=wall,
                )

            tid = str(data.get("task_id") or task_id or "unknown")
            scores_raw = data.get("scores") or {}
            scores: dict[str, float] = {}
            for k, v in scores_raw.items():
                try:
                    scores[str(k)] = float(v)
                except (TypeError, ValueError):
                    scores[str(k)] = 0.0

            trace = list(data.get("trace_events") or [])
            inner_error = data.get("error")
            inner_duration = data.get("duration_seconds")
            try:
                duration = float(inner_duration) if inner_duration is not None else wall
            except (TypeError, ValueError):
                duration = wall

            err_msg: str | None = None
            if inner_error:
                err_msg = str(inner_error)
            elif proc.returncode != 0:
                err_msg = (
                    f"subprocess exited with code {proc.returncode}"
                    + (f": {stderr_text}" if stderr_text else "")
                )

            if err_msg and not any(
                e.get("type") == "error" for e in trace if isinstance(e, dict)
            ):
                trace = [
                    *trace,
                    {"type": "error", "message": err_msg},
                ]

            return TaskResult(
                task_id=tid,
                scores=scores,
                output=data.get("output"),
                trace_events=trace,
                error=err_msg,
                duration_seconds=duration,
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
