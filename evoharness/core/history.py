from __future__ import annotations

import json
import re
from difflib import SequenceMatcher, unified_diff
from pathlib import Path

from evoharness.core.candidate import Candidate, CandidateMetadata, CandidateScores


def _trace_filename_for_task(task_id: str) -> str:
    safe = re.sub(r"[^\w.\-]+", "_", task_id).strip("._") or "task"
    return f"{safe}.jsonl"


class HistoryStore:
    """Manages the .evo/candidates/ directory."""

    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.evo_dir = project_dir / ".evo"
        self.candidates_dir = self.evo_dir / "candidates"
        self.candidates_dir.mkdir(parents=True, exist_ok=True)

    def next_candidate_id(self) -> str:
        """Next sequential ID, zero-padded to 3 digits (e.g., '000', '001', ...)."""
        nums = [int(x) for x in self._list_candidate_ids() if x.isdigit()]
        if not nums:
            return "000"
        return f"{max(nums) + 1:03d}"

    def store_candidate(
        self,
        candidate_id: str,
        harness_files: dict[str, str],
        scores: CandidateScores,
        metadata: CandidateMetadata,
        summary: str,
        traces: dict[str, list[dict]] | None = None,
        access_log: list[dict] | None = None,
    ) -> Path:
        cdir = self.candidate_dir(candidate_id)
        cdir.mkdir(parents=True, exist_ok=True)
        harness_root = cdir / "harness"
        harness_root.mkdir(parents=True, exist_ok=True)

        for rel_path, content in harness_files.items():
            out = harness_root / rel_path
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(content, encoding="utf-8")

        (cdir / "metadata.json").write_text(
            json.dumps(metadata.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (cdir / "scores.json").write_text(
            json.dumps(scores.model_dump(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (cdir / "summary.txt").write_text(summary, encoding="utf-8")

        if traces:
            traces_dir = cdir / "traces"
            traces_dir.mkdir(parents=True, exist_ok=True)
            for task_id, events in traces.items():
                tpath = traces_dir / _trace_filename_for_task(task_id)
                lines = [json.dumps(ev, ensure_ascii=False) for ev in events]
                tpath.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        parent_id = metadata.parent_ids[0] if metadata.parent_ids else metadata.parent_id
        if parent_id and self.candidate_dir(parent_id).is_dir():
            patch = self._generate_diff(candidate_id, parent_id)
            if patch.strip():
                (cdir / "diff_from_parent.patch").write_text(patch, encoding="utf-8")

        if access_log:
            apath = cdir / "proposer_access.jsonl"
            apath.write_text(
                "\n".join(json.dumps(rec, ensure_ascii=False) for rec in access_log)
                + ("\n" if access_log else ""),
                encoding="utf-8",
            )

        return cdir

    def get_candidate(self, candidate_id: str) -> Candidate:
        cdir = self.candidate_dir(candidate_id)
        meta_path = cdir / "metadata.json"
        if not cdir.is_dir() or not meta_path.is_file():
            raise FileNotFoundError(f"Candidate not found: {candidate_id}")

        metadata = CandidateMetadata.model_validate_json(meta_path.read_text(encoding="utf-8"))
        scores_path = cdir / "scores.json"
        if scores_path.is_file():
            scores = CandidateScores.model_validate_json(scores_path.read_text(encoding="utf-8"))
        else:
            scores = CandidateScores()
        summary_path = cdir / "summary.txt"
        summary = summary_path.read_text(encoding="utf-8") if summary_path.is_file() else ""
        return Candidate(metadata=metadata, scores=scores, summary=summary)

    def list_candidates(self) -> list[Candidate]:
        return [self.get_candidate(cid) for cid in self._list_candidate_ids()]

    def _list_candidate_ids(self) -> list[str]:
        if not self.candidates_dir.is_dir():
            return []
        ids = [
            p.name
            for p in self.candidates_dir.iterdir()
            if p.is_dir() and (p / "metadata.json").is_file()
        ]

        def _id_sort_key(s: str) -> tuple:
            return (0, int(s)) if s.isdigit() else (1, s)

        return sorted(ids, key=_id_sort_key)

    def get_leaderboard(self, metric: str, direction: str = "maximize") -> list[dict]:
        rows: list[dict] = []
        for c in self.list_candidates():
            rows.append(
                {
                    "candidate_id": c.metadata.candidate_id,
                    "scores": c.scores.model_dump(),
                    "parent_id": c.metadata.parent_id,
                    "strategy_tag": c.metadata.strategy_tag,
                }
            )

        def sort_key(row: dict) -> float:
            agg = row["scores"].get("aggregate") or {}
            v = agg.get(metric)
            if v is None:
                return float("-inf") if direction == "maximize" else float("inf")
            return float(v)

        reverse = direction == "maximize"
        rows.sort(key=sort_key, reverse=reverse)
        return rows

    def get_all_scores(self) -> list[dict]:
        return [
            {"candidate_id": c.metadata.candidate_id, "scores": c.scores.model_dump()}
            for c in self.list_candidates()
        ]

    def get_task_matrix(self, metric: str) -> dict:
        candidates = self.list_candidates()
        candidate_ids = [c.metadata.candidate_id for c in candidates]
        task_set: set[str] = set()
        for c in candidates:
            task_set.update(c.scores.per_task.keys())
        task_ids = sorted(task_set)
        matrix: list[list[float | None]] = []
        for c in candidates:
            row: list[float | None] = []
            for tid in task_ids:
                task_scores = c.scores.per_task.get(tid) or {}
                if metric in task_scores:
                    row.append(float(task_scores[metric]))
                else:
                    row.append(None)
            matrix.append(row)
        return {"candidate_ids": candidate_ids, "task_ids": task_ids, "matrix": matrix}

    def get_failures(
        self,
        candidate_id: str,
        metric: str = "accuracy",
        threshold: float = 0.5,
        top: int = 10,
    ) -> list[dict]:
        c = self.get_candidate(candidate_id)
        traces_dir = self.candidate_dir(candidate_id) / "traces"
        failed: list[dict] = []

        for task_id, task_scores in c.scores.per_task.items():
            if metric not in task_scores:
                continue
            score = float(task_scores[metric])
            if score >= threshold:
                continue
            err_type, err_msg = self._last_trace_error(traces_dir, task_id)
            failed.append(
                {
                    "task_id": task_id,
                    "score": score,
                    "error_type": err_type,
                    "error_message": err_msg,
                }
            )

        failed.sort(key=lambda r: r["score"])
        return failed[:top]

    def _last_trace_error(self, traces_dir: Path, task_id: str) -> tuple[str | None, str | None]:
        if not traces_dir.is_dir():
            return None, None
        tpath = traces_dir / _trace_filename_for_task(task_id)
        if not tpath.is_file():
            return None, None
        last_type: str | None = None
        last_msg: str | None = None
        for line in tpath.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") != "error":
                continue
            last_type = rec.get("error_type")
            if last_type is not None and not isinstance(last_type, str):
                last_type = str(last_type)
            last_msg = rec.get("error_message") or rec.get("message")
            if last_msg is not None and not isinstance(last_msg, str):
                last_msg = str(last_msg)
        return last_type, last_msg

    def _generate_diff(self, candidate_id: str, parent_id: str) -> str:
        cand_root = self.candidate_dir(candidate_id) / "harness"
        parent_root = self.candidate_dir(parent_id) / "harness"

        def py_relpaths(root: Path) -> list[str]:
            if not root.is_dir():
                return []
            return sorted(
                {p.relative_to(root).as_posix() for p in root.rglob("*.py")},
            )

        rels = sorted(set(py_relpaths(cand_root)) | set(py_relpaths(parent_root)))
        chunks: list[str] = []
        for rel in rels:
            p_old = parent_root / rel
            p_new = cand_root / rel
            old_lines = (
                p_old.read_text(encoding="utf-8").splitlines(keepends=True)
                if p_old.is_file()
                else []
            )
            new_lines = (
                p_new.read_text(encoding="utf-8").splitlines(keepends=True)
                if p_new.is_file()
                else []
            )
            if old_lines == new_lines:
                continue
            diff_iter = unified_diff(
                old_lines,
                new_lines,
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
            )
            chunk = "".join(diff_iter)
            if chunk:
                chunks.append(chunk)
        return "\n".join(chunks)

    def is_duplicate(self, new_files: dict[str, str], threshold: float = 0.95) -> str | None:
        new_blob = self._concat_py_sources(new_files)
        for cid in self._list_candidate_ids():
            existing = self._harness_py_blob(self.candidate_dir(cid) / "harness")
            if not existing and not new_blob:
                continue
            ratio = SequenceMatcher(a=new_blob, b=existing).ratio()
            if ratio >= threshold:
                return cid
        return None

    def _concat_py_sources(self, files: dict[str, str]) -> str:
        py_keys = sorted(k for k in files if k.endswith(".py"))
        return "".join(files[k] for k in py_keys)

    def _harness_py_blob(self, harness_root: Path) -> str:
        if not harness_root.is_dir():
            return ""
        parts: list[str] = []
        for p in sorted(harness_root.rglob("*.py"), key=lambda x: x.as_posix()):
            parts.append(p.read_text(encoding="utf-8"))
        return "".join(parts)

    def candidate_dir(self, candidate_id: str) -> Path:
        return self.candidates_dir / candidate_id
