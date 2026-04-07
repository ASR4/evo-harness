from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FrontierPoint:
    candidate_id: str
    scores: dict[str, float]


class ParetoFrontier:
    def __init__(self, objectives: list[dict[str, str]]) -> None:
        """
        objectives: [{"name": "accuracy", "direction": "maximize"},
                      {"name": "context_tokens", "direction": "minimize"}]
        """
        self.objectives = objectives
        self.frontier: list[FrontierPoint] = []

    def _objective_names(self) -> list[str]:
        return [o["name"] for o in self.objectives]

    def _validate_scores(self, scores: dict[str, float]) -> None:
        names = set(self._objective_names())
        missing = names - set(scores.keys())
        if missing:
            raise ValueError(f"scores missing objectives: {sorted(missing)}")

    def _collapse_single_objective(self) -> None:
        if len(self.objectives) != 1 or not self.frontier:
            return
        name = self.objectives[0]["name"]
        direction = self.objectives[0]["direction"]
        if direction == "maximize":
            best_val = max(p.scores[name] for p in self.frontier)
        elif direction == "minimize":
            best_val = min(p.scores[name] for p in self.frontier)
        else:
            raise ValueError(f"invalid direction: {direction!r}")
        tied = [p for p in self.frontier if p.scores[name] == best_val]
        tied.sort(key=lambda p: p.candidate_id)
        self.frontier = [tied[0]]

    def update(self, candidate_id: str, scores: dict[str, float]) -> bool:
        self._validate_scores(scores)
        self.frontier = [p for p in self.frontier if p.candidate_id != candidate_id]
        for existing in self.frontier:
            if self.dominates(existing.scores, scores):
                self._collapse_single_objective()
                return False
        self.frontier = [
            p for p in self.frontier if not self.dominates(scores, p.scores)
        ]
        self.frontier.append(FrontierPoint(candidate_id=candidate_id, scores=dict(scores)))
        self._collapse_single_objective()
        return True

    def dominates(self, scores_a: dict[str, float], scores_b: dict[str, float]) -> bool:
        self._validate_scores(scores_a)
        self._validate_scores(scores_b)
        strictly_better_on_one = False
        for obj in self.objectives:
            name = obj["name"]
            direction = obj["direction"]
            va = scores_a[name]
            vb = scores_b[name]
            if direction == "maximize":
                if va < vb:
                    return False
                if va > vb:
                    strictly_better_on_one = True
            elif direction == "minimize":
                if va > vb:
                    return False
                if va < vb:
                    strictly_better_on_one = True
            else:
                raise ValueError(f"invalid direction: {direction!r}")
        return strictly_better_on_one

    def get_best(self, metric: str) -> FrontierPoint | None:
        if not self.frontier:
            return None
        direction: str | None = None
        for obj in self.objectives:
            if obj["name"] == metric:
                direction = obj["direction"]
                break
        if direction is None:
            raise ValueError(f"unknown metric: {metric!r}")
        best = self.frontier[0]
        for p in self.frontier[1:]:
            vb, vp = best.scores[metric], p.scores[metric]
            if direction == "maximize" and vp > vb:
                best = p
            elif direction == "minimize" and vp < vb:
                best = p
        return best

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path, objectives: list[dict[str, str]]) -> ParetoFrontier:
        raw = path.read_text(encoding="utf-8")
        data: list[dict[str, object]] = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("frontier JSON must be a list of point objects")
        return cls.from_json(data, objectives)

    def to_json(self) -> list[dict[str, object]]:
        return [
            {"candidate_id": p.candidate_id, "scores": dict(p.scores)}
            for p in self.frontier
        ]

    @classmethod
    def from_json(
        cls, data: list[dict[str, object]], objectives: list[dict[str, str]]
    ) -> ParetoFrontier:
        pf = cls(objectives)
        for item in data:
            if not isinstance(item, dict):
                raise ValueError("each frontier entry must be an object")
            cid = item.get("candidate_id")
            scores_raw = item.get("scores")
            if not isinstance(cid, str):
                raise ValueError("candidate_id must be a string")
            if not isinstance(scores_raw, dict):
                raise ValueError("scores must be an object")
            scores: dict[str, float] = {str(k): float(v) for k, v in scores_raw.items()}
            pf.update(cid, scores)
        return pf
