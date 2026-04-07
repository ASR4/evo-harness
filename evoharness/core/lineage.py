from __future__ import annotations

import json
from collections import deque
from pathlib import Path


class LineageGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, list[str]]] = {}

    def _ensure_node(self, candidate_id: str) -> dict[str, list[str]]:
        if candidate_id not in self.nodes:
            self.nodes[candidate_id] = {"parents": [], "children": []}
        return self.nodes[candidate_id]

    def add_candidate(self, candidate_id: str, parent_ids: list[str]) -> None:
        if candidate_id in self.nodes:
            for old_pid in self.nodes[candidate_id]["parents"]:
                pnode = self.nodes.get(old_pid)
                if pnode is not None and candidate_id in pnode["children"]:
                    pnode["children"].remove(candidate_id)
        node = self._ensure_node(candidate_id)
        node["parents"] = list(parent_ids)
        for pid in parent_ids:
            pnode = self._ensure_node(pid)
            if candidate_id not in pnode["children"]:
                pnode["children"].append(candidate_id)

    def get_parents(self, candidate_id: str) -> list[str]:
        if candidate_id not in self.nodes:
            return []
        return list(self.nodes[candidate_id]["parents"])

    def get_children(self, candidate_id: str) -> list[str]:
        if candidate_id not in self.nodes:
            return []
        return list(self.nodes[candidate_id]["children"])

    def get_ancestors(self, candidate_id: str) -> list[str]:
        if candidate_id not in self.nodes:
            return []
        seen: set[str] = set()
        out: list[str] = []
        q: deque[str] = deque(self.nodes[candidate_id]["parents"])
        while q:
            pid = q.popleft()
            if pid in seen:
                continue
            seen.add(pid)
            out.append(pid)
            if pid in self.nodes:
                q.extend(self.nodes[pid]["parents"])
        return out

    def get_roots(self) -> list[str]:
        return sorted(cid for cid, data in self.nodes.items() if not data["parents"])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.nodes, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> LineageGraph:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("lineage JSON must be an object mapping ids to nodes")
        g = cls()
        for cid, node in data.items():
            if not isinstance(cid, str):
                raise ValueError("lineage keys must be string candidate ids")
            if not isinstance(node, dict):
                raise ValueError(f"node {cid!r} must be an object")
            parents = node.get("parents", [])
            children = node.get("children", [])
            if not isinstance(parents, list) or not all(isinstance(x, str) for x in parents):
                raise ValueError(f"node {cid!r}: parents must be a list of strings")
            if not isinstance(children, list) or not all(isinstance(x, str) for x in children):
                raise ValueError(f"node {cid!r}: children must be a list of strings")
            g.nodes[cid] = {"parents": list(parents), "children": list(children)}
        return g

    def format_lineage(self, candidate_id: str) -> str:
        if candidate_id not in self.nodes:
            return f"{candidate_id}\n  (unknown candidate)"
        parents = self.get_parents(candidate_id)
        children = self.get_children(candidate_id)
        ancestors = self.get_ancestors(candidate_id)
        lines = [candidate_id]
        lines.append(f"  parents: {', '.join(parents) if parents else '(none)'}")
        lines.append(f"  children: {', '.join(children) if children else '(none)'}")
        lines.append(
            f"  ancestors: {', '.join(ancestors) if ancestors else '(none)'}"
        )
        return "\n".join(lines)
