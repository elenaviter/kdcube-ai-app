# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/sdk/codegen/solve_result.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

@dataclass
class SolveResult:
    raw: Dict[str, Any]
    _round0: Dict[str, Any] = field(default_factory=dict, init=False)

    # ----- core blocks -----
    def codegen(self) -> Dict[str, Any]:
        return (self.raw or {}).get("codegen") or {}

    def rounds(self) -> List[Dict[str, Any]]:
        return self.codegen().get("rounds") or []

    def result_json(self) -> Optional[Dict[str, Any]]:
        r = self._first_round()
        try:
            items = ((r.get("outputs") or {}).get("items") or [])
            for it in items:
                if it.get("filename") == "result.json" and isinstance(it.get("data"), dict):
                    return it.get("data")
        except Exception:
            pass
        return None

    # ----- out accessors (result.json as source of truth) -----
    def out_items(self) -> List[Dict[str, Any]]:
        """Everything that codegen produced under result.json['out']."""
        rj = self.result_json() or {}
        arr = rj.get("out") or []
        return arr if isinstance(arr, list) else []

    def execution_id(self) -> Optional[str]:
        """Execution/session id emitted by codegen (if present in result.json)."""
        rj = self.result_json() or {}
        eid = rj.get("execution_id")
        if isinstance(eid, str) and eid.strip():
            return eid.strip()
        # fallback to codegen run_id if not present
        return self.run_id()

    # ----- contract/deliverables -----
    def deliverables_map(self) -> Dict[str, Any]:
        """
        Standardized structure returned by CodegenToolManager.solve():
          { slot_name: { "description": str, "value": <artifact dict or None> }, ... }
        NOTE: 'value' is a SINGLE artifact dict for the slot (not a list).
        """
        return (self.raw or {}).get("deliverables") or {}

    def deliverable_slots(self) -> Set[str]:
        """Names of contract slots."""
        return set(self.deliverables_map().keys())

    def deliverables_out(self) -> List[Dict[str, Any]]:
        """Flattened SINGLE artifact per slot (if present)."""
        out: List[Dict[str, Any]] = []
        for _, spec in (self.deliverables_map() or {}).items():
            val = (spec or {}).get("value")
            if isinstance(val, dict):
                out.append(val)
        return out

    # ----- reasoning & hints from the first codegen round -----
    def interpretation_instruction(self) -> str:
        r = self._first_round()
        return r.get("result_interpretation_instruction") or ""

    def round_reasoning(self) -> str:
        r = self._first_round()
        return r.get("internal_thinking") or ""

    def round_notes(self) -> str:
        """Return the last non-empty note string from the first round, if any."""
        r = self._first_round()
        notes = r.get("notes")
        if isinstance(notes, list):
            for s in reversed(notes):
                if isinstance(s, str) and s.strip():
                    return s.strip()
        if isinstance(notes, str):
            return notes.strip()
        return ""

    # ----- derived fields -----
    def run_id(self) -> Optional[str]:
        cg = self.codegen()
        rid = cg.get("run_id")
        if rid:
            return rid
        r = self._first_round()
        return r.get("run_id") if isinstance(r, dict) else None

    def outdir_workdir(self) -> Tuple[Optional[str], Optional[str]]:
        r = self._first_round()
        return (r.get("outdir"), r.get("workdir"))

    # ----- citations: url+title(+text) only -----
    def citations(self) -> List[Dict[str, Any]]:
        """
        Extract citable items from result.json['out'] and normalize:
          [{url, title, text?}, ...]
        """
        cites: List[Dict[str, Any]] = []
        for row in self.out_items():
            if not isinstance(row, dict):
                continue
            if row.get("type") != "inline" or not bool(row.get("citable")):
                continue

            data = row.get("output") or row.get("value")
            # Unified downstream expects url+title
            def _push(d: Dict[str, Any]):
                d = d or {}
                url = str(d.get("url") or "").strip()
                if not url:
                    return
                d["url"] = url
                cites.append({**d,
                              "tool_id": row.get("tool_id") or "",
                              "resource_id": row.get("resource_id") or ""})

            if isinstance(data, list):
                for c in data:
                    if isinstance(c, dict):
                        _push(c)
            elif isinstance(data, dict):
                _push(data)

        # dedupe by URL
        seen, uniq = set(), []
        for c in cites:
            u = c.get("url")
            if u and u not in seen:
                seen.add(u)
                uniq.append(c)
        return uniq

    def indexable_tool_ids(self) -> Set[str]:
        """
        Tools we might want to index (tags-only) from result.json style items.
        Keep it conservative and only pick obvious search-like tools.
        """
        def _is_search(tid: str) -> bool:
            tid = (tid or "").lower()
            return tid.endswith(".web_search") or tid.endswith(".kb_search")
        return {it.get("tool_id") for it in self.out_items() if _is_search(it.get("tool_id") or "")} - {None}

    # ----- helpers -----
    def _first_round(self) -> Dict[str, Any]:
        if self._round0:
            return self._round0
        rr = self.rounds()
        self._round0 = (rr[0] if rr else {}) if isinstance(rr, list) else {}
        return self._round0