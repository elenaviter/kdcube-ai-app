# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# skills_registry.py

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from functools import lru_cache
from typing import Any, Dict, List, Optional, Iterable, Literal
import pathlib
import logging

import yaml  # PyYAML; if you don't want this dependency, we can switch to JSON.

log = logging.getLogger("skills_registry")

# Bundle root = directory containing this file (same pattern as tool_descriptor.py)
BUNDLE_ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parent

COORDINATOR = "solver.coordinator"
REACT_DECISION = "solver.react.decision"
REACT_FOCUS = "solver.react.focus"
SkillConsumer = Literal["solver.coordinator", "solver.react.decision", "solver.react.focus"]


@dataclass
class SkillInstructionPaths:
    """Paths to instruction snippets, relative to bundle root."""
    full: Optional[pathlib.Path] = None
    compact: Optional[pathlib.Path] = None


@dataclass
class SkillToolRef:
    """Reference to a tool that belongs to this skill."""
    id: str
    role: Optional[str] = None  # e.g. "search", "fetch", "writer", "reconcile"


@dataclass
class SkillSpec:
    """
    Declarative description of a 'skill' – a bundle of related tools + instructions.

    - id:         short identifier used in prompts and selection decisions
    - name:       human-readable label shown in the 'skills gallery'
    - description: short description of what the skill gives
    - tools:      list of tool ids (alias.fn) associated with this skill
    - instruction_paths: paths to full / compact instruction snippets
    - built_in:   if true, include in default galleries for relevant consumers
    - include_for: set of consumers where this skill is relevant
    - category/tags: optional metadata for grouping / filtering
    """

    id: str
    name: str
    description: str

    tools: List[SkillToolRef] = field(default_factory=list)
    instruction_paths: SkillInstructionPaths = field(
        default_factory=SkillInstructionPaths
    )

    built_in: bool = False
    include_for: List[SkillConsumer] = field(default_factory=list)

    category: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    # ---- helpers ----

    def tool_ids(self) -> List[str]:
        return [t.id for t in self.tools]

    def to_prompt_dict(self, *, consumer: SkillConsumer) -> Dict[str, Any]:
        """
        Shape for the 'skills gallery' in prompts.
        Keep it compact and stable so it cache-hits well.
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category or "",
            "tags": self.tags,
            "tools": [asdict(t) for t in self.tools],
            "built_in": self.built_in,
            "include_for": list(self.include_for),
            # Consumer-specific hints can be added later if needed
        }


# -------------------- YAML loading --------------------


def _resolve_rel(path_str: str) -> pathlib.Path:
    """Resolve a path relative to BUNDLE_ROOT."""
    p = pathlib.Path(path_str)
    if not p.is_absolute():
        p = (BUNDLE_ROOT / p).resolve()
    return p


def _load_skill_yaml(path: pathlib.Path) -> SkillSpec:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Skill YAML {path} must contain a mapping at top level")

    sid = str(raw.get("id") or path.stem)
    name = str(raw.get("name") or sid)
    desc = str(raw.get("description") or "").strip()

    if not desc:
        log.warning("Skill %s (%s) has empty description", sid, path)

    # ---- instruction paths ----
    instr_conf = raw.get("instruction") or {}
    full_path = None
    compact_path = None

    if isinstance(instr_conf, str):
        # single path used as "full"
        full_path = _resolve_rel(instr_conf)
    elif isinstance(instr_conf, dict):
        full_str = instr_conf.get("full")
        compact_str = instr_conf.get("compact")
        if full_str:
            full_path = _resolve_rel(str(full_str))
        if compact_str:
            compact_path = _resolve_rel(str(compact_str))

    instr_paths = SkillInstructionPaths(full=full_path, compact=compact_path)

    # ---- tool refs ----
    tools_raw = raw.get("tools") or []
    tools: List[SkillToolRef] = []

    if isinstance(tools_raw, list):
        for item in tools_raw:
            if isinstance(item, str):
                tools.append(SkillToolRef(id=item.strip() or ""))
            elif isinstance(item, dict):
                tid = str(item.get("id") or "").strip()
                if not tid:
                    continue
                role = item.get("role")
                tools.append(SkillToolRef(id=tid, role=str(role) if role else None))
            else:
                log.warning(
                    "Skill %s: unexpected tools entry %r in %s", sid, item, path
                )

    # ---- flags & metadata ----
    built_in = bool(raw.get("built_in", False))

    include_for_raw = raw.get("include_for") or []
    include_for: List[SkillConsumer] = []
    if isinstance(include_for_raw, str):
        include_for_raw = [include_for_raw]

    for c in include_for_raw:
        c = str(c).strip().lower()
        if c in (COORDINATOR, REACT_DECISION, REACT_FOCUS):
            include_for.append(c)  # type: ignore[arg-type]
        elif c:
            log.warning("Skill %s: unknown consumer '%s' in %s", sid, c, path)

    category = raw.get("category")
    if category is not None:
        category = str(category)

    tags_raw = raw.get("tags") or []
    tags: List[str] = []
    if isinstance(tags_raw, str):
        tags_raw = [tags_raw]
    for t in tags_raw:
        s = str(t).strip()
        if s:
            tags.append(s)

    return SkillSpec(
        id=sid,
        name=name,
        description=desc,
        tools=tools,
        instruction_paths=instr_paths,
        built_in=built_in,
        include_for=include_for,
        category=category,
        tags=tags,
    )


def _iter_skill_files(skills_dir: Optional[pathlib.Path] = None) -> Iterable[pathlib.Path]:
    """Yield all *.yaml / *.yml files under skills/."""
    root = skills_dir or (BUNDLE_ROOT / "skills")
    if not root.exists():
        log.warning("Skills directory %s does not exist; no skills loaded", root)
        return []
    return sorted(
        p for p in root.glob("*.y*ml") if p.is_file()
    )


@lru_cache(maxsize=1)
def get_skill_registry() -> Dict[str, SkillSpec]:
    """
    Load all skills from skills/*.yaml under bundle root.

    The result is cached; call clear_skill_registry_cache() to reload.
    """
    registry: Dict[str, SkillSpec] = {}
    for path in _iter_skill_files():
        try:
            s = _load_skill_yaml(path)
        except Exception as e:
            log.error("Failed to load skill from %s: %s", path, e)
            continue

        if s.id in registry:
            log.warning(
                "Duplicate skill id %s from %s (existing from %s) – overriding",
                s.id,
                path,
                registry[s.id],
            )
        registry[s.id] = s

    log.info("Loaded %d skills: %s", len(registry), ", ".join(sorted(registry.keys())))
    return registry


def clear_skill_registry_cache() -> None:
    """Force reload of skills on next get_skill_registry() call."""
    get_skill_registry.cache_clear()  # type: ignore[attr-defined]


# -------------------- Public helpers --------------------


def get_skill(skill_id: str) -> Optional[SkillSpec]:
    return get_skill_registry().get(skill_id)


def skills_for_consumer(
        consumer: SkillConsumer,
        *,
        include_builtins: bool = True,
        include_non_builtins: bool = True,
) -> List[SkillSpec]:
    """
    Return skills relevant for a given consumer type.

    - consumer: COORDINATOR | REACT_DECISION | REACT_FOCUS
    - include_builtins: include skills with built_in=True
    - include_non_builtins: include skills with built_in=False
    """
    out: List[SkillSpec] = []
    for s in get_skill_registry().values():
        if consumer not in s.include_for:
            continue
        if s.built_in and not include_builtins:
            continue
        if (not s.built_in) and not include_non_builtins:
            continue
        out.append(s)
    return sorted(out, key=lambda s: s.id)


def skills_catalog_for_prompt(
        *,
        consumer: SkillConsumer,
        include_builtins: bool = True,
        include_non_builtins: bool = True,
) -> List[Dict[str, Any]]:
    """
    Shape for embedding into prompts as a 'skills gallery'.

    Example element:
      {
        "id": "web_search",
        "name": "Web search & evidence gathering",
        "description": "...",
        "category": "research",
        "tags": ["built_in", "discovery"],
        "tools": [{"id":"generic_tools.web_search","role":"search"}, ...],
        "built_in": true,
        "include_for": ["coordinator","react_decision"]
      }
    """
    skills = skills_for_consumer(
        consumer,
        include_builtins=include_builtins,
        include_non_builtins=include_non_builtins,
    )
    return [s.to_prompt_dict(consumer=consumer) for s in skills]


def validate_skill_tools(known_tool_ids: Iterable[str]) -> List[str]:
    """
    Optional sanity check: ensure all skill tool ids exist in the tool catalog.

    Returns a list of human-readable warning strings (no exception).
    You can call this once after ToolSubsystem initialization.
    """
    known = set(known_tool_ids)
    warnings: List[str] = []

    for skill in get_skill_registry().values():
        for t in skill.tools:
            if t.id not in known:
                msg = f"Skill '{skill.id}' references unknown tool id '{t.id}'"
                log.warning(msg)
                warnings.append(msg)
    return warnings
