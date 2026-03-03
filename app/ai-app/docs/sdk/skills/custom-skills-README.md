---
id: ks:docs/sdk/skills/custom-skills-README.md
title: "Custom Skills"
summary: "Define bundle‑local skills: structure, sources/tools descriptors, and React integration."
tags: ["sdk", "skills", "custom", "bundle", "sources", "tools"]
keywords: ["skills_descriptor.py", "sources.yaml", "tools.yaml", "skill gallery", "react.read", "source pool"]
see_also:
  - ks:docs/sdk/skills/skills-README.md
  - ks:docs/sdk/skills/skills-infra-README.md
  - ks:docs/sdk/agents/react/source-pool-README.md
---
# Custom Skills (Bundle‑Local)

This is a quick guide to defining **bundle‑local skills**.

For full details, see:
- `docs/sdk/skills/skills-README.md`
- `docs/sdk/skills/skills-infra-README.md`

---

## Minimal pattern

1. Create a skills folder in your bundle.
2. Add a `skills_descriptor.py` to register skills.

Example bundle:
- `services/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/examples/bundles/react@2026-02-10-02-44`

Files:
- `skills/product`
- `skills_descriptor.py`

---

## What a skill can include

Inside a skill folder you can include:
- `SKILL.md` (required) with front‑matter fields:
  - `name`, `id`, `description`, `version`, `namespace`, `tags`, `when_to_use`, `imports`
- `compact.md` (optional): compact instruction variant
- `sources.yaml` (optional): sources injected into `sources_pool` when the skill is loaded
- `tools.yaml` (optional): recommended tools for the skill

Both `SKILL.md` and `skill.yml` are supported.

---

## Descriptor example

```python
# skills_descriptor.py
AGENTS_CONFIG = {
    # Apply to all agents if no agent-specific entry exists
    "default": {
        # Explicit allow-list (fully qualified skill ids)
        "enabled": [
            "product.kdcube",
        ]
    },

    # Agent-specific overrides (use the agent id)
    "solver.react.decision.v2": {
        # Example: disable a namespace for this agent only
        "disabled": ["public.*"],
    },
}
```

---

## Sources + tools metadata

Skills can include:
- `sources.yaml` — sources added to the **sources_pool** when the skill is loaded (via `react.read`).
- `tools.yaml` — tools recommended for this skill (used by planners/UX).

When a skill is loaded with `react.read("sk:<skill>")`, its sources are merged into
`sources_pool` and can be referenced as `so:sources_pool[...]`. Citation tokens inside
skill text are rewritten to match the merged source IDs.

---

## Notes

- Skills are prompt‑time modifiers, not tools.
- Use skills to specialize domain knowledge and response style.
- Skills can be scoped per agent role.

---

## AGENTS_CONFIG semantics (how filtering works)

`AGENTS_CONFIG` controls **which skills are visible** to a given agent.
Skill IDs are **fully qualified** as:

```
<namespace>.<skill_id>
```

Examples:
- `product.kdcube`
- `public.docx-press`

### Resolution rules

1. If an agent has a specific entry (e.g. `solver.react.decision.v2`), it is used.
2. Otherwise, the `default` entry applies.
3. You can define either:
   - `enabled`: allow‑list (only these skills are visible)
   - `disabled`: deny‑list (hide these skills)
4. Patterns are supported:
   - `namespace.*` hides or enables an entire namespace
   - glob patterns like `product.*` or `public.docx-*`

### Example: allow only product skills for ReAct

```python
AGENTS_CONFIG = {
    "solver.react.decision.v2": {
        "enabled": ["product.*"]
    }
}
```

### Example: hide built‑in public skills, keep custom

```python
AGENTS_CONFIG = {
    "default": {
        "disabled": ["public.*"]
    }
}
```

---

## References (code)

- Example bundle: `services/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/examples/bundles/react@2026-02-10-02-44`
- Descriptor: `services/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/examples/bundles/react@2026-02-10-02-44/skills_descriptor.py`
- Example skill: `services/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/examples/bundles/react@2026-02-10-02-44/skills/product`
