# ── skills_descriptor.py ──
# Declares skills available to the solver agent.
#
# A skill is more than a SKILL.md prompt template:
#   - May include sources added to the sources pool
#   - Can reference recommended tools
#   - Provides hints / best practices to guide the agent when relevant

# Skills are read by the agent and incorporated into its context
# (e.g., in ReAct agent, added directly to the timeline). Presentation
# of the skill catalog depends on the agent (flat list, tree by category, etc.).

# To add a custom skill:
#   1. Create a folder under skills/ (e.g. skills/custom/my_skill/)
#   2. Add SKILL.md with prompt template and optional metadata
#   3. Optionally restrict visibility via AGENTS_CONFIG

# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

from __future__ import annotations

import pathlib
from typing import Optional, Dict, Any

BUNDLE_ROOT = pathlib.Path(__file__).resolve().parent

# Root directory for custom skills shipped with this bundle.
# SDK scans for <namespace>/<skill_id>/SKILL.md files here.
# Set to None to disable bundle-local skills entirely.
CUSTOM_SKILLS_ROOT: Optional[pathlib.Path] = BUNDLE_ROOT / "skills"

# Per-agent skill visibility filters.
# Keys are agent role strings. Empty dict = show all available skills.
# Use fully qualified ids: "<namespace>.<skill_id>".
AGENTS_CONFIG: Dict[str, Dict[str, Any]] = {
    "solver.react.decision.v2": {
        # Optional filter example:
        # "enabled": [
        #     "public.url-gen",
        # ]
    }
}
