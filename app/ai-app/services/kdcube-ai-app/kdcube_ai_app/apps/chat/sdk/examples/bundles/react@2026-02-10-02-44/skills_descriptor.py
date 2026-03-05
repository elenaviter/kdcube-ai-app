# ── skills_descriptor.py ──
# Declares which skills (prompt templates) are available to the solver.
#
# A skill is a reusable prompt template that appears in the chat UI gallery
# (e.g. "Summarise this page", "Write a SQL query"). Skills live as
# SKILL.md files in a folder hierarchy:
#   <skills_root>/<namespace>/<skill_id>/SKILL.md
#
# The solver agent can invoke skills just like tools, but instead of
# running code they inject a curated prompt into the conversation.
#
# To add a custom skill:
#   1. Create a directory under skills/ (e.g. skills/custom/my_skill/)
#   2. Add a SKILL.md file with the prompt template
#   3. (Optional) Restrict visibility per agent in AGENTS_CONFIG

# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

from __future__ import annotations

import pathlib
from typing import Optional, Dict, Any

# Bundle root = directory containing this file
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
