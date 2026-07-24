# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Agentic configuration — agents configured, extended, reconfigured, tuned.

The solutions-level home for agent configuration as a first-class, managed
artifact: instruction sets are authored, stored with an id and a version,
and reattached to agents as quickly as tools and skills connect and
disconnect. Submodules own one concern each:

- ``instructions`` — the composition vocabulary, the ``instr:`` ref grammar,
  the versioned store, and the async ref expansion.
- ``projection`` — the forge's cost meter: the token breakdown of the REAL
  system instruction a DRAFT agent config would produce.

Runtime resolution (profile picks, rosters) stays in
``runtime/agent_inventory.py``; this package owns authoring, storage, and
presentation of the configuration itself.
"""

from __future__ import annotations

from kdcube_ai_app.apps.chat.sdk.solutions.agentic_config.instructions import (
    AgenticInstructionsStore,
    compose_instruction_body,
    expand_instruction_items,
    has_custom_instruction_refs,
)
from kdcube_ai_app.apps.chat.sdk.solutions.agentic_config.projection import (
    project_agent_config,
)

__all__ = [
    "AgenticInstructionsStore",
    "compose_instruction_body",
    "expand_instruction_items",
    "has_custom_instruction_refs",
    "project_agent_config",
]
