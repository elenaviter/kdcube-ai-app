# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

"""Stable ReAct harness facts that every decision instruction receives."""

from __future__ import annotations


REACT_HARNESS_TOOL_AVAILABILITY = """
[TOOL AVAILABILITY]
- The current tool catalogs are the authority for capabilities available in this conversation. Tool families are configuration-dependent and can include web search and fetch, rendering, code execution, and integrations.
- Use a tool-dependent strategy when its required tool ID appears in the current catalog.
""".strip()


REACT_HARNESS_CONTEXT_AND_ARTIFACT_ACCESS = """
[REACT HARNESS CONTEXT AND ARTIFACT ACCESS]
- You receive a rendered chronological context stream of artifacts. Every artifact has a URI, and its representation in context may be complete, partial, or only a reference.
- `External` and `historical` describe an artifact's relationship to this turn, not its kind or producer. An external artifact is owned outside the ReAct turn workspace; a historical artifact is known from an earlier turn.
- Turns can execute on different workers, and every turn starts with a fresh local artifact workspace. `[WORKSPACE] LOCAL` in ANNOUNCE reports which artifact bytes are physically present in this turn. Visibility in context, including an earlier turn's workspace report, does not make an artifact local now.
- The namespace in an artifact URI identifies its owner and the capability used to access it. Follow the available tool instructions for that namespace and pass the artifact URI exactly as supplied.
- When an external or historical artifact must be used beyond its visible context representation, call `react.pull` in THIS turn. After pulling, choose only the route the task needs:
  ```text
  if you want to inspect content (visible-context caps still apply):
    use react.read(logical_path)
  if you want to search content:
    use react.rg(materialized path)
  if you want to process content:
    use generated code or a file-processing capability with physical_path
  if you want to edit historical project content:
    use react.checkout(pulled git/projects ref) to create an editable current-turn project copy
  ```
- Pulled content is reference material. Edit historical `git/projects/...` state through the current-turn copy created by `react.checkout`. For other pulled artifacts, preserve the source and write transformed content as a new current-turn artifact.
- Let the artifact's reported shape choose the inspection route:
  ```text
  if the visible representation contains the evidence you need:
    use it
  if text fit or line shape is unknown:
    use react.read(stats_only=true)
  if fits_visible_context=true:
    use one whole read
  if fits_visible_context=false:
    choose the task-relevant route: react.rg, bounded line/symbol reads, or programmatic inspection that writes small derived artifacts
  ```
""".strip()


__all__ = [
    "REACT_HARNESS_CONTEXT_AND_ARTIFACT_ACCESS",
    "REACT_HARNESS_TOOL_AVAILABILITY",
]
