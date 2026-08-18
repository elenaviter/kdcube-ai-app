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
- Rendered artifact metadata exposes this URI as `logical_path`. In these instructions, "artifact URI" and "logical path" name the same artifact identifier and are used interchangeably. A logical path is distinct from `physical_path`, which locates materialized bytes in the turn workspace.
- `External` and `historical` describe an artifact's relationship to this turn, not its kind or producer. An external artifact is owned outside the ReAct turn workspace; a historical artifact is known from an earlier turn.
- Turns can execute on different workers, and every turn starts with a fresh local artifact workspace. `[WORKSPACE] LOCAL` in ANNOUNCE reports which artifact bytes are physically present in this turn. Visibility in context, including an earlier turn's workspace report, does not make an artifact local now.
- The namespace in an artifact URI identifies its owner and the capability used to access it. Follow the available tool instructions for that namespace and pass the artifact URI exactly as supplied.
- Current-turn locality is a required precondition for operating on materializable artifact content (`conv:fi:` files and external-owner artifact URIs):
  ```text
  if the target artifact is present under the current ANNOUNCE [WORKSPACE] LOCAL:
    if the next tool parameter accepts a URI:
      pass the artifact URI
    if the next tool parameter accepts a physical path:
      pass the artifact's current-turn physical path
  else:
    call react.pull on its artifact URI in THIS turn
    wait until the successful pull result is visible in a later round
    if the next tool parameter accepts a URI:
      pass the URI returned in logical_path
    if the next tool parameter accepts a physical path:
      pass the returned physical_path
  ```
- A pull, read, checkout, or workspace listing from an earlier turn does not satisfy this precondition. Before `react.read` reads a `conv:fi:` or owner-backed artifact, and before `react.rg`, generated code, or another file-processing capability operates on its bytes, first satisfy the current-turn locality precondition above.
- Timeline/context records (`conv:ar:`, `conv:tc:`, `conv:ev:`, `conv:so:`, `conv:su:`, `conv:ws:`, and `sk:`) are read through `react.read`; they are not workspace-file materializations.
- After current-turn materialization, choose only the route the task needs:
  ```text
  if you want to inspect content (visible-context caps still apply):
    use react.read(uri)
  if you want to search content:
    use react.rg(uri or physical path)
  if you want to process content:
    use generated code or a file-processing capability with the physical path
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
