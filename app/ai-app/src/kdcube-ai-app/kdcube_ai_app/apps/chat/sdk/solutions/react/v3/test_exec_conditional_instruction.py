# SPDX-License-Identifier: MIT

"""The code channel and everything exec is taught ONLY when the exec tool is
in the effective roster.

Surfaced case (2026-07-25): with the exec tool switched off, the rendered
system instruction still taught <channel:code> in every protocol block and
rendered the [TOOLS AVAILABLE ONLY IN CODE SNIPPET] catalog section — the
agent believed it could execute code. The contract now teaches the base
channels; a connected tool may extend the protocol with a channel of its
own, and the exec teaching (protocol extension, codegen body sections,
exec-only catalog) renders only with the tool present.
"""

from kdcube_ai_app.apps.chat.sdk.solutions.react.layout import (
    build_instruction_catalog_block,
)
from kdcube_ai_app.apps.chat.sdk.solutions.react.v2.agents.decision import (
    build_decision_system_text as v2_build,
)
from kdcube_ai_app.apps.chat.sdk.solutions.react.v3.agents.decision import (
    build_decision_system_text as v3_build,
)

WEB = {"id": "web_tools.web_search", "doc": {"purpose": "search"}, "call_template": "t"}
EXEC = {"id": "exec_tools.execute_code_python", "doc": {"purpose": "run python"}, "call_template": "t"}
IO = {"id": "io_tools.tool_call", "doc": {"purpose": "exec-only bridge"}, "call_template": "t"}

BODIES = (None, ["instr:profile:extra-lite"], ["instr:profile:lite"])

# ANY exec signal counts as a leak — the surfaced incident (2026-07-25 round
# two) was carried by soft mentions in generic tool docs ("exec/code",
# "inspect with code and exec tool"), not only by the channel teaching.
# Generic surfaces speak in class rules (file-processing tools, computation
# tools, physical paths); everything exec-specific lives with the tool.
import re

_EXEC_PATTERN = re.compile(
    r"channel:code|execute_code_python|ONLY IN CODE SNIPPET|fetch_ctx"
    r"|\bexec\b|exec[_/`-]|\bExec\b"
)
_BENIGN = re.compile(
    r"executes|execute yet|not to execute|improves execution|execute the write"
)


def _leaks(text: str) -> list[str]:
    found = []
    for line in text.splitlines():
        if _EXEC_PATTERN.search(line) and not _BENIGN.search(line):
            found.append(line.strip()[:120])
    return found


def test_no_exec_means_no_code_channel_anywhere():
    for blocks in BODIES:
        for build, kwargs in (
            (v3_build, {"multi_action_mode": "off"}),
            (v3_build, {"multi_action_mode": "on"}),
            (v2_build, {}),
        ):
            text = build(
                adapters=[WEB, IO],
                include_skill_gallery=False,
                instruction_blocks=blocks,
                **kwargs,
            )
            assert _leaks(text) == [], f"{build.__module__} blocks={blocks}: {_leaks(text)}"
            # the base contract teaches extensibility positively
            assert "may extend the protocol with a channel of its own" in text


def test_exec_present_teaches_the_code_channel():
    for blocks in BODIES:
        for build, kwargs in (
            (v3_build, {"multi_action_mode": "off"}),
            (v3_build, {"multi_action_mode": "on"}),
            (v2_build, {}),
        ):
            text = build(
                adapters=[WEB, IO, EXEC],
                include_skill_gallery=False,
                instruction_blocks=blocks,
                **kwargs,
            )
            assert "channel:code" in text
            assert "TOOLS AVAILABLE ONLY IN CODE SNIPPET" in text


def test_exec_only_catalog_section_requires_the_exec_tool():
    # exec-only helpers WITHOUT the exec tool: no code-snippet section —
    # those tools are uncallable and must not be advertised.
    without = build_instruction_catalog_block(
        consumer="test", tool_catalog=[WEB, IO], include_skill_gallery=False,
    )
    assert "TOOLS AVAILABLE ONLY IN CODE SNIPPET" not in without
    assert "io_tools.tool_call" not in without

    with_exec = build_instruction_catalog_block(
        consumer="test", tool_catalog=[WEB, IO, EXEC], include_skill_gallery=False,
    )
    assert "TOOLS AVAILABLE ONLY IN CODE SNIPPET" in with_exec
    assert "io_tools.tool_call" in with_exec
