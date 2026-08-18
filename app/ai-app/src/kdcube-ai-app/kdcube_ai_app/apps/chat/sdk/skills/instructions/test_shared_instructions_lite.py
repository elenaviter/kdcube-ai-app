# SPDX-License-Identifier: MIT

"""The moderate (lite) tier's distillation contract.

Mirrors ``test_instructions_extra_lite.py``: every grammar-critical signal of
the full composed body must survive, and the size ladder must hold —
xlite < lite < full, with lite well under half the full body.
"""

from kdcube_ai_app.apps.chat.sdk.skills.instructions.shared_instructions_lite import (
    REACT_LITE_PROFILE_BLOCKS,
    default_lite_system_instruction,
)


def test_profiles_mirror_the_ladder_profile_names():
    assert set(REACT_LITE_PROFILE_BLOCKS) == {
        "core", "workspace", "workspace_exec", "document", "web", "all_capabilities",
    }


def test_workspace_profiles_state_current_turn_locality_as_a_critical_fact():
    for profile in ("workspace", "workspace_exec", "all_capabilities"):
        text = default_lite_system_instruction(profile)
        assert "CRITICAL DISTRIBUTED-WORKSPACE FACT — LOCAL THIS TURN" in text
        assert "only files listed under ANNOUNCE `[WORKSPACE] LOCAL`" in text
        assert "prior-turn pull is not a local file" in text
        assert "Apply this before every local file operation" in text


def test_hard_signals_survive_moderate_distillation():
    text = default_lite_system_instruction("all_capabilities")
    signals = [
        # path grammar + conversion
        "conv:ar:conv_<conversation_id>.turn_<id>.react.turn.index",
        "conv:ar:conv_<conversation_id>.plan.latest:<plan_id>",
        "conv:so:conv_<conversation_id>.sources_pool[1,3]",
        "turn_<id>/external/<event_kind>/attachments/<event_id>/<rel>",
        # strict param orders
        "path, channel, content, kind, then optional scratchpad",
        "path, channel, patch, kind",
        # citation forms
        "[[S:1,3]]",
        '<sup class="cite" data-sids="1,3">',
        '"citations": [{"path": "<json pointer>", "sids": [1,3]}]',
        # plan ack markers
        "✓ [1]", "✗ [1]", "… [2]",
        # exec contract semantics
        "params.contract",
        "BYTE-IDENTICAL",
        "agent_io_tools.tool_call",
        "FLIP YOUR DEFAULT",
        "Write every contracted artifact to `Path(OUTPUT_DIR) / filepath`",
        # workspace hard rule
        "EACH TURN STARTS BLANK",
        "CRITICAL DISTRIBUTED-WORKSPACE FACT — LOCAL THIS TURN",
        "only files listed under ANNOUNCE `[WORKSPACE] LOCAL`",
        # fetch_ctx namespace limits
        "`conv:ar:`, `conv:tc:`, and `conv:so:`",
        # canvas extensions
        ".mermaid/.mmd",
        # announce budget form
        "reactive bonus",
        # hide window
        "last 4 rounds",
        # press skills
        "sk:public.pdf-press",
        # live-event markers + steer semantics
        "[FOLLOWUP DURING TURN]",
        "[STEER DURING TURN]",
        # DONE semantics
        "the result record of that very action is visible and reports success",
        # ref binding
        'ref:<visible logical path>',
    ]
    for signal in signals:
        assert signal in text, f"moderate distillation lost signal: {signal!r}"


def test_size_ladder_holds():
    from kdcube_ai_app.apps.chat.sdk.skills.instructions.instructions_extra_lite import (
        default_extra_lite_system_instruction,
    )
    from kdcube_ai_app.apps.chat.sdk.solutions.react.decision_prompt import (
        build_default_decision_instruction_body,
    )

    full = build_default_decision_instruction_body(
        module_label="ReAct Action Module v3", workspace_implementation="git",
    )
    lite = default_lite_system_instruction("all_capabilities")
    xlite = default_extra_lite_system_instruction(
        "all_capabilities", workspace_implementation="git",
    )
    assert len(xlite) < len(lite) < len(full)
    assert len(lite) < len(full) / 2


def test_no_composition_meta_leaks_into_llm_text():
    text = default_lite_system_instruction("all_capabilities")
    assert "include this block only" not in text.lower()
    assert "include when" not in text.lower()
