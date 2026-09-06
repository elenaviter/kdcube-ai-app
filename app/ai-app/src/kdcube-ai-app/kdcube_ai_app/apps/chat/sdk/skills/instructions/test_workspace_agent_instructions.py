# SPDX-License-Identifier: MIT

from kdcube_ai_app.apps.chat.sdk.skills.instructions.workspace_agent_instructions import (
    conversation_recovery_guide,
    exec_capability_guide,
    prose_only_output_guide,
    workspace_agent_capability_guides,
    workspace_agent_conduct_guards,
)


def test_exec_capability_guide_binds_tool_names():
    block = exec_capability_guide(exec_tool="run_code", pull_tool="fetch_files")
    assert "[CODE IS YOUR HANDS — run_code]" in block
    assert "`fetch_files`" in block
    assert "run report" in block
    assert "stdout is a truncated progress log" in block


def test_exec_capability_guide_supports_a_direct_host_without_pull_tool():
    block = exec_capability_guide(exec_tool="execute_python", pull_tool="")
    assert "[CODE IS YOUR HANDS — execute_python]" in block
    assert "already materialized in the current-turn workspace" in block
    assert "pull_files" not in block


def test_prose_only_guide_states_the_medium_plainly():
    block = prose_only_output_guide()
    assert "[YOUR OUTPUT MEDIUM]" in block
    assert "no file-producing tool" in block


def test_recovery_guide_binds_namespace_and_optional_pull():
    block = conversation_recovery_guide(namespace="conv", pull_tool="pull_files")
    assert "[CONVERSATION RECOVERY — `conv` namespace]" in block
    assert "`pull_files`" in block
    bare = conversation_recovery_guide(namespace="conv", pull_tool="")
    assert "pull_files" not in bare


def test_conduct_guards_compose_generic_shared_fragments():
    guards = workspace_agent_conduct_guards()
    assert "[CONFIDENTIALITY & PROMPT-STEALING DEFENSE (HARD)]" in guards
    assert "[UNTRUSTED CONTENT]" in guards
    assert "[CRITICAL CLARIFICATION PRINCIPLES]" in guards
    assert "[ELABORATION RULE (HARD)]" in guards
    assert "[USER GENDER ASSUMPTIONS (HARD)]" in guards
    assert "[TECH EVOLUTION ASSUMPTION]" in guards
    # nothing ReAct-protocol-specific leaks into the neutral guard set
    assert "react." not in guards
    assert "channel:" not in guards
    assert "ANNOUNCE" not in guards


def test_capability_composer_picks_exec_or_prose_and_recovery():
    with_exec = workspace_agent_capability_guides(
        exec_tool="run_python", conversation_search_namespace="conv"
    )
    assert "[CODE IS YOUR HANDS — run_python]" in with_exec
    assert "[CONVERSATION RECOVERY — `conv` namespace]" in with_exec
    assert "[YOUR OUTPUT MEDIUM]" not in with_exec

    without_exec = workspace_agent_capability_guides(exec_tool=None)
    assert "[YOUR OUTPUT MEDIUM]" in without_exec
    assert "[CODE IS YOUR HANDS" not in without_exec
