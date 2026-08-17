# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
"""Reaching a Claude Code run that is already going.

The CLI's loop is not ours: no await point to cancel, no iteration to fold
into, and streaming input was measured to be a receipt only: a message written
to stdin mid-run is acknowledged in milliseconds and the running turn never
acts on it. A `PreToolUse` hook does reach it, before every tool call.

These tests run the SEEDED SCRIPT as the CLI runs it: a bare subprocess in the
workspace, reading its buffer, answering on stdout, because the thing that
matters is what a hook process actually prints, not what a function returns.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from kdcube_ai_app.apps.chat.sdk.solutions.claude_code import live_control as mod


def _fire_hook(workspace: Path) -> dict:
    """One tool call's worth of hook, exactly as the CLI invokes it."""
    hook = mod.control_dir(workspace) / mod.HOOK_FILENAME
    completed = subprocess.run(
        [sys.executable, str(hook)],
        input=json.dumps({"tool_name": "Read", "tool_input": {}}),
        capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    body = completed.stdout.strip()
    return json.loads(body) if body else {}


def test_a_quiet_run_is_left_alone(tmp_path):
    """Nothing arrived: the hook says nothing and the tool call proceeds. A
    hook that spoke on every call would put a note in front of every tool the
    agent uses."""
    mod.seed_live_control(tmp_path)

    assert _fire_hook(tmp_path) == {}


def test_a_message_reaches_the_run_without_stopping_it(tmp_path):
    """The follow-up path: the model reads what was said and keeps working."""
    mod.seed_live_control(tmp_path)
    mod.publish_arrivals(tmp_path, ["actually make it about the runtime"])

    out = _fire_hook(tmp_path)["hookSpecificOutput"]

    assert out["permissionDecision"] == "allow"
    assert "actually make it about the runtime" in out["permissionDecisionReason"]
    assert "actually make it about the runtime" in out["additionalContext"]


def test_a_stop_refuses_the_tool_call_and_says_what_to_do(tmp_path):
    """The stop: the call does not happen, and the model is told to answer with
    what it has rather than left to infer that its tools broke."""
    mod.seed_live_control(tmp_path)
    mod.publish_arrivals(tmp_path, [], stop=True)

    out = _fire_hook(tmp_path)["hookSpecificOutput"]

    assert out["permissionDecision"] == "deny"
    assert "Do not call any more tools" in out["permissionDecisionReason"]
    assert "Reply now with what you have" in out["permissionDecisionReason"]


def test_a_stop_carries_what_was_said_with_it(tmp_path):
    """"stop, use the other channel" is one thing: the refusal carries the
    words AND the instruction, in that order."""
    mod.seed_live_control(tmp_path)
    mod.publish_arrivals(tmp_path, ["use the other channel"], stop=True)

    reason = _fire_hook(tmp_path)["hookSpecificOutput"]["permissionDecisionReason"]

    assert reason.index("use the other channel") < reason.index("Do not call any more tools")


def test_a_message_is_said_once(tmp_path):
    """The hook fires before EVERY tool call. Repeating the same message in
    front of each one would read as the person saying it over and over."""
    mod.seed_live_control(tmp_path)
    mod.publish_arrivals(tmp_path, ["one thing"])

    first = _fire_hook(tmp_path)
    second = _fire_hook(tmp_path)

    assert first["hookSpecificOutput"]["permissionDecision"] == "allow"
    assert second == {}


def test_a_stop_keeps_refusing_after_its_message_is_delivered(tmp_path):
    """Delivered-once applies to the words, not to the stop: a run that could
    make a second tool call after being stopped was not stopped."""
    mod.seed_live_control(tmp_path)
    mod.publish_arrivals(tmp_path, ["that's wrong"], stop=True)

    first = _fire_hook(tmp_path)["hookSpecificOutput"]
    second = _fire_hook(tmp_path)["hookSpecificOutput"]

    assert first["permissionDecision"] == "deny"
    assert second["permissionDecision"] == "deny"


def test_the_hook_survives_a_buffer_it_cannot_read(tmp_path):
    """A broken buffer must not break the agent's tool call: the hook says
    nothing and the run continues unreachable, which is where it started."""
    mod.seed_live_control(tmp_path)
    mod.buffer_path(tmp_path).write_text("{not json", encoding="utf-8")

    assert _fire_hook(tmp_path) == {}


def test_the_settings_turn_it_on_for_every_tool(tmp_path):
    """A matcher naming a few tools leaves the run reachable through the ones
    it forgot: a stop that works sometimes."""
    args = mod.seed_live_control(tmp_path)

    assert args[0] == "--settings"
    settings = json.loads(Path(args[1]).read_text(encoding="utf-8"))
    entry = settings["hooks"]["PreToolUse"][0]
    assert entry["matcher"] == "*"
    assert mod.HOOK_FILENAME in entry["hooks"][0]["command"]


def test_seeding_into_an_unwritable_place_fails_open(tmp_path):
    """No live control is a run that behaves exactly as it did before this
    module existed, not a run that fails to start."""
    blocked = tmp_path / "file-not-a-dir"
    blocked.write_text("", encoding="utf-8")

    assert mod.seed_live_control(blocked / "workspace") == []
