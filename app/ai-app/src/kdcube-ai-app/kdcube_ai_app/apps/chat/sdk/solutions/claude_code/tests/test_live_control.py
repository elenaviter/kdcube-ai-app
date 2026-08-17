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


def _fire_hook(workspace: Path, turn_id: str = "turn-1") -> dict:
    """One tool call's worth of hook, exactly as the CLI invokes it — including
    the turn id the settings put on its command line."""
    hook = mod.control_dir(workspace) / mod.HOOK_FILENAME
    completed = subprocess.run(
        [sys.executable, str(hook), turn_id],
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
    mod.seed_live_control(tmp_path, turn_id="turn-1")

    assert _fire_hook(tmp_path) == {}


def test_a_message_reaches_the_run_without_stopping_it(tmp_path):
    """The follow-up path: the model reads what was said and keeps working."""
    mod.seed_live_control(tmp_path, turn_id="turn-1")
    mod.publish_arrivals(tmp_path, ["actually make it about the runtime"])

    out = _fire_hook(tmp_path)["hookSpecificOutput"]

    assert out["permissionDecision"] == "allow"
    assert "actually make it about the runtime" in out["permissionDecisionReason"]
    assert "actually make it about the runtime" in out["additionalContext"]


def test_a_stop_refuses_the_tool_call_and_says_what_to_do(tmp_path):
    """The stop: the call does not happen, and the model is told to answer with
    what it has rather than left to infer that its tools broke."""
    mod.seed_live_control(tmp_path, turn_id="turn-1")
    mod.publish_arrivals(tmp_path, [], stop=True)

    out = _fire_hook(tmp_path)["hookSpecificOutput"]

    assert out["permissionDecision"] == "deny"
    assert "Do not call any more tools" in out["permissionDecisionReason"]
    assert "Reply now with what you have" in out["permissionDecisionReason"]


def test_a_stop_carries_what_was_said_with_it(tmp_path):
    """"stop, use the other channel" is one thing: the refusal carries the
    words AND the instruction, in that order."""
    mod.seed_live_control(tmp_path, turn_id="turn-1")
    mod.publish_arrivals(tmp_path, ["use the other channel"], stop=True)

    reason = _fire_hook(tmp_path)["hookSpecificOutput"]["permissionDecisionReason"]

    assert reason.index("use the other channel") < reason.index("Do not call any more tools")


def test_a_message_is_said_once(tmp_path):
    """The hook fires before EVERY tool call. Repeating the same message in
    front of each one would read as the person saying it over and over."""
    mod.seed_live_control(tmp_path, turn_id="turn-1")
    mod.publish_arrivals(tmp_path, ["one thing"])

    first = _fire_hook(tmp_path)
    second = _fire_hook(tmp_path)

    assert first["hookSpecificOutput"]["permissionDecision"] == "allow"
    assert second == {}


def test_a_stop_keeps_refusing_after_its_message_is_delivered(tmp_path):
    """Delivered-once applies to the words, not to the stop: a run that could
    make a second tool call after being stopped was not stopped."""
    mod.seed_live_control(tmp_path, turn_id="turn-1")
    mod.publish_arrivals(tmp_path, ["that's wrong"], stop=True)

    first = _fire_hook(tmp_path)["hookSpecificOutput"]
    second = _fire_hook(tmp_path)["hookSpecificOutput"]

    assert first["permissionDecision"] == "deny"
    assert second["permissionDecision"] == "deny"


def test_the_hook_survives_a_buffer_it_cannot_read(tmp_path):
    """A broken buffer must not break the agent's tool call: the hook says
    nothing and the run continues unreachable, which is where it started."""
    mod.seed_live_control(tmp_path, turn_id="turn-1")
    mod.buffer_path(tmp_path).write_text("{not json", encoding="utf-8")

    assert _fire_hook(tmp_path) == {}


def test_the_settings_turn_it_on_for_every_tool(tmp_path):
    """A matcher naming a few tools leaves the run reachable through the ones
    it forgot: a stop that works sometimes."""
    args = mod.seed_live_control(tmp_path, turn_id="turn-1")

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


# ── delivered vs merely arrived ──────────────────────────────────────────────


def test_the_buffer_says_which_lane_events_the_model_read(tmp_path):
    """The distinction the reconciliation rests on.

    A hook fired means the model read those words INSIDE this turn and answered
    with them in view, so the turn owns them. A message that arrived while the
    run sat in one long tool call reached no hook and is still the next turn's.
    Only the buffer knows which is which.
    """
    mod.seed_live_control(tmp_path, turn_id="turn-1")
    mod.publish_arrivals(tmp_path, [
        {"text": "read this", "message_id": "m-1"},
    ])

    assert mod.delivered_message_ids(tmp_path) == []   # arrived, not yet read
    _fire_hook(tmp_path)                               # the model reads it
    assert mod.delivered_message_ids(tmp_path) == ["m-1"]


def test_what_no_tool_call_ever_reached_stays_unanswered(tmp_path):
    """The long-tool-call case, which is exactly when a person is most likely
    to be typing: nothing fires, so nothing is claimed as delivered and the
    handoff still owes them a turn."""
    mod.seed_live_control(tmp_path, turn_id="turn-1")
    mod.publish_arrivals(tmp_path, [{"text": "are you stuck?", "message_id": "m-2"}])

    assert mod.delivered_message_ids(tmp_path) == []


def test_a_stop_that_was_read_is_reported_as_read(tmp_path):
    """A stop the model saw ended the run one round later; a stop it never saw
    did not stop anything, and the difference has to be visible to whoever
    reports what happened."""
    mod.seed_live_control(tmp_path, turn_id="turn-1")
    mod.publish_arrivals(tmp_path, [{"text": "stop that", "message_id": "m-3"}], stop=True)
    _fire_hook(tmp_path)

    assert mod.delivered_message_ids(tmp_path) == ["m-3"]


# ── a stop belongs to ONE turn ───────────────────────────────────────────────


def test_a_stop_does_not_reach_the_next_turn(tmp_path):
    """The bug this guard exists for, seen live.

    The workspace is per CONVERSATION, so the buffer outlives the turn that
    wrote it. A stop written in one turn was still there in the next, and the
    hook denied EVERY tool call of every later turn — the agent reported being
    interrupted at the same command three turns running and began guessing at
    permission scoping. A stop belongs to one run.
    """
    mod.seed_live_control(tmp_path, turn_id="turn-1")
    mod.publish_arrivals(tmp_path, [], stop=True)
    assert _fire_hook(tmp_path, "turn-1")["hookSpecificOutput"]["permissionDecision"] == "deny"

    # The next turn, same workspace, before it reseeds: the stale stop is inert.
    assert _fire_hook(tmp_path, "turn-2") == {}


def test_reseeding_clears_the_previous_turns_stop(tmp_path):
    """And the reseed a new turn does is what actually resets it."""
    mod.seed_live_control(tmp_path, turn_id="turn-1")
    mod.publish_arrivals(tmp_path, ["stop that"], stop=True)

    mod.seed_live_control(tmp_path, turn_id="turn-2")
    assert _fire_hook(tmp_path, "turn-2") == {}
    assert mod.read_buffer(tmp_path)["stop"] is False


def test_nothing_is_seeded_into_the_session_stores_checkout(tmp_path):
    """WHY THIS LIVES SOMEWHERE ELSE, and the bug that taught it.

    `.claude/` is the session store's checkout: at the start of every turn it
    deletes everything there except `.git` and resets to the previous turn's
    snapshot. Seeding into it meant the fresh buffer and settings were wiped a
    second after they were written and REPLACED by last turn's pair — a stop
    with the matching turn id on the hook's command line — so every turn after
    a stop had its tool calls refused, while the first turn of a conversation
    worked because there was no snapshot to restore yet.
    """
    mod.seed_live_control(tmp_path, turn_id="turn-1")

    session_store_root = tmp_path / ".claude"
    seeded = list(session_store_root.rglob("kdcube-live-*")) if session_store_root.exists() else []
    assert seeded == []
    assert not str(mod.control_dir(tmp_path)).startswith(str(session_store_root) + "/")


def test_the_seed_survives_the_store_restoring_its_checkout(tmp_path):
    """The same thing, played out: the store wipes and restores `.claude`
    between the seed and the first tool call, and live control is unaffected."""
    args = mod.seed_live_control(tmp_path, turn_id="turn-2")

    store_root = tmp_path / ".claude"
    store_root.mkdir(parents=True, exist_ok=True)
    (store_root / "kdcube-live-events.json").write_text(
        json.dumps({"turn_id": "turn-1", "stop": True, "messages": []}), encoding="utf-8",
    )

    assert _fire_hook(tmp_path, "turn-2") == {}
    assert Path(args[1]).is_file()


def test_a_writer_from_a_finished_turn_cannot_stop_the_running_one(tmp_path):
    """The other half of the same bug, on the WRITE side.

    The reseed above resets the buffer, but nothing stopped a writer belonging
    to an earlier turn from putting a stop back into it a moment later: the
    buffer takes its turn id from the file, so such a write is stamped with the
    RUNNING turn's id and the hook honours it. That is a stop nobody pressed on
    this run, and it is indistinguishable after the fact from a real one.
    """
    mod.seed_live_control(tmp_path, turn_id="turn-1")
    mod.seed_live_control(tmp_path, turn_id="turn-2")

    mod.publish_arrivals(tmp_path, [], stop=True, turn_id="turn-1")

    assert mod.read_buffer(tmp_path)["stop"] is False
    assert _fire_hook(tmp_path, "turn-2") == {}

    # The running turn's own watcher still reaches it.
    mod.publish_arrivals(tmp_path, [], stop=True, turn_id="turn-2")
    assert _fire_hook(tmp_path, "turn-2")["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_an_unnamed_writer_is_still_honoured(tmp_path):
    """Saying which turn you are is optional: a caller that does not (tests,
    and any lane that has no turn id to give) behaves exactly as before."""
    mod.seed_live_control(tmp_path, turn_id="turn-1")
    mod.publish_arrivals(tmp_path, ["look at this"])
    assert "look at this" in _fire_hook(tmp_path, "turn-1")["hookSpecificOutput"][
        "permissionDecisionReason"
    ]


def test_the_settings_name_the_turn_on_the_hook_command(tmp_path):
    args = mod.seed_live_control(tmp_path, turn_id="turn-9")
    settings = json.loads(Path(args[1]).read_text(encoding="utf-8"))
    command = settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    assert command.endswith(" turn-9")
