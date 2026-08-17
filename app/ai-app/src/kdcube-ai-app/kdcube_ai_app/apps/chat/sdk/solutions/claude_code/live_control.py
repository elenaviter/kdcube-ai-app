# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
#
# live_control.py - reaching a Claude Code run that is already going
#
# The lane can be watched while a hosted Claude Code turn runs (see the
# foreign-runtime watcher), but the CLI's agentic loop is not ours: there is no
# await point to cancel and no iteration to fold into. Streaming input was
# measured and does not help: a message written to stdin mid-run is ingested
# and acknowledged within milliseconds, and the running turn never acts on it.
#
# What DOES reach a live run is a `PreToolUse` hook. It fires before every tool
# call, a boundary finer than the turn and one we control rather than the
# CLI's scheduling, and its decision is read by the model:
#
#   * `allow` + a reason: the model reads the text and KEEPS WORKING. That is
#     how a follow-up reaches a run in flight.
#   * `deny` + a reason: the tool call does not happen and the model, out of
#     ways to continue, answers with what it has. That is the stop: one more
#     round, enforced, with nothing killed.
#
# Both were verified against the CLI before this module existed.
#
# The buffer is a plain JSON file in the per-turn workspace because the hook is
# a SUBPROCESS PER TOOL CALL: it must start, decide and exit in the time a
# person would not notice, so it reads a local file and never dials Redis.
#
# What the model reads is plain language. A hook that answered with a tagged
# token would be naming a concept the model was never taught, which is how
# invented structure gets seeded.

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

LOGGER = logging.getLogger("kdcube.claude_code.live_control")

#: Everything lives under the workspace's own `.claude/`, the same place the
#: agent definition and the MCP config are seeded.
CONTROL_DIRNAME = ".claude"
BUFFER_FILENAME = "kdcube-live-events.json"
HOOK_FILENAME = "kdcube-live-hook.py"
SETTINGS_FILENAME = "kdcube-live-settings.json"

STOP_INSTRUCTION = (
    "The user stopped this run. Do not call any more tools. Reply now with what "
    "you have: say briefly what you finished, and what you had not started."
)
ARRIVAL_PREFACE = "The user sent this while you were working:"


def control_dir(workspace: Path) -> Path:
    return Path(workspace) / CONTROL_DIRNAME


def buffer_path(workspace: Path) -> Path:
    return control_dir(workspace) / BUFFER_FILENAME


def read_buffer(workspace: Path) -> Dict[str, Any]:
    try:
        raw = buffer_path(workspace).read_text(encoding="utf-8")
    except OSError:
        return {"stop": False, "messages": []}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"stop": False, "messages": []}
    if not isinstance(parsed, dict):
        return {"stop": False, "messages": []}
    parsed.setdefault("stop", False)
    parsed.setdefault("messages", [])
    return parsed


def _write_buffer(workspace: Path, payload: Mapping[str, Any]) -> None:
    path = buffer_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written whole, then moved: the hook may read at any instant, and half a
    # JSON file is a hook that crashes in front of a tool call.
    temp = path.with_suffix(".writing")
    temp.write_text(json.dumps(dict(payload), ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def publish_arrivals(
    workspace: Path,
    messages: Sequence[str],
    *,
    stop: bool = False,
) -> None:
    """Put what arrived where the next tool call will read it.

    ``messages`` are the person's own words, in order. ``stop`` marks the run as
    stopped, which the hook turns into a refusal rather than a note.
    """
    buffer = read_buffer(workspace)
    pending: List[Dict[str, Any]] = list(buffer.get("messages") or [])
    for text in messages:
        body = str(text or "").strip()
        if body:
            pending.append({"text": body, "delivered": False})
    payload = {"stop": bool(buffer.get("stop")) or bool(stop), "messages": pending}
    _write_buffer(workspace, payload)
    LOGGER.info(
        "[claude-code] live control: %d message(s) pending, stop=%s",
        sum(1 for item in pending if not item.get("delivered")), payload["stop"],
    )


def undelivered(buffer: Mapping[str, Any]) -> List[str]:
    return [
        str(item.get("text") or "")
        for item in (buffer.get("messages") or [])
        if isinstance(item, dict) and not item.get("delivered") and str(item.get("text") or "").strip()
    ]


def hook_decision(buffer: Mapping[str, Any]) -> Dict[str, Any] | None:
    """What the hook answers for one tool call, or None to say nothing.

    Pure, so the decision is testable without a CLI: the seeded script is a thin
    shell around this shape.
    """
    fresh = undelivered(buffer)
    if buffer.get("stop"):
        reason = STOP_INSTRUCTION
        if fresh:
            reason = f"{ARRIVAL_PREFACE}\n" + "\n".join(fresh) + f"\n\n{STOP_INSTRUCTION}"
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }
    if fresh:
        note = f"{ARRIVAL_PREFACE}\n" + "\n".join(fresh)
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow",
                "permissionDecisionReason": note,
                "additionalContext": note,
            }
        }
    return None


#: The seeded hook. Self-contained on purpose: the CLI runs it as a bare
#: subprocess in the workspace, with no path to this package.
_HOOK_SOURCE = '''#!/usr/bin/env python3
"""Carries what the user said into a run that is already going.

Fires before every tool call. Reads the buffer the turn's watcher writes, and
either notes what arrived (the run continues) or refuses the call because the
user stopped it (the model answers with what it has). Marks what it delivered,
so the same message is not repeated before every later call.

Silent and harmless when there is nothing to say: an empty answer leaves the
tool call exactly as it was.
"""
import json, os, sys

BUFFER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kdcube-live-events.json")
STOP_INSTRUCTION = {stop_instruction!r}
ARRIVAL_PREFACE = {arrival_preface!r}


def main() -> int:
    try:
        sys.stdin.read()
    except Exception:
        pass
    try:
        with open(BUFFER, "r", encoding="utf-8") as handle:
            buffer = json.load(handle)
    except Exception:
        return 0
    if not isinstance(buffer, dict):
        return 0
    messages = [m for m in (buffer.get("messages") or []) if isinstance(m, dict)]
    fresh = [str(m.get("text") or "") for m in messages
             if not m.get("delivered") and str(m.get("text") or "").strip()]
    stop = bool(buffer.get("stop"))
    if not stop and not fresh:
        return 0

    if stop:
        reason = STOP_INSTRUCTION
        if fresh:
            reason = ARRIVAL_PREFACE + "\\n" + "\\n".join(fresh) + "\\n\\n" + STOP_INSTRUCTION
        decision = "deny"
        out = {{"hookSpecificOutput": {{"hookEventName": "PreToolUse",
                                     "permissionDecision": decision,
                                     "permissionDecisionReason": reason}}}}
    else:
        note = ARRIVAL_PREFACE + "\\n" + "\\n".join(fresh)
        out = {{"hookSpecificOutput": {{"hookEventName": "PreToolUse",
                                     "permissionDecision": "allow",
                                     "permissionDecisionReason": note,
                                     "additionalContext": note}}}}

    for message in messages:
        if not message.get("delivered"):
            message["delivered"] = True
    try:
        tmp = BUFFER + ".writing"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump({{"stop": stop, "messages": messages}}, handle, ensure_ascii=False)
        os.replace(tmp, BUFFER)
    except Exception:
        pass

    sys.stdout.write(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def seed_live_control(workspace: Path, *, matcher: str = "*") -> List[str]:
    """Seed the hook, its settings and an empty buffer into the workspace.

    Returns the CLI args that turn it on; pass them as the agent's
    ``extra_args``. Fails open: a workspace that cannot be written leaves the
    run exactly as it was, without live control, and says so in the log.
    """
    try:
        directory = control_dir(workspace)
        directory.mkdir(parents=True, exist_ok=True)
        hook = directory / HOOK_FILENAME
        hook.write_text(
            _HOOK_SOURCE.format(
                stop_instruction=STOP_INSTRUCTION,
                arrival_preface=ARRIVAL_PREFACE,
            ),
            encoding="utf-8",
        )
        hook.chmod(0o755)
        _write_buffer(workspace, {"stop": False, "messages": []})
        settings = directory / SETTINGS_FILENAME
        settings.write_text(
            json.dumps({
                "hooks": {
                    "PreToolUse": [{
                        # EVERY tool, MCP tools included. A matcher that names a
                        # few leaves the run reachable through the ones it
                        # forgot, which is a stop that works sometimes.
                        "matcher": matcher,
                        "hooks": [{
                            "type": "command",
                            "command": f"{_python_for_hook()} {hook}",
                        }],
                    }],
                }
            }, ensure_ascii=False),
            encoding="utf-8",
        )
        return ["--settings", str(settings)]
    except Exception:
        LOGGER.warning(
            "[claude-code] could not seed live control; the run will not be "
            "reachable mid-flight", exc_info=True,
        )
        return []


def _python_for_hook() -> str:
    import sys as _sys

    return _sys.executable or "python3"
