#!/usr/bin/env python3
"""Run Claude Code through ClaudeCodeAgent without a KDCube deployment."""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

import yaml


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SDK_ROOT = HERE.parents[1] / "app" / "ai-app" / "src" / "kdcube-ai-app"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(SDK_ROOT))

from agents.infrastructure import (  # noqa: E402
    direct_harness_config,
)
from agents.evidence import (  # noqa: E402
    ConsoleEmitter,
)
from kdcube_ai_app.apps.chat.sdk.runtime.direct_harness import (  # noqa: E402
    DirectAgentHarness,
)
from kdcube_ai_app.apps.chat.sdk.solutions.claude_code import (  # noqa: E402
    ClaudeCodeAgent,
    ClaudeCodeAgentConfig,
    ClaudeCodeBinding,
    ClaudeCodeWorkspaceConfig,
)


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError("configuration root must be a mapping")
    return value


def agent_config(config: dict[str, Any], *, workspace: Path, check_only: bool) -> ClaudeCodeAgentConfig:
    claude = dict(config.get("claude") or {})
    command = str(claude.get("command") or "claude")
    if not check_only and shutil.which(command) is None:
        raise RuntimeError(f"Claude Code executable {command!r} is not on PATH")
    key_ref = str(claude.get("api_key_ref") or "CLAUDE_CODE_KEY")
    api_key = os.environ.get(key_ref, "")
    python_bin = Path(sys.executable).resolve().parent
    env = {
        "PATH": os.pathsep.join((str(python_bin), os.environ.get("PATH", ""))),
        "VIRTUAL_ENV": str(python_bin.parent),
    }
    if api_key:
        env["CLAUDE_CODE_KEY"] = api_key
    allowed_tools = tuple(str(item) for item in claude.get("allowed_tools") or ())
    instructions = (
        "You are the standalone Claude Code Agent Harness demonstration. Work only inside this "
        "workspace. Preserve public-web source URLs in research.json. When asked for deliverables, "
        "use Python with reportlab and openpyxl to create deliverables/research-brief.pdf and "
        "deliverables/research-data.xlsx. Report tool failures truthfully."
    )
    return ClaudeCodeAgentConfig(
        agent_name="standalone-claude",
        workspace_path=workspace,
        model=str(claude.get("model") or "sonnet"),
        allowed_tools=allowed_tools,
        command=command,
        env=env,
        timeout_seconds=float(claude.get("timeout_seconds") or 900),
        permission_mode="acceptEdits",
        workspace_config=ClaudeCodeWorkspaceConfig(
            instructions_markdown=instructions,
            allowed_tools=allowed_tools,
            overwrite=True,
        ),
    )


async def run_one_turn(
    *,
    prompt: str,
    number: int,
    resume: bool,
    config: ClaudeCodeAgentConfig,
    binding: ClaudeCodeBinding,
    harness: DirectAgentHarness,
) -> tuple[str, str]:
    turn_id = f"turn-{number:02d}-{uuid.uuid4().hex[:8]}"
    async with harness.turn(
        conversation_id=binding.conversation_id,
        turn_id=turn_id,
    ) as turn:
        agent = ClaudeCodeAgent(config=config, binding=binding, comm=turn.comm)
        await turn.comm.start(message=prompt)
        result = await agent.run_turn(prompt, resume_existing=resume)
        if result.status != "completed":
            raise RuntimeError(result.error_message or f"Claude exited with {result.exit_code}")
        await turn.complete(prompt=prompt, final_answer=result.final_text)
    print(f"[accounting] Redis turn cache contains {len(turn.accounting_events)} event(s)")
    return result.final_text, turn_id


async def main_async(args: argparse.Namespace) -> None:
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    output = (config_path.parent / str((config.get("output") or {}).get("directory") or "./output")).resolve()
    workspace = output / "workspace"
    cfg = agent_config(config, workspace=workspace, check_only=args.check or args.infra_check)
    harness_config = direct_harness_config(
        config,
        config_path=config_path,
        project="claude-agent-demo",
        bundle_id="standalone-claude-demo@1-0",
        agent_id="claude",
        check_only=args.check,
    )
    print("mode: standalone SDK process (no KDCube runtime)")
    print("adapter: ClaudeCodeAgent -> local Claude Code subprocess")
    print(f"workspace: {workspace}")
    print(f"conversation storage: {harness_config.storage_uri}")
    if args.check:
        binding = ClaudeCodeBinding(
            user_id="demo-user",
            conversation_id="claude-check",
            session_id="local-session",
            claude_session_id=str(uuid.uuid5(uuid.NAMESPACE_URL, "kdcube/demo/claude-check")),
        )
        emitter = ConsoleEmitter(Path(os.devnull))
        harness = DirectAgentHarness(
            config=harness_config,
            model_service=None,
            emitter=emitter,
        )
        agent = ClaudeCodeAgent(
            config=cfg,
            binding=binding,
            comm=harness.communicator(
                conversation_id=binding.conversation_id,
                turn_id="turn-check",
            ),
        )
        if not isinstance(agent, ClaudeCodeAgent):
            raise RuntimeError("ClaudeCodeAgent construction failed")
        print("check: PASS")
        return

    workspace.mkdir(parents=True, exist_ok=True)
    emitter = ConsoleEmitter(output / "communicator.jsonl")
    harness = DirectAgentHarness(
        config=harness_config,
        model_service=None,
        emitter=emitter,
    )
    conversation_id = f"claude-{uuid.uuid4().hex[:10]}"
    binding = ClaudeCodeBinding(
        user_id="demo-user",
        conversation_id=conversation_id,
        session_id="local-session",
        claude_session_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"kdcube/demo/{conversation_id}")),
    )
    topic = str((config.get("agent") or {}).get("topic") or "accountable agent runtimes")
    prompts = [
        f"Use WebSearch to research recent, concrete information about {topic}. Save five sourced findings to research.json so the next turn can use them.",
        "Continue this same session. Read research.json, then create deliverables/research-brief.pdf and deliverables/research-data.xlsx. Verify both files and report their exact paths.",
    ]
    turn_ids: list[str] = []
    async with harness:
        print("infrastructure: Redis, Postgres conversation tables, and storage ready")
        if args.infra_check:
            print("infrastructure check: PASS")
            return
        for number, prompt in enumerate(prompts, start=1):
            answer, turn_id = await run_one_turn(
                prompt=prompt,
                number=number,
                resume=number > 1,
                config=cfg,
                binding=binding,
                harness=harness,
            )
            turn_ids.append(turn_id)
            print(f"\n[turn {number} answer]\n{answer}\n")
        records = await harness.verify_conversation(
            conversation_id=conversation_id,
            expected_turn_ids=turn_ids,
        )
        print(f"[conversation] materialized {len(records)} durable turn record(s)")

    expected_files = [
        workspace / "research.json",
        workspace / "deliverables" / "research-brief.pdf",
        workspace / "deliverables" / "research-data.xlsx",
    ]
    missing = [str(path.relative_to(workspace)) for path in expected_files if not path.is_file()]
    if missing:
        raise RuntimeError(f"agent completed without required artifacts: {', '.join(missing)}")
    print("demonstration: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(HERE / "config.template.yaml"))
    parser.add_argument("--check", action="store_true", help="Construct the direct SDK path without starting Claude Code.")
    parser.add_argument(
        "--infra-check",
        action="store_true",
        help="Verify independent support services without starting Claude Code.",
    )
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
