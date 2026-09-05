#!/usr/bin/env python3
"""Run Claude Code through ClaudeCodeAgent as a direct SDK process."""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SDK_ROOT = REPO_ROOT / "app" / "ai-app" / "src" / "kdcube-ai-app"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(SDK_ROOT))

from kdcube_ai_app.apps.chat.sdk.runtime.direct_hosting.infrastructure import (  # noqa: E402
    activate_platform_descriptors,
    direct_harness_config,
)
from kdcube_ai_app.apps.chat.sdk.runtime.direct_hosting.configuration import (  # noqa: E402
    activate_configured_skills,
    agent_instructions,
    configured_run_directory,
    configured_tools,
    configured_web_search,
)
from kdcube_ai_app.apps.chat.sdk.runtime.direct_hosting.evidence import (  # noqa: E402
    ConsoleEmitter,
)
from kdcube_ai_app.apps.chat.sdk.runtime.direct_harness import (  # noqa: E402
    DirectAgentHarness,
)
from kdcube_ai_app.apps.chat.sdk.config import get_secret  # noqa: E402
from kdcube_ai_app.apps.chat.sdk.solutions.claude_code import (  # noqa: E402
    ClaudeCodeAgent,
    ClaudeCodeAgentConfig,
    ClaudeCodeBinding,
    ClaudeCodeSessionStoreConfig,
    ClaudeCodeWorkspaceConfig,
    bootstrap_claude_code_session_store,
    claude_code_session_branch_ref,
    publish_claude_code_session_store,
    run_claude_code_turn,
)


WEB_SEARCH_TOOL_ID = "mcp__kdcube_web_search__web_search"


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError("configuration root must be a mapping")
    return value


async def agent_config(
    config: dict[str, Any],
    *,
    workspace: Path,
    config_path: Path,
    check_only: bool,
    skill_ids: tuple[str, ...],
) -> ClaudeCodeAgentConfig:
    agent = config.get("agent") or {}
    adapter = agent.get("adapter") or {}
    if not isinstance(adapter, dict):
        raise ValueError("agent.adapter must be a mapping")
    command = str(adapter.get("command") or "claude")
    if not check_only and shutil.which(command) is None:
        raise RuntimeError(f"Claude Code executable {command!r} is not on PATH")
    api_key = str(
        await get_secret("services.anthropic.claude_code_key")
        or await get_secret("services.anthropic.api_key")
        or ""
    )
    python_bin = Path(sys.executable).resolve().parent
    env = {
        "PATH": os.pathsep.join((str(python_bin), os.environ.get("PATH", ""))),
        "VIRTUAL_ENV": str(python_bin.parent),
    }
    if api_key:
        env["CLAUDE_CODE_KEY"] = api_key
    tool_config = configured_tools(config)
    unsupported_runtimes = sorted(
        tool.id for tool in tool_config if tool.enabled and tool.runtime != "local"
    )
    if unsupported_runtimes:
        raise ValueError(
            "the direct Claude Code example runs CLI tools locally; unsupported runtime on: "
            + ", ".join(unsupported_runtimes)
        )
    allowed_tools = tuple(tool.id for tool in tool_config if tool.enabled)
    web_search_server_id = "kdcube_web_search"
    instructions = agent_instructions(
        config,
        fallback=(
            "You are a research agent. Work only inside this workspace, preserve public-web source "
            "URLs, create the requested deliverables, and report tool failures truthfully."
        ),
    )
    return ClaudeCodeAgentConfig(
        agent_name="standalone-claude",
        workspace_path=workspace,
        model=str(adapter.get("model") or "claude-haiku-4-5-20251001"),
        allowed_tools=allowed_tools,
        command=command,
        env=env,
        timeout_seconds=float(adapter.get("timeout_seconds") or 900),
        permission_mode="acceptEdits",
        workspace_config=ClaudeCodeWorkspaceConfig(
            mcp_servers={
                web_search_server_id: {
                    "type": "stdio",
                    "command": sys.executable,
                    "args": [
                        "-m",
                        "kdcube_ai_app.apps.chat.sdk.tools.mcp.web_search.web_search_server",
                        "--transport",
                        "stdio",
                        "--config",
                        str(config_path),
                        "--tool-id",
                        WEB_SEARCH_TOOL_ID,
                    ],
                    "env": {"PYTHONPATH": str(SDK_ROOT)},
                }
            },
            enabled_mcp_servers=(web_search_server_id,),
            instructions_markdown=instructions,
            allowed_tools=allowed_tools,
            denied_tools=("WebSearch", "WebFetch"),
            skill_ids=skill_ids,
            overwrite=True,
        ),
    )


def _git_repo(settings: Any, *, descriptors_dir: Path) -> str | None:
    raw = str(getattr(settings, "CLAUDE_CODE_SESSION_GIT_REPO", "") or "").strip()
    if not raw:
        return None
    if raw.startswith("git@") or urlparse(raw).scheme:
        return raw
    return str((descriptors_dir / raw).expanduser().resolve())


def session_store_config(
    settings: Any,
    *,
    descriptors_dir: Path,
    workspace: Path,
    conversation_id: str,
) -> ClaudeCodeSessionStoreConfig:
    return ClaudeCodeSessionStoreConfig(
        implementation=str(
            getattr(settings, "CLAUDE_CODE_SESSION_STORE_IMPLEMENTATION", "local")
            or "local"
        ),
        local_root=workspace / ".claude",
        tenant=str(getattr(settings, "TENANT", "") or "standalone"),
        project=str(getattr(settings, "PROJECT", "") or "claude-agent-demo"),
        user_id="demo-user",
        conversation_id=conversation_id,
        agent_name="standalone-claude",
        git_repo=_git_repo(settings, descriptors_dir=descriptors_dir),
    )


async def run_one_turn(
    *,
    prompt: str,
    number: int,
    resume: bool,
    config: ClaudeCodeAgentConfig,
    binding: ClaudeCodeBinding,
    harness: DirectAgentHarness,
    session_store: ClaudeCodeSessionStoreConfig,
) -> tuple[str, str]:
    turn_id = f"turn-{number:02d}-{uuid.uuid4().hex[:8]}"
    async with harness.turn(
        conversation_id=binding.conversation_id,
        turn_id=turn_id,
    ) as turn:
        agent = ClaudeCodeAgent(config=config, binding=binding, comm=turn.comm)
        await turn.comm.start(message=prompt)
        result = await run_claude_code_turn(
            agent=agent,
            prompt=prompt,
            resume_existing=resume,
            session_store=session_store,
        )
        if result.status != "completed":
            raise RuntimeError(result.error_message or f"Claude exited with {result.exit_code}")
        await turn.complete(prompt=prompt, final_answer=result.final_text)
    print(f"[accounting] Redis turn cache contains {len(turn.accounting_events)} event(s)")
    return result.final_text, turn_id


async def main_async(args: argparse.Namespace) -> None:
    config_path = Path(args.config).expanduser().resolve()
    descriptors_dir = Path(args.descriptors).expanduser().resolve()
    settings = activate_platform_descriptors(descriptors_dir)
    config = load_config(config_path)
    configured_web_search(config, tool_id=WEB_SEARCH_TOOL_ID)
    output = configured_run_directory(config, config_path=config_path)
    if args.check:
        conversation_id = "claude-check"
    elif args.infra_check:
        conversation_id = "claude-infra-check"
    else:
        conversation_id = str(
            (config.get("agent") or {}).get("conversation_id") or "claude-demo"
        ).strip()
    run_output = output / "runs" / conversation_id
    workspace = run_output / "workspace"
    _skills_subsystem, skill_config = activate_configured_skills(
        config,
        config_path=config_path,
        consumers=("claude",),
    )
    cfg = await agent_config(
        config,
        workspace=workspace,
        config_path=config_path,
        check_only=args.check or args.infra_check,
        skill_ids=skill_config.enabled,
    )
    harness_config = direct_harness_config(
        settings=settings,
        descriptors_dir=descriptors_dir,
        bundle_id="standalone-claude-demo@1-0",
        agent_id="claude",
        check_only=args.check,
    )
    session_store = session_store_config(
        settings,
        descriptors_dir=descriptors_dir,
        workspace=workspace,
        conversation_id=conversation_id,
    )
    print("mode: standalone SDK process")
    print("adapter: ClaudeCodeAgent -> local Claude Code subprocess")
    print(f"model: anthropic/{cfg.model}")
    print(f"tools: {', '.join(cfg.allowed_tools) or '(none)'}")
    print(
        "web search: KDCube Web Search MCP "
        f"({config_path}#agent.tools[id={WEB_SEARCH_TOOL_ID}].settings)"
    )
    print(f"skills: {', '.join(skill_config.enabled) or '(none)'}")
    print(f"workspace: {workspace}")
    print(f"conversation storage: {harness_config.storage_uri}")
    print(
        "Claude transcript store: "
        f"{session_store.implementation} -> {session_store.git_repo or session_store.local_root}"
    )
    if session_store.implementation == "git":
        print(f"Claude transcript branch: {claude_code_session_branch_ref(session_store)}")
    if args.check:
        binding = ClaudeCodeBinding(
            user_id="demo-user",
            conversation_id=conversation_id,
            session_id="local-session",
            claude_session_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"kdcube/demo/{conversation_id}")),
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
    emitter = ConsoleEmitter(run_output / "communicator.jsonl")
    harness = DirectAgentHarness(
        config=harness_config,
        model_service=None,
        emitter=emitter,
    )
    binding = ClaudeCodeBinding(
        user_id="demo-user",
        conversation_id=conversation_id,
        session_id="local-session",
        claude_session_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"kdcube/demo/{conversation_id}")),
    )
    topic = str((config.get("agent") or {}).get("topic") or "accountable agent runtimes")
    prompts = [
        (
            "Use the kdcube_web_search web_search MCP tool with use_llm=false and "
            f"fetch_content=false to research recent, concrete information about {topic}. "
            "Save five sourced findings to research.json so the next turn can use them."
        ),
        (
            "Continue this same session. Read research.json, then create "
            "deliverables/research-brief.pdf and deliverables/research-data.xlsx. "
            "Verify both files and report their exact paths."
        ),
    ]
    turn_ids: list[str] = []
    async with harness:
        if args.infra_check:
            await bootstrap_claude_code_session_store(config=session_store)
            await publish_claude_code_session_store(config=session_store)
            print(
                "infrastructure: Redis, Postgres conversation tables, storage, and "
                "Claude git session store ready"
            )
            print("infrastructure check: PASS")
            return
        print("infrastructure: Redis, Postgres conversation tables, and storage ready")
        expected_files = [
            workspace / "research.json",
            workspace / "deliverables" / "research-brief.pdf",
            workspace / "deliverables" / "research-data.xlsx",
        ]
        for path in expected_files:
            path.unlink(missing_ok=True)
        for number, prompt in enumerate(prompts, start=1):
            answer, turn_id = await run_one_turn(
                prompt=prompt,
                number=number,
                resume=number > 1,
                config=cfg,
                binding=binding,
                harness=harness,
                session_store=session_store,
            )
            turn_ids.append(turn_id)
            print(f"\n[turn {number} answer]\n{answer}\n")
        records = await harness.verify_conversation(
            conversation_id=conversation_id,
            expected_turn_ids=turn_ids,
        )
        print(f"[conversation] materialized {len(records)} durable turn record(s)")

    missing = [str(path.relative_to(workspace)) for path in expected_files if not path.is_file()]
    if missing:
        raise RuntimeError(f"agent completed without required artifacts: {', '.join(missing)}")
    print("demonstration: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(
            HERE / "config.local.yaml"
            if (HERE / "config.local.yaml").is_file()
            else HERE / "config.template.yaml"
        ),
    )
    parser.add_argument(
        "--descriptors",
        default=str(
            HERE / "descriptors.local"
            if (HERE / "descriptors.local").is_dir()
            else HERE / "descriptors.template"
        ),
        help="Directory containing the standard platform descriptor set.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Construct the direct SDK path without starting Claude Code.",
    )
    parser.add_argument(
        "--infra-check",
        action="store_true",
        help="Verify independent support services without starting Claude Code.",
    )
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
