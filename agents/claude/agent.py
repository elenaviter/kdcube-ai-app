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
    platform_exec_profile,
)
from kdcube_ai_app.apps.chat.sdk.runtime.direct_hosting.configuration import (  # noqa: E402
    activate_configured_skills,
    configured_agent_input,
    configured_run_directory,
    configured_tools,
    configured_web_search,
    require_supported_tools,
    verify_docker_image,
    verify_playwright_chromium,
)
from kdcube_ai_app.apps.chat.sdk.runtime.direct_hosting.instructions import (  # noqa: E402
    DirectInstructionSelection,
    PROVIDER_NATIVE_WORKSPACE_FILES_PROFILE,
    compose_provider_native_instructions,
    configured_instruction_selection,
)
from kdcube_ai_app.apps.chat.sdk.runtime.direct_hosting.evidence import (  # noqa: E402
    ConsoleEmitter,
    print_evidence_summary,
    write_evidence_index,
)
from kdcube_ai_app.apps.chat.sdk.runtime.direct_hosting.model_service import (  # noqa: E402
    configured_model_selection,
)
from kdcube_ai_app.apps.chat.sdk.runtime.direct_hosting.workspace import (  # noqa: E402
    DirectTurnWorkspace,
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
EXEC_TOOL_ID = "mcp__kdcube_harness__execute_python"
RENDER_TOOL_IDS = (
    "mcp__kdcube_harness__write_pdf",
    "mcp__kdcube_harness__write_docx",
    "mcp__kdcube_harness__write_pptx",
)
BUILTIN_TOOL_IDS = ("Read", "Write", "Edit", "Grep", "Glob")
BUNDLE_ID = "standalone-claude-demo@1-0"
AGENT_ID = "claude"


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
    descriptors_dir: Path,
    run_root: Path,
    tool_events_path: Path,
    conversation_id: str,
    user_id: str,
    user_type: str,
    session_id: str,
    turn_id: str,
    bundle_id: str,
    agent_id: str,
    check_only: bool,
    skill_ids: tuple[str, ...],
    instruction_selection: DirectInstructionSelection,
    model: str,
) -> ClaudeCodeAgentConfig:
    agent = config.get("agent") or {}
    adapter = agent.get("adapter") or {}
    if not isinstance(adapter, dict):
        raise ValueError("agent.adapter must be a mapping")
    command = str(adapter.get("command") or "claude")
    if not check_only and shutil.which(command) is None:
        raise RuntimeError(f"Claude Code executable {command!r} is not on PATH")
    api_key = str(
        await get_secret("platform.services.anthropic.claude_code_key")
        or await get_secret("platform.services.anthropic.api_key")
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
    require_supported_tools(
        tool_config,
        supported={*BUILTIN_TOOL_IDS, WEB_SEARCH_TOOL_ID, EXEC_TOOL_ID, *RENDER_TOOL_IDS},
        adapter="Claude Code direct example",
    )
    unsupported_runtimes = sorted(
        tool.id
        for tool in tool_config
        if tool.enabled
        and tool.runtime != ("docker" if tool.id == EXEC_TOOL_ID else "local")
    )
    if unsupported_runtimes:
        raise ValueError(
            "Claude Code tool runtime does not match the sample contract: "
            + ", ".join(unsupported_runtimes)
        )
    allowed_tools = tuple(tool.id for tool in tool_config if tool.enabled)
    web_search_server_id = "kdcube_web_search"
    harness_server_id = "kdcube_harness"
    instructions = compose_provider_native_instructions(
        instruction_selection,
        exec_tool=EXEC_TOOL_ID if EXEC_TOOL_ID in allowed_tools else None,
        rendering_tools=tuple(
            tool_id for tool_id in RENDER_TOOL_IDS if tool_id in allowed_tools
        ),
        web_search_tool=(
            WEB_SEARCH_TOOL_ID if WEB_SEARCH_TOOL_ID in allowed_tools else None
        ),
        native_skill_ids=skill_ids,
    )
    return ClaudeCodeAgentConfig(
        agent_name=agent_id,
        workspace_path=workspace,
        model=model,
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
                },
                harness_server_id: {
                    "type": "stdio",
                    "command": sys.executable,
                    "args": [
                        "-m",
                        "kdcube_ai_app.apps.chat.sdk.runtime.direct_hosting.tool_server",
                        "--descriptors",
                        str(descriptors_dir),
                        "--run-root",
                        str(run_root),
                        "--events",
                        str(tool_events_path),
                        "--conversation-id",
                        conversation_id,
                        "--user-id",
                        user_id,
                        "--user-type",
                        user_type,
                        "--session-id",
                        session_id,
                        "--turn-id",
                        turn_id,
                        "--bundle-id",
                        bundle_id,
                        "--agent-id",
                        agent_id,
                        "--bundle-root",
                        str(HERE),
                        "--bundle-module",
                        "agent",
                    ],
                    "env": {"PYTHONPATH": str(SDK_ROOT)},
                },
            },
            enabled_mcp_servers=(web_search_server_id, harness_server_id),
            instructions_markdown=instructions,
            allowed_tools=allowed_tools,
            denied_tools=("WebSearch", "WebFetch", "Bash"),
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
    user_id: str,
    conversation_id: str,
    agent_id: str,
) -> ClaudeCodeSessionStoreConfig:
    return ClaudeCodeSessionStoreConfig(
        implementation=str(
            getattr(settings, "CLAUDE_CODE_SESSION_STORE_IMPLEMENTATION", "local")
            or "local"
        ),
        local_root=workspace / ".claude",
        tenant=str(getattr(settings, "TENANT", "") or "standalone"),
        project=str(getattr(settings, "PROJECT", "") or "claude-agent-demo"),
        user_id=user_id,
        conversation_id=conversation_id,
        agent_name=agent_id,
        git_repo=_git_repo(settings, descriptors_dir=descriptors_dir),
    )


async def run_one_turn(
    *,
    prompt: str,
    number: int,
    resume: bool,
    raw_config: dict[str, Any],
    config_path: Path,
    descriptors_dir: Path,
    run_root: Path,
    workspace: Path,
    skill_ids: tuple[str, ...],
    binding: ClaudeCodeBinding,
    harness: DirectAgentHarness,
    session_store: ClaudeCodeSessionStoreConfig,
    instruction_selection: DirectInstructionSelection,
    model: str,
    attachment_source: Path | None = None,
) -> tuple[str, str, Any]:
    turn_id = f"turn_{number:02d}_{uuid.uuid4().hex[:8]}"
    turn_workspace = DirectTurnWorkspace(run_root=run_root, turn_id=turn_id)
    async with harness.turn(
        conversation_id=binding.conversation_id,
        turn_id=turn_id,
    ) as turn:
        if attachment_source is not None:
            await turn.add_user_attachment(
                attachment_source,
                materialize_to=turn_workspace.current_attachment(
                    attachment_source.name
                ),
            )
        config = await agent_config(
            raw_config,
            workspace=workspace,
            config_path=config_path,
            descriptors_dir=descriptors_dir,
            run_root=run_root,
            tool_events_path=turn_workspace.turn_root / "mcp-events.jsonl",
            conversation_id=binding.conversation_id,
            user_id=binding.user_id,
            user_type=harness.config.user_type,
            session_id=binding.session_id,
            turn_id=turn_id,
            bundle_id=harness.config.bundle_id,
            agent_id=harness.config.agent_id,
            check_only=False,
            skill_ids=skill_ids,
            instruction_selection=instruction_selection,
            model=model,
        )
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
        expected = (
            (
                "research/research-data.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "external",
                EXEC_TOOL_ID,
            ),
            (
                "research/research-brief.html",
                "text/html",
                "internal",
                EXEC_TOOL_ID,
            ),
            (
                "research/research-brief.pdf",
                "application/pdf",
                "external",
                "mcp__kdcube_harness__write_pdf",
            ),
        )
        files: list[dict[str, Any]] = []
        for relpath, mime, visibility, tool_id in expected:
            path = turn_workspace.current_file(relpath)
            if not path.is_file():
                continue
            files.append(
                {
                    "type": "file",
                    "output": {
                        "type": "file",
                        "path": f"{turn_id}/files/{relpath}",
                        "filename": path.name,
                        "mime": mime,
                        "visibility": visibility,
                    },
                    "mime": mime,
                    "visibility": visibility,
                    "description": f"Claude demo output: {path.name}",
                    "resource_id": path.stem,
                    "tool_id": tool_id,
                }
            )
        if files:
            await turn.host_files(files=files, outdir=turn_workspace.runtime_outdir)
        await turn.persist_workspace(
            outdir=turn_workspace.runtime_outdir,
            workdir=turn_workspace.workdir,
            execution_id=f"{turn_id}_workspace",
        )
        await turn.complete(prompt=prompt, final_answer=result.final_text)
    print(f"[accounting] Redis turn cache contains {len(turn.accounting_events)} event(s)")
    return result.final_text, turn_id, turn


async def main_async(args: argparse.Namespace) -> None:
    config_path = Path(args.config).expanduser().resolve()
    descriptors_dir = Path(args.descriptors).expanduser().resolve()
    settings = activate_platform_descriptors(descriptors_dir)
    selected_model = configured_model_selection()
    if selected_model.provider != "anthropic":
        raise ValueError(
            "the Claude Code adapter requires models.default_llm_provider: "
            "anthropic; use Native or LangGraph for KDCube custom-model routing"
        )
    config = load_config(config_path)
    agent_input = configured_agent_input(
        config,
        user_id=args.user_id,
        conversation_id=args.conversation_id,
        session_id=args.session_id,
    )
    configured_web_search(config, tool_id=WEB_SEARCH_TOOL_ID)
    output = configured_run_directory(config, config_path=config_path)
    conversation_id = agent_input.conversation_id
    run_id = "check" if args.check else f"run_{uuid.uuid4().hex[:12]}"
    run_root = agent_input.run_path(output, run_id=run_id)
    workspace = run_root / "workspace"
    _skills_subsystem, skill_config = activate_configured_skills(
        config,
        config_path=config_path,
        consumers=("claude",),
    )
    instruction_selection = configured_instruction_selection(
        config,
        default_profile=PROVIDER_NATIVE_WORKSPACE_FILES_PROFILE,
        fallback_additional_instructions=(
            "You are a research agent. Work only inside this workspace, preserve public-web "
            "source URLs, create the requested deliverables, and report tool failures truthfully."
        ),
    )
    harness_config = direct_harness_config(
        settings=settings,
        descriptors_dir=descriptors_dir,
        bundle_id=BUNDLE_ID,
        agent_id=AGENT_ID,
        user_id=agent_input.user_id,
        user_type=agent_input.user_type,
        session_id=agent_input.session_id,
        check_only=args.check,
    )
    private_state_key = agent_input.continuity_key(
        tenant=harness_config.tenant,
        project=harness_config.project,
        agent_id=harness_config.agent_id,
    )
    binding = ClaudeCodeBinding(
        user_id=agent_input.user_id,
        conversation_id=conversation_id,
        session_id=agent_input.session_id,
        claude_session_id=str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"kdcube-agent:{private_state_key}",
            )
        ),
    )
    cfg = await agent_config(
        config,
        workspace=workspace,
        config_path=config_path,
        descriptors_dir=descriptors_dir,
        run_root=run_root,
        tool_events_path=run_root / "turn_check" / "mcp-events.jsonl",
        conversation_id=conversation_id,
        user_id=agent_input.user_id,
        user_type=agent_input.user_type,
        session_id=agent_input.session_id,
        turn_id="turn_check",
        bundle_id=BUNDLE_ID,
        agent_id=AGENT_ID,
        check_only=args.check or args.infra_check,
        skill_ids=skill_config.enabled,
        instruction_selection=instruction_selection,
        model=selected_model.model,
    )
    session_store = session_store_config(
        settings,
        descriptors_dir=descriptors_dir,
        workspace=workspace,
        user_id=agent_input.user_id,
        conversation_id=conversation_id,
        agent_id=harness_config.agent_id,
    )
    required_demo_tools = {WEB_SEARCH_TOOL_ID, EXEC_TOOL_ID, RENDER_TOOL_IDS[0]}
    missing_demo_tools = sorted(required_demo_tools.difference(cfg.allowed_tools))
    if missing_demo_tools:
        raise RuntimeError(
            "the built-in demonstration requires tools: "
            + ", ".join(missing_demo_tools)
        )
    exec_runtime = platform_exec_profile(settings)
    print("mode: standalone SDK process")
    print("adapter: ClaudeCodeAgent -> local Claude Code subprocess")
    print(f"model: {selected_model.provider}/{cfg.model}")
    print(f"tools: {', '.join(cfg.allowed_tools) or '(none)'}")
    print(f"instruction profile: {instruction_selection.profile}")
    print(
        "custom instructions: "
        + ("configured" if instruction_selection.additional_instructions else "(none)")
    )
    print(
        "web search: KDCube Web Search MCP "
        f"({config_path}#agent.tools[id={WEB_SEARCH_TOOL_ID}].settings)"
    )
    print(f"skills: {', '.join(skill_config.enabled) or '(none)'}")
    print(f"workspace: {workspace}")
    print(f"conversation storage: {harness_config.storage_uri}")
    print(f"user: {agent_input.user_id} ({agent_input.user_type})")
    print(f"session: {agent_input.session_id}")
    print(f"conversation: {agent_input.conversation_id}")
    print(f"agent state scope: {private_state_key}")
    print(
        "Claude transcript store: "
        f"{session_store.implementation} -> {session_store.git_repo or session_store.local_root}"
    )
    if session_store.implementation == "git":
        print(f"Claude transcript branch: {claude_code_session_branch_ref(session_store)}")
    if not args.check:
        image = verify_docker_image(exec_runtime)
        print(f"isolated execution image: {image}")
        browser = await verify_playwright_chromium()
        print(f"document renderer: Playwright {browser} ready")
    if args.check:
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
                turn_id="turn_check",
            ),
        )
        if not isinstance(agent, ClaudeCodeAgent):
            raise RuntimeError("ClaudeCodeAgent construction failed")
        print("check: PASS")
        return

    workspace.mkdir(parents=True, exist_ok=True)
    emitter = ConsoleEmitter(run_root / "communicator.jsonl")
    harness = DirectAgentHarness(
        config=harness_config,
        model_service=None,
        emitter=emitter,
    )
    topic = str((config.get("agent") or {}).get("topic") or "accountable agent runtimes")
    request_path = workspace / "research-request.md"
    request_path.write_text(
        f"# Research request\n\nResearch topic: {topic}\n\n"
        "Produce sourced findings, an XLSX evidence table, and a polished PDF brief.\n",
        encoding="utf-8",
    )
    prompts = [
        (
            "Read the attached research-request.md, then use the kdcube_web_search "
            "web_search MCP tool with use_llm=false and fetch_content=false to research "
            f"recent, concrete information about {topic}. Return five sourced findings and "
            "retain them for the next turn."
        ),
        (
            "Continue this same session and use the retained findings. Author complete Python "
            "and call the kdcube_harness execute_python MCP tool. The Python must use openpyxl "
            "to create files/research/research-data.xlsx and create polished, print-ready HTML "
            "at files/research/research-brief.html. Declare the XLSX external and the HTML "
            "internal in the artifact contract. After execution succeeds, call the "
            "kdcube_harness write_pdf MCP tool with source_path "
            "files/research/research-brief.html and output_path "
            "files/research/research-brief.pdf. Do not use Bash and do not construct PDF bytes "
            "in Python. Verify and report the exact paths."
        ),
    ]
    turn_ids: list[str] = []
    completed_turns: list[Any] = []
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
        for number, prompt in enumerate(prompts, start=1):
            answer, turn_id, turn = await run_one_turn(
                prompt=prompt,
                number=number,
                resume=number > 1,
                raw_config=config,
                config_path=config_path,
                descriptors_dir=descriptors_dir,
                run_root=run_root,
                workspace=workspace,
                skill_ids=skill_config.enabled,
                binding=binding,
                harness=harness,
                session_store=session_store,
                instruction_selection=instruction_selection,
                model=selected_model.model,
                attachment_source=request_path if number == 1 else None,
            )
            turn_ids.append(turn_id)
            completed_turns.append(turn)
            print(f"\n[turn {number} answer]\n{answer}\n")
        records = await harness.verify_conversation(
            conversation_id=conversation_id,
            expected_turn_ids=turn_ids,
        )
        print(f"[conversation] materialized {len(records)} durable turn record(s)")
        evidence_path = run_root / "evidence.json"
        evidence = write_evidence_index(
            evidence_path,
            config=harness_config,
            conversation_id=conversation_id,
            turns=completed_turns,
            conversation_records=records,
            adapter_evidence={
                "transcript_store": session_store.implementation,
                "transcript_branch": (
                    claude_code_session_branch_ref(session_store)
                    if session_store.implementation == "git"
                    else None
                ),
                "generated_source_archive_path": f"{turn_ids[-1]}/executions/*/pkg/user_code.py",
            },
        )
        print_evidence_summary(evidence_path, evidence)

    deliverable_workspace = DirectTurnWorkspace(
        run_root=run_root,
        turn_id=turn_ids[-1],
    )
    expected_files = (
        deliverable_workspace.current_file("research/research-brief.pdf"),
        deliverable_workspace.current_file("research/research-data.xlsx"),
    )
    missing = [path.name for path in expected_files if not path.is_file()]
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
    parser.add_argument("--user-id", help="Override agent.input.user_id.")
    parser.add_argument(
        "--conversation-id",
        help="Override agent.input.conversation_id and resume that conversation.",
    )
    parser.add_argument("--session-id", help="Override agent.input.session_id.")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
