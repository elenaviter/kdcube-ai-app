#!/usr/bin/env python3
"""Run the KDCube native ReAct agent as a direct SDK process."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

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
    agent_instructions,
    configured_run_directory,
    configured_web_search,
    verify_docker_image,
    verify_playwright_chromium,
)
from kdcube_ai_app.apps.chat.sdk.runtime.direct_hosting.evidence import (  # noqa: E402
    ConsoleEmitter,
    print_evidence_summary,
    utc_now,
    write_evidence_index,
)
from kdcube_ai_app.apps.chat.sdk.runtime.direct_hosting.workspace import (  # noqa: E402
    DirectTurnWorkspace,
)
from kdcube_ai_app.apps.chat.sdk.runtime.direct_hosting.model_service import (  # noqa: E402
    build_model_service,
)
from kdcube_ai_app.apps.chat.sdk.runtime.direct_hosting.native_tool_bindings import (  # noqa: E402
    NativeToolBindings,
    NativeToolSource,
    resolve_native_tool_bindings,
)
from kdcube_ai_app.apps.chat.emitters import ChatCommunicator  # noqa: E402
from kdcube_ai_app.apps.chat.sdk.protocol import (  # noqa: E402
    ExternalEventActor,
    ExternalEventPayload,
    ExternalEventRequest,
    ExternalEventRouting,
    ExternalEventUser,
)
from kdcube_ai_app.apps.chat.sdk.runtime.scratchpad import TurnScratchpad  # noqa: E402
from kdcube_ai_app.apps.chat.sdk.runtime.tool_subsystem import ToolSubsystem  # noqa: E402
from kdcube_ai_app.apps.chat.sdk.runtime.direct_harness import (  # noqa: E402
    DirectAgentHarness,
)
from kdcube_ai_app.apps.chat.sdk.runtime.harness.timeline import (  # noqa: E402
    block_event_source_id,
)
from kdcube_ai_app.apps.chat.sdk.solutions.react.browser import ContextBrowser  # noqa: E402
from kdcube_ai_app.apps.chat.sdk.solutions.react.layout import (  # noqa: E402
    build_assistant_completion_blocks,
    build_user_input_blocks,
)
from kdcube_ai_app.apps.chat.sdk.solutions.react.proto import RuntimeCtx  # noqa: E402
from kdcube_ai_app.apps.chat.sdk.solutions.react.v3.runtime import ReactSolverV2  # noqa: E402
from kdcube_ai_app.infra.plugin.bundle_registry import BundleSpec  # noqa: E402
from kdcube_ai_app.infra.service_hub.inventory import AgentLogger, ModelServiceBase  # noqa: E402


ROLE = "solver.react.v2.decision.v2.strong"
WEB_SEARCH_TOOL_ID = "demo.web_search"
EXEC_TOOL_ID = "exec_tools.execute_code_python"
RENDER_TOOL_IDS = (
    "rendering_tools.write_pdf",
    "rendering_tools.write_docx",
    "rendering_tools.write_pptx",
)
TOOL_SOURCES = {
    "demo": NativeToolSource(
        path=HERE / "tools.py",
        tool_names=("web_search",),
    ),
    "exec_tools": NativeToolSource(
        module="kdcube_ai_app.apps.chat.sdk.tools.exec_tools",
        tool_names=("execute_code_python",),
        discovery="semantic_kernel",
    ),
    "rendering_tools": NativeToolSource(
        module="kdcube_ai_app.apps.chat.sdk.tools.rendering_tools",
        tool_names=("write_pdf", "write_docx", "write_pptx"),
        discovery="semantic_kernel",
    ),
}


class _ConstructionContextClient:
    """No-infrastructure context client used only by ``--check``."""

    class _Store:
        async def get_blob_bytes(self, uri_or_path: str) -> bytes:
            return Path(uri_or_path).read_bytes()

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = self._Store()

    def _path(self, kind: str) -> Path:
        safe = kind.replace("/", "_").replace(":", "_")
        return self.root / f"{safe}.json"

    async def save_artifact(self, *, kind: str, content: Any, **_: Any) -> dict[str, Any]:
        path = self._path(kind)
        path.write_text(json.dumps(content, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return {"ok": True, "path": str(path)}

    async def recent(self, *, kinds=(), **_: Any) -> dict[str, Any]:
        for raw_kind in kinds or ():
            kind = str(raw_kind).removeprefix("artifact:")
            path = self._path(kind)
            if path.is_file():
                return {
                    "items": [
                        {
                            "role": "artifact",
                            "payload": json.loads(path.read_text(encoding="utf-8")),
                        }
                    ]
                }
        return {"items": []}

    async def fetch_latest_feedback_reactions(self, *_: Any, **__: Any) -> dict[str, Any]:
        return {"items": []}


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    return config


def event_source_ids(blocks: list[dict[str, Any]]) -> set[str]:
    """Resolve tool/event source ids, including JSON-only ReAct call blocks."""
    sources: set[str] = set()
    call_meta: dict[str, dict[str, str]] = {}
    for block in blocks:
        if not isinstance(block, dict):
            continue
        meta = block.get("meta") if isinstance(block.get("meta"), dict) else {}
        tool_id = str(block.get("tool_id") or meta.get("tool_id") or "").strip()
        call_id = str(block.get("call_id") or meta.get("tool_call_id") or "").strip()
        if block.get("type") == "react.tool.call" and isinstance(block.get("text"), str):
            try:
                payload = json.loads(block["text"])
            except json.JSONDecodeError:
                payload = {}
            if isinstance(payload, dict):
                tool_id = tool_id or str(payload.get("tool_id") or "").strip()
                call_id = call_id or str(payload.get("tool_call_id") or "").strip()
        if tool_id:
            sources.add(tool_id)
        if call_id and tool_id:
            call_meta[call_id] = {"tool_id": tool_id}
    sources.update(
        source_id
        for source_id in (
            block_event_source_id(block, call_meta=call_meta) for block in blocks
        )
        if source_id
    )
    return sources


def comm_context(
    *,
    conversation_id: str,
    turn_id: str,
    tenant: str,
    project: str,
    bundle_id: str,
) -> ExternalEventPayload:
    return ExternalEventPayload(
        request=ExternalEventRequest(request_id=f"req-{turn_id}", payload={}),
        routing=ExternalEventRouting(
            session_id="local-session",
            conversation_id=conversation_id,
            turn_id=turn_id,
            bundle_id=bundle_id,
        ),
        actor=ExternalEventActor(tenant_id=tenant, project_id=project),
        user=ExternalEventUser(user_type="regular", user_id="demo-user", timezone="UTC"),
    )


async def build_turn(
    *,
    prompt: str,
    turn_id: str,
    conversation_id: str,
    root: Path,
    max_iterations: int,
    max_tokens: int,
    service: ModelServiceBase,
    context_client: Any,
    comm: ChatCommunicator,
    tool_bindings: NativeToolBindings,
    exec_runtime: dict[str, Any] | None,
    skills_subsystem: Any,
    skills_enabled: bool,
    instructions: str,
    harness_config: Any,
    hosting_service: Any = None,
    user_attachments: list[dict[str, Any]] | None = None,
) -> tuple[ReactSolverV2, ContextBrowser, ChatCommunicator, RuntimeCtx, ToolSubsystem]:
    workspace = DirectTurnWorkspace(run_root=root, turn_id=turn_id)
    outdir = workspace.runtime_outdir
    workdir = workspace.workdir
    runtime = RuntimeCtx(
        tenant=harness_config.tenant,
        project=harness_config.project,
        user_id="demo-user",
        user_type="regular",
        conversation_id=conversation_id,
        turn_id=turn_id,
        bundle_id=harness_config.bundle_id,
        agent_id=harness_config.agent_id,
        started_at=utc_now(),
        outdir=str(outdir),
        workdir=str(workdir),
        max_iterations=max_iterations,
        max_tokens=max_tokens,
        exec_runtime=exec_runtime or {"mode": "local"},
    )
    logger = AgentLogger("standalone.native")
    browser = ContextBrowser(ctx_client=context_client, logger=logger, model_service=service, runtime_ctx=runtime)
    await browser.load_timeline()
    browser.contribute(
        blocks=build_user_input_blocks(
            runtime=runtime,
            user_text=prompt,
            user_attachments=list(user_attachments or []),
            block_factory=browser.timeline.block,
            event_type="message",
        )
    )
    tools = ToolSubsystem(
        service=service,
        comm=comm,
        logger=logger,
        bundle_spec=BundleSpec(
            id=harness_config.bundle_id,
            path=str(HERE),
            module="agent",
        ),
        context_rag_client=context_client,
        tools_specs=list(tool_bindings.tool_specs),
        tool_runtime=tool_bindings.tool_runtime,
        hosting_service=hosting_service,
    )
    scratchpad = TurnScratchpad(
        "demo-user",
        conversation_id,
        turn_id,
        prompt,
        attachments=list(user_attachments or []),
    )
    solver = ReactSolverV2(
        service=service,
        logger=logger,
        tools_subsystem=tools,
        skills_subsystem=skills_subsystem,
        scratchpad=scratchpad,
        comm=comm,
        comm_context=comm_context(
            conversation_id=conversation_id,
            turn_id=turn_id,
            tenant=harness_config.tenant,
            project=harness_config.project,
            bundle_id=harness_config.bundle_id,
        ),
        ctx_browser=browser,
        hosting_service=hosting_service,
        instruction_body=instructions,
        include_tool_catalog=True,
        include_skill_gallery=skills_enabled,
        tool_catalog_detail="compact",
    )
    return solver, browser, comm, runtime, tools


async def run_turn(
    *,
    prompt: str,
    turn_number: int,
    conversation_id: str,
    root: Path,
    config: dict[str, Any],
    service: ModelServiceBase,
    harness: DirectAgentHarness,
    tool_bindings: NativeToolBindings,
    exec_runtime: dict[str, Any] | None,
    skills_subsystem: Any,
    skills_enabled: bool,
    instructions: str,
    attachment_source: Path | None = None,
) -> tuple[str, str, set[str], Any]:
    turn_id = f"turn_{turn_number:02d}_{uuid.uuid4().hex[:8]}"
    agent_config = dict(config.get("agent") or {})
    async with harness.turn(conversation_id=conversation_id, turn_id=turn_id) as turn:
        workspace = DirectTurnWorkspace(run_root=root, turn_id=turn_id)
        attachments: list[dict[str, Any]] = []
        if attachment_source is not None:
            attachments.append(
                await turn.add_user_attachment(
                    attachment_source,
                    materialize_to=workspace.current_attachment(attachment_source.name),
                )
            )
        solver, browser, comm, runtime, _tools = await build_turn(
            prompt=prompt,
            turn_id=turn_id,
            conversation_id=conversation_id,
            root=root,
            max_iterations=int(agent_config.get("max_iterations") or 8),
            max_tokens=int(agent_config.get("max_tokens") or 12000),
            service=service,
            context_client=turn.conversation_client,
            comm=turn.comm,
            tool_bindings=tool_bindings,
            exec_runtime=exec_runtime,
            skills_subsystem=skills_subsystem,
            skills_enabled=skills_enabled,
            instructions=instructions,
            harness_config=harness.config,
            hosting_service=turn.hosting_service,
            user_attachments=attachments,
        )
        await comm.start(message=prompt)
        result = await solver.run(
            allowed_plugins=tool_bindings.allowed_plugins,
            allowed_tool_names_by_alias=tool_bindings.allowed_tool_names_by_alias,
        )
        answer = str(result.final_answer or "")
        ended_at = utc_now()
        browser.contribute(
            blocks=build_assistant_completion_blocks(
                runtime=runtime,
                final_answer_text=answer,
                ended_at=ended_at,
                block_factory=browser.timeline.block,
            )
        )
        turn_blocks = browser.current_turn_blocks()
        event_sources = event_source_ids(turn_blocks)
        required = (
            ("research/research-data.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "external", EXEC_TOOL_ID),
            ("research/research-brief.html", "text/html", "internal", EXEC_TOOL_ID),
            ("research/research-brief.pdf", "application/pdf", "external", "rendering_tools.write_pdf"),
        )
        known = {str(row.get("filename") or "") for row in turn.assistant_files}
        pending: list[dict[str, Any]] = []
        for relpath, mime, visibility, tool_id in required:
            path = workspace.current_file(relpath)
            if not path.is_file() or path.name in known:
                continue
            pending.append(
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
                    "description": f"Native demo output: {path.name}",
                    "resource_id": path.stem,
                    "tool_id": tool_id,
                }
            )
        if pending:
            await turn.host_files(files=pending, outdir=workspace.runtime_outdir)
        await turn.persist_workspace(
            outdir=workspace.runtime_outdir,
            workdir=workspace.workdir,
            execution_id=f"{turn_id}_workspace",
        )
        await browser.persist_timeline()
        await turn.complete(
            prompt=prompt,
            final_answer=answer,
            rich_blocks=turn_blocks,
            started_at=runtime.started_at or "",
            ended_at=ended_at,
        )
    print(f"[accounting] Redis turn cache contains {len(turn.accounting_events)} event(s)")
    return answer, turn_id, event_sources, turn


async def main_async(args: argparse.Namespace) -> None:
    config_path = Path(args.config).expanduser().resolve()
    descriptors_dir = Path(args.descriptors).expanduser().resolve()
    settings = activate_platform_descriptors(descriptors_dir)
    config = load_config(config_path)
    configured_web_search(config, tool_id=WEB_SEARCH_TOOL_ID)
    from kdcube_ai_app.apps.chat.sdk.tools.mcp.web_search import web_search_server

    web_search_server.load_config(config_path, tool_id=WEB_SEARCH_TOOL_ID)
    service = None if args.infra_check else await build_model_service(
        role=ROLE,
        check_only=args.check,
    )
    root = configured_run_directory(config, config_path=config_path)
    harness_config = direct_harness_config(
        settings=settings,
        descriptors_dir=descriptors_dir,
        bundle_id="standalone-native-demo@1-0",
        agent_id="native",
        check_only=args.check,
    )
    tool_bindings = resolve_native_tool_bindings(
        config,
        sources=TOOL_SOURCES,
        adapter_name="native direct example",
    )
    exec_runtime = None
    if EXEC_TOOL_ID in tool_bindings.enabled_ids:
        if tool_bindings.tool_runtime[EXEC_TOOL_ID] != "docker":
            raise ValueError(f"{EXEC_TOOL_ID} must use runtime: docker in this example")
        exec_runtime = platform_exec_profile(settings)
    skills_subsystem, skill_config = activate_configured_skills(
        config,
        config_path=config_path,
        consumers=(ROLE, "native"),
    )
    instructions = agent_instructions(
        config,
        fallback=(
            "You are a research agent. Use configured tools for public-web research and "
            "deliverable creation. Preserve source URLs and follow enabled skills."
        ),
    )
    print("mode: standalone SDK process")
    print(f"adapter: ReactSolverV2 ({ROLE})")
    if service is not None:
        selected_model = service.config.ensure_role(ROLE)
        print(f"model: {selected_model['provider']}/{selected_model['model']}")
    print(f"tools: {', '.join(tool_bindings.enabled_ids) or '(none)'}")
    print(
        "web search: KDCube Web Search "
        f"({config_path}#agent.tools[id={WEB_SEARCH_TOOL_ID}].settings)"
    )
    print(f"skills: {', '.join(skill_config.enabled) or '(none)'}")
    if exec_runtime:
        if not args.check:
            image = verify_docker_image(exec_runtime)
            print(f"isolated execution image: {image}")
    if "rendering_tools.write_pdf" in tool_bindings.enabled_ids and not args.check:
        browser = await verify_playwright_chromium()
        print(f"document renderer: Playwright {browser} ready")
    print(f"run directory: {root}")
    print(f"conversation storage: {harness_config.storage_uri}")
    if args.check:
        assert service is not None
        agent_config = dict(config.get("agent") or {})
        with tempfile.TemporaryDirectory(prefix="kdcube-native-check-") as temp_root:
            check_root = Path(temp_root)
            check_harness = DirectAgentHarness(
                config=harness_config,
                model_service=service,
                emitter=ConsoleEmitter(check_root / "communicator.jsonl"),
            )
            solver, _browser, _comm, _runtime, tools = await build_turn(
                prompt="Offline construction check.",
                turn_id="turn_check",
                conversation_id="native-check",
                root=check_root,
                max_iterations=int(agent_config.get("max_iterations") or 8),
                max_tokens=int(agent_config.get("max_tokens") or 12000),
                service=service,
                context_client=_ConstructionContextClient(check_root / "context"),
                comm=check_harness.communicator(
                    conversation_id="native-check",
                    turn_id="turn_check",
                ),
                tool_bindings=tool_bindings,
                exec_runtime=exec_runtime,
                skills_subsystem=skills_subsystem,
                skills_enabled=bool(skill_config.enabled),
                instructions=instructions,
                harness_config=harness_config,
            )
            tool_ids = {str(item.get("id") or "") for item in tools.tools_info}
            expected = set(tool_bindings.enabled_ids)
            if not isinstance(solver, ReactSolverV2) or not expected.issubset(tool_ids):
                raise RuntimeError(f"native adapter construction incomplete: tools={sorted(tool_ids)}")
        print("check: PASS")
        return
    conversation_id = f"native-{uuid.uuid4().hex[:10]}"
    run_root = root / "runs" / conversation_id
    run_root.mkdir(parents=True, exist_ok=True)
    emitter = ConsoleEmitter(run_root / "communicator.jsonl")
    harness = DirectAgentHarness(
        config=harness_config,
        model_service=service,
        emitter=emitter,
    )
    topic = str((config.get("agent") or {}).get("topic") or "accountable agent runtimes")
    if WEB_SEARCH_TOOL_ID not in tool_bindings.enabled_ids:
        raise RuntimeError(
            f"the built-in demonstration requires agent.tools id {WEB_SEARCH_TOOL_ID}"
        )
    if EXEC_TOOL_ID not in tool_bindings.enabled_ids:
        raise RuntimeError(
            "the built-in demonstration requires " + EXEC_TOOL_ID
        )
    if "rendering_tools.write_pdf" not in tool_bindings.enabled_ids:
        raise RuntimeError("the built-in demonstration requires rendering_tools.write_pdf")
    deliverable_prompt = (
        "Use the findings from the previous turn. You must author and execute Python with "
        "exec_tools.execute_code_python. The Python must use openpyxl to create "
        "files/research/research-data.xlsx and must create a polished, print-ready HTML brief "
        "at files/research/research-brief.html. Contract the XLSX as external and HTML as "
        "internal. After execution succeeds, call rendering_tools.write_pdf in a later round "
        "with the HTML artifact as its ref: content and write "
        "files/research/research-brief.pdf. Do not construct PDF bytes in Python. Verify and "
        "report the exact output paths."
    )
    request_path = run_root / "inputs" / "research-request.md"
    request_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_text(
        f"# Research request\n\nResearch topic: {topic}\n\nProduce sourced findings, an XLSX evidence table, and a polished PDF brief.\n",
        encoding="utf-8",
    )
    completed_turns: list[Any] = []
    async with harness:
        print("infrastructure: Redis, Postgres conversation tables, and storage ready")
        if args.infra_check:
            print("infrastructure check: PASS")
            return
        assert service is not None
        first, first_turn_id, _first_sources, first_turn = await run_turn(
            prompt=(
                f"The attached research-request.md asks about {topic}. "
                "Search the web for recent, concrete information about it. "
                "Return five sourced findings and retain them for the next turn."
            ),
            turn_number=1,
            conversation_id=conversation_id,
            root=run_root,
            config=config,
            service=service,
            harness=harness,
            tool_bindings=tool_bindings,
            exec_runtime=exec_runtime,
            skills_subsystem=skills_subsystem,
            skills_enabled=bool(skill_config.enabled),
            instructions=instructions,
            attachment_source=request_path,
        )
        completed_turns.append(first_turn)
        print(f"\n[first answer]\n{first}\n")
        second, second_turn_id, _second_sources, second_turn = await run_turn(
            prompt=deliverable_prompt,
            turn_number=2,
            conversation_id=conversation_id,
            root=run_root,
            config=config,
            service=service,
            harness=harness,
            tool_bindings=tool_bindings,
            exec_runtime=exec_runtime,
            skills_subsystem=skills_subsystem,
            skills_enabled=bool(skill_config.enabled),
            instructions=instructions,
        )
        completed_turns.append(second_turn)
        print(f"\n[second answer]\n{second}\n")
        records = await harness.verify_conversation(
            conversation_id=conversation_id,
            expected_turn_ids=(first_turn_id, second_turn_id),
        )
        print(f"[conversation] materialized {len(records)} durable turn record(s)")
        if bool((config.get("agent") or {}).get("cross_conversation_search", True)):
            recall_conversation_id = f"native-recall-{uuid.uuid4().hex[:10]}"
            recall, recall_turn_id, recall_sources, recall_turn = await run_turn(
                prompt=(
                    f"Call react.memsearch with scope='user' to find the research about {topic!r} "
                    "from my other conversation. Report one recovered source URL and say that it "
                    "came from cross-conversation recall."
                ),
                turn_number=3,
                conversation_id=recall_conversation_id,
                root=run_root,
                config=config,
                service=service,
                harness=harness,
                tool_bindings=tool_bindings,
                exec_runtime=exec_runtime,
                skills_subsystem=skills_subsystem,
                skills_enabled=bool(skill_config.enabled),
                instructions=instructions,
            )
            completed_turns.append(recall_turn)
            if "react.memsearch" not in recall_sources:
                raise RuntimeError("the recall turn completed without calling react.memsearch")
            await harness.verify_conversation(
                conversation_id=recall_conversation_id,
                expected_turn_ids=(recall_turn_id,),
            )
            print(f"\n[cross-conversation answer]\n{recall}\n")
            print("[conversation] react.memsearch recovered a different conversation for this user")
        evidence_path = run_root / "evidence.json"
        evidence = write_evidence_index(
            evidence_path,
            config=harness_config,
            conversation_id=conversation_id,
            turns=completed_turns[:2],
            conversation_records=records,
            adapter_evidence={
                "generated_source_archive_path": f"{second_turn_id}/executions/*/pkg/user_code.py",
                "cross_conversation_turn": completed_turns[2].turn_id if len(completed_turns) > 2 else None,
            },
        )
        print_evidence_summary(evidence_path, evidence)
    second_turn_root = run_root / second_turn_id
    missing = [
        name
        for name in ("research-brief.pdf", "research-data.xlsx")
        if not list(second_turn_root.rglob(name))
    ]
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
        help="Construct the direct SDK path without calling a provider.",
    )
    parser.add_argument(
        "--infra-check",
        action="store_true",
        help="Verify independent support services without calling a provider.",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
