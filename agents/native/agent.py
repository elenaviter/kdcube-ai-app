#!/usr/bin/env python3
"""Run the KDCube native ReAct agent directly, without a KDCube deployment."""

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
    utc_now,
)
from agents.model_service import build_model_service  # noqa: E402
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


def comm_context(*, conversation_id: str, turn_id: str) -> ExternalEventPayload:
    return ExternalEventPayload(
        request=ExternalEventRequest(request_id=f"req-{turn_id}", payload={}),
        routing=ExternalEventRouting(
            session_id="local-session",
            conversation_id=conversation_id,
            turn_id=turn_id,
            bundle_id="standalone-native-demo@1-0",
        ),
        actor=ExternalEventActor(tenant_id="standalone", project_id="native-agent-demo"),
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
) -> tuple[ReactSolverV2, ContextBrowser, ChatCommunicator, RuntimeCtx, ToolSubsystem]:
    turn_root = root / turn_id
    outdir = turn_root / "output"
    workdir = turn_root / "workspace"
    outdir.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)
    runtime = RuntimeCtx(
        tenant="standalone",
        project="native-agent-demo",
        user_id="demo-user",
        user_type="regular",
        conversation_id=conversation_id,
        turn_id=turn_id,
        bundle_id="standalone-native-demo@1-0",
        agent_id="native",
        started_at=utc_now(),
        outdir=str(outdir),
        workdir=str(workdir),
        max_iterations=max_iterations,
        max_tokens=max_tokens,
        exec_runtime={"mode": "local"},
    )
    logger = AgentLogger("standalone.native")
    browser = ContextBrowser(ctx_client=context_client, logger=logger, model_service=service, runtime_ctx=runtime)
    await browser.load_timeline()
    browser.contribute(
        blocks=build_user_input_blocks(
            runtime=runtime,
            user_text=prompt,
            user_attachments=[],
            block_factory=browser.timeline.block,
            event_type="message",
        )
    )
    tools = ToolSubsystem(
        service=service,
        comm=comm,
        logger=logger,
        bundle_spec=BundleSpec(
            id="standalone-native-demo@1-0",
            path=str(HERE),
            module="agent",
        ),
        context_rag_client=context_client,
        tools_specs=[{"ref": str(HERE / "tools.py"), "alias": "demo", "use_sk": False}],
        tool_runtime={"demo.web_search": "local", "demo.create_briefing": "local"},
    )
    scratchpad = TurnScratchpad("demo-user", conversation_id, turn_id, prompt, attachments=[])
    solver = ReactSolverV2(
        service=service,
        logger=logger,
        tools_subsystem=tools,
        skills_subsystem=None,
        scratchpad=scratchpad,
        comm=comm,
        comm_context=comm_context(conversation_id=conversation_id, turn_id=turn_id),
        ctx_browser=browser,
        instruction_body=(
            "You are a standalone research agent. Use the demo tools for public-web research "
            "and deliverable creation. Keep source URLs in findings. A later turn may refer to "
            "facts from earlier turns in this same local timeline."
        ),
        include_tool_catalog=True,
        include_skill_gallery=False,
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
) -> tuple[str, str]:
    turn_id = f"turn-{turn_number:02d}-{uuid.uuid4().hex[:8]}"
    agent_config = dict(config.get("agent") or {})
    async with harness.turn(conversation_id=conversation_id, turn_id=turn_id) as turn:
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
        )
        await comm.start(message=prompt)
        result = await solver.run(
            allowed_plugins=["demo"],
            allowed_tool_names_by_alias={"demo": ["web_search", "create_briefing"]},
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
        await browser.persist_timeline()
        await turn.complete(
            prompt=prompt,
            final_answer=answer,
            rich_blocks=browser.current_turn_blocks(),
            started_at=runtime.started_at or "",
            ended_at=ended_at,
        )
    print(f"[accounting] Redis turn cache contains {len(turn.accounting_events)} event(s)")
    return answer, turn_id


async def main_async(args: argparse.Namespace) -> None:
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    service = None if args.infra_check else build_model_service(
        config,
        role=ROLE,
        check_only=args.check,
    )
    root = (config_path.parent / str((config.get("output") or {}).get("directory") or "./output")).resolve()
    harness_config = direct_harness_config(
        config,
        config_path=config_path,
        project="native-agent-demo",
        bundle_id="standalone-native-demo@1-0",
        agent_id="native",
        check_only=args.check,
    )
    print("mode: standalone SDK process (no KDCube runtime)")
    print(f"adapter: ReactSolverV2 ({ROLE})")
    print(f"output: {root}")
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
                turn_id="turn-check",
                conversation_id="native-check",
                root=check_root,
                max_iterations=int(agent_config.get("max_iterations") or 8),
                max_tokens=int(agent_config.get("max_tokens") or 12000),
                service=service,
                context_client=_ConstructionContextClient(check_root / "context"),
                comm=check_harness.communicator(
                    conversation_id="native-check",
                    turn_id="turn-check",
                ),
            )
            tool_ids = {str(item.get("id") or "") for item in tools.tools_info}
            expected = {"demo.web_search", "demo.create_briefing"}
            if not isinstance(solver, ReactSolverV2) or not expected.issubset(tool_ids):
                raise RuntimeError(f"native adapter construction incomplete: tools={sorted(tool_ids)}")
        print("check: PASS")
        return
    root.mkdir(parents=True, exist_ok=True)
    emitter = ConsoleEmitter(root / "communicator.jsonl")
    harness = DirectAgentHarness(
        config=harness_config,
        model_service=service,
        emitter=emitter,
    )
    conversation_id = f"native-{uuid.uuid4().hex[:10]}"
    topic = str((config.get("agent") or {}).get("topic") or "accountable agent runtimes")
    async with harness:
        print("infrastructure: Redis, Postgres conversation tables, and storage ready")
        if args.infra_check:
            print("infrastructure check: PASS")
            return
        assert service is not None
        first, first_turn_id = await run_turn(
            prompt=f"Search the web for recent, concrete information about {topic}. Return five sourced findings and retain them for the next turn.",
            turn_number=1,
            conversation_id=conversation_id,
            root=root,
            config=config,
            service=service,
            harness=harness,
        )
        print(f"\n[first answer]\n{first}\n")
        second, second_turn_id = await run_turn(
            prompt=(
                "Use the findings from the previous turn. Call demo.create_briefing to create "
                "research-brief.pdf and research-data.xlsx, then report the exact filenames."
            ),
            turn_number=2,
            conversation_id=conversation_id,
            root=root,
            config=config,
            service=service,
            harness=harness,
        )
        print(f"\n[second answer]\n{second}\n")
        records = await harness.verify_conversation(
            conversation_id=conversation_id,
            expected_turn_ids=(first_turn_id, second_turn_id),
        )
        print(f"[conversation] materialized {len(records)} durable turn record(s)")
    missing = [name for name in ("research-brief.pdf", "research-data.xlsx") if not list(root.rglob(name))]
    if missing:
        raise RuntimeError(f"agent completed without required artifacts: {', '.join(missing)}")
    print("demonstration: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(HERE / "config.template.yaml"))
    parser.add_argument("--check", action="store_true", help="Construct the direct SDK path without calling a provider.")
    parser.add_argument(
        "--infra-check",
        action="store_true",
        help="Verify independent support services without calling a provider.",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
