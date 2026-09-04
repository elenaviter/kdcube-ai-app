"""CLI orchestration shared by the native, LangGraph, and Claude launchers."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path

from runtime_client import (
    AgentTarget,
    DemoError,
    RuntimeChatClient,
    default_artifact_prompt,
    default_research_prompt,
    discover_workdir,
    load_bearer_token,
    load_runtime_descriptor,
    require_bundle,
    require_exec_image,
    validate_demonstration,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def _parser(target: AgentTarget) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Run the KDCube Agent Harness demonstration through {target.description}."
    )
    parser.add_argument("--workdir", help="Local KDCube runtime workdir. Auto-detected when unique.")
    parser.add_argument("--base-url", help="Override the ingress URL discovered from assembly.yaml.")
    parser.add_argument(
        "--token-file",
        help="Untracked file containing a bearer token or JSON with access_token. Otherwise prompt securely.",
    )
    parser.add_argument("--conversation-id", help="Continue an existing conversation instead of creating one.")
    parser.add_argument("--evidence-dir", help="Directory for events.jsonl. Defaults under runtime data.")
    parser.add_argument("--turn-timeout", type=float, default=900.0, help="Seconds allowed per turn.")
    parser.add_argument("--raw-events", action="store_true", help="Print complete SSE envelopes.")
    parser.add_argument("--preflight-only", action="store_true", help="Check descriptors and execution image only.")
    parser.add_argument(
        "--skip-exec-image-check",
        action="store_true",
        help="Skip local Docker image inspection for a remote/non-Docker execution profile.",
    )
    parser.add_argument("--research-prompt", default=default_research_prompt())
    parser.add_argument("--artifact-prompt", default=default_artifact_prompt())
    return parser


def main_for(target: AgentTarget) -> None:
    args = _parser(target).parse_args()
    try:
        workdir = discover_workdir(args.workdir)
        runtime = load_runtime_descriptor(workdir, base_url=args.base_url)
        require_bundle(runtime, target)
        if target.needs_exec_image and not args.skip_exec_image_check:
            require_exec_image(runtime, repo_root=REPO_ROOT)
        print(f"adapter: {target.adapter}")
        print(f"target: {target.bundle_id} / {target.agent_id}")
        print(f"runtime: {runtime.tenant}/{runtime.project} at {runtime.base_url}")
        print(f"workdir: {runtime.workdir}")
        if target.needs_exec_image:
            print(f"isolated execution image: {runtime.exec_image}")
        else:
            print("execution boundary: Claude Code tools in the trusted processor workspace")
        if args.preflight_only:
            print("preflight: PASS")
            return

        token = load_bearer_token(args.token_file)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        evidence_dir = (
            Path(args.evidence_dir).expanduser().resolve()
            if args.evidence_dir
            else runtime.workdir / "data" / "harness-demos" / target.adapter / stamp
        )
        evidence_path = evidence_dir / "events.jsonl"

        async def _run() -> None:
            async with RuntimeChatClient(
                runtime,
                bearer_token=token,
                evidence_path=evidence_path,
                raw_events=args.raw_events,
            ) as client:
                first = await client.submit_turn(
                    target=target,
                    text=args.research_prompt,
                    conversation_id=args.conversation_id,
                    timeout_seconds=args.turn_timeout,
                )
                first.require_baseline()
                print("\n\nresearch turn: PASS")
                second = await client.submit_turn(
                    target=target,
                    text=args.artifact_prompt,
                    conversation_id=first.conversation_id,
                    timeout_seconds=args.turn_timeout,
                )
                validate_demonstration(first, second)
                print("\n\ndemonstration: PASS")
                print(f"conversation: {second.conversation_id}")
                print(f"files: {', '.join(sorted(second.hosted_file_names))}")
                print(f"evidence: {evidence_path}")

        asyncio.run(_run())
    except (DemoError, OSError, ValueError) as exc:
        raise SystemExit(f"demonstration: FAIL\n{exc}") from exc
