"""Framework-neutral input and output contracts for direct agent channels."""

from __future__ import annotations

import asyncio
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

from kdcube_ai_app.apps.chat.sdk.runtime.direct_hosting.configuration import (
    ConfiguredAgentInput,
)


@dataclass(frozen=True)
class DirectInputAttachment:
    """One caller-provided file to host and materialize for a direct turn."""

    filename: str
    mime: str
    content: bytes = field(repr=False)

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        filename: str | None = None,
        mime: str | None = None,
    ) -> "DirectInputAttachment":
        source = Path(path).expanduser().resolve()
        name = Path(filename or source.name).name
        return cls(
            filename=name,
            mime=mime or mimetypes.guess_type(name)[0] or "application/octet-stream",
            content=source.read_bytes(),
        )


@dataclass(frozen=True)
class DirectTurnRequest:
    """One explicit caller message submitted to a directly hosted agent."""

    prompt: str
    user_id: str
    user_type: str
    session_id: str
    conversation_id: str
    attachments: tuple[DirectInputAttachment, ...] = ()
    source: str = "direct"
    source_id: str = ""

    @property
    def agent_input(self) -> ConfiguredAgentInput:
        return ConfiguredAgentInput(
            user_id=self.user_id,
            user_type=self.user_type,
            session_id=self.session_id,
            conversation_id=self.conversation_id,
        )


@dataclass(frozen=True)
class DirectTurnResult:
    """Completed direct turn in the shape needed by another transport."""

    answer: str
    turn_id: str
    turn_log: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def transport_payload(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "turn_log": dict(self.turn_log),
            **dict(self.metadata),
        }


DirectTurnRunner = Callable[[DirectTurnRequest], Awaitable[DirectTurnResult]]


async def add_direct_input_attachments(
    *,
    turn: Any,
    workspace: Any,
    attachments: Sequence[DirectInputAttachment],
    mirror_to: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Host and materialize caller files in the current direct-turn workspace."""
    hosted: list[dict[str, Any]] = []
    used_names: set[str] = set()
    for index, attachment in enumerate(attachments, start=1):
        base_name = (
            Path(str(attachment.filename or "")).name or f"attachment-{index}.bin"
        )
        name = base_name
        counter = 1
        while name in used_names:
            stem = Path(base_name).stem or "attachment"
            suffix = Path(base_name).suffix
            counter += 1
            name = f"{stem}-{counter}{suffix}"
        used_names.add(name)
        if mirror_to is not None:
            mirror = Path(mirror_to).expanduser().resolve() / name
            mirror.parent.mkdir(parents=True, exist_ok=True)
            mirror.write_bytes(attachment.content)
        hosted.append(
            await turn.add_user_attachment_bytes(
                attachment.content,
                filename=name,
                mime=attachment.mime,
                materialize_to=workspace.current_attachment(name),
            )
        )
    return hosted


def prompt_with_attachment_manifest(
    prompt: str,
    attachments: Sequence[Mapping[str, Any]],
) -> str:
    """Name current-turn attachment paths for provider-native agent loops."""
    rows = []
    for item in attachments:
        name = Path(str(item.get("filename") or "")).name
        if not name:
            continue
        mime = str(item.get("mime") or "application/octet-stream")
        rows.append(f"- `attachments/{name}` ({mime})")
    if not rows:
        return str(prompt or "")
    return "\n\n".join(
        (
            str(prompt or "").strip(),
            "[CALLER ATTACHMENTS FOR THIS TURN]\n" + "\n".join(rows),
        )
    ).strip()


async def completed_direct_turn_result(
    *,
    harness: Any,
    conversation_id: str,
    turn_id: str,
    answer: str,
    metadata: Mapping[str, Any] | None = None,
) -> DirectTurnResult:
    """Read back the just-persisted turn so transports use durable truth."""
    records = await harness.verify_conversation(
        conversation_id=conversation_id,
        expected_turn_ids=(turn_id,),
    )
    record = next(
        (
            item
            for item in records
            if str(item.get("turn_id") or "") == turn_id
            and isinstance(item.get("payload"), dict)
        ),
        None,
    )
    if record is None:
        raise RuntimeError(f"completed direct turn {turn_id!r} has no durable payload")
    return DirectTurnResult(
        answer=str(answer or ""),
        turn_id=turn_id,
        turn_log=dict(record["payload"]),
        metadata=dict(metadata or {}),
    )


async def run_terminal_chat(
    *,
    agent_input: ConfiguredAgentInput,
    run_turn: DirectTurnRunner,
    read_line: Callable[[str], str] = input,
    write_line: Callable[[str], Any] = print,
) -> None:
    """Read terminal messages and run each as another durable conversation turn."""
    write_line(
        "terminal chat ready; type a message, /exit to stop "
        f"(user={agent_input.user_id}, conversation={agent_input.conversation_id})"
    )
    while True:
        try:
            raw = await asyncio.to_thread(read_line, "you> ")
        except (EOFError, KeyboardInterrupt):
            write_line("")
            return
        prompt = str(raw or "").strip()
        if prompt.lower() in {"/exit", "/quit"}:
            return
        if not prompt:
            continue
        result = await run_turn(
            DirectTurnRequest(
                prompt=prompt,
                user_id=agent_input.user_id,
                user_type=agent_input.user_type,
                session_id=agent_input.session_id,
                conversation_id=agent_input.conversation_id,
                source="terminal",
            )
        )
        write_line(f"\nassistant> {result.answer}\n")


__all__ = [
    "DirectInputAttachment",
    "DirectTurnRequest",
    "DirectTurnResult",
    "DirectTurnRunner",
    "add_direct_input_attachments",
    "completed_direct_turn_result",
    "prompt_with_attachment_manifest",
    "run_terminal_chat",
]
