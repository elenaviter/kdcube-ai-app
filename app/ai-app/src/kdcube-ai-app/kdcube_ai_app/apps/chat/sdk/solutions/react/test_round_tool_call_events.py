# SPDX-License-Identifier: MIT

import json
from types import SimpleNamespace

import pytest

import kdcube_ai_app.apps.chat.sdk.solutions.react.round as round_mod
from kdcube_ai_app.apps.chat.sdk.solutions.react.proto import RuntimeCtx
from kdcube_ai_app.apps.chat.sdk.solutions.react.round import ReactRound
from kdcube_ai_app.apps.chat.sdk.solutions.react.v2.tools.tests.helpers import FakeBrowser


class _FakeComm:
    def __init__(self):
        self.events = []

    async def service_event(self, **kwargs):
        self.events.append(kwargs)


@pytest.mark.asyncio
async def test_react_round_emits_redacted_tool_call_event(monkeypatch, tmp_path):
    async def _fake_rg(*, ctx_browser, state, tool_call_id):
        state["last_tool_result"] = []
        return state

    monkeypatch.setattr(round_mod.react_tools, "handle_react_rg", _fake_rg)

    runtime = RuntimeCtx(turn_id="turn_tool", outdir=str(tmp_path), workdir=str(tmp_path))
    ctx = FakeBrowser(runtime)
    comm = _FakeComm()
    react = SimpleNamespace(ctx_browser=ctx, comm=comm)
    state = {
        "pending_tool_origin_iteration": 2,
        "last_decision": {
            "tool_call": {
                "tool_id": "react.rg",
                "tool_call_id": "tc_rg",
                "params": {
                    "query": "sensitive search text",
                    "paths": ["conv:fi:turn_tool.files/report.md", "sk:public.skill"],
                    "items": [{"path": "ks:docs/readme.md"}],
                    "top_k": 5,
                },
            },
        },
    }

    await ReactRound.execute(react, state)

    assert len(comm.events) == 2
    call_event, result_event = comm.events
    assert call_event["type"] == "react.tool.call"
    assert call_event["step"] == "react.tool.call.tc_rg"
    assert call_event["status"] == "started"
    assert call_event["data"]["phase"] == "call"
    assert call_event["data"]["executed"] is False
    assert call_event["data"]["tool_id"] == "react.rg"
    assert call_event["data"]["tool_call_id"] == "tc_rg"
    assert call_event["data"]["iteration"] == 2
    assert call_event["data"]["params"]["redacted"] is True
    assert call_event["data"]["params"]["query_len"] == len("sensitive search text")
    assert call_event["data"]["params"]["paths_count"] == 2
    assert call_event["data"]["params"]["path_prefixes"] == ["conv:fi:", "sk:"]
    assert call_event["data"]["params"]["item_path_prefixes"] == ["ks:"]
    assert result_event["type"] == "react.tool.result"
    assert result_event["step"] == "react.tool.result.tc_rg"
    assert result_event["status"] == "completed"
    assert result_event["data"]["phase"] == "result"
    assert result_event["data"]["executed"] is True
    assert result_event["data"]["result"] == []
    assert "sensitive search text" not in json.dumps(comm.events, ensure_ascii=False)


@pytest.mark.asyncio
async def test_react_round_tool_call_event_reports_managed_error(monkeypatch, tmp_path):
    async def _fake_pull(*, react, ctx_browser, state, tool_call_id):
        state["last_tool_result"] = [{"error": {"code": "pull_failed"}}]
        return state

    monkeypatch.setattr(round_mod.react_tools, "handle_react_pull", _fake_pull)

    runtime = RuntimeCtx(turn_id="turn_tool", outdir=str(tmp_path), workdir=str(tmp_path))
    ctx = FakeBrowser(runtime)
    comm = _FakeComm()
    react = SimpleNamespace(ctx_browser=ctx, comm=comm)
    state = {
        "last_decision": {
            "tool_call": {
                "tool_id": "react.pull",
                "tool_call_id": "tc_pull",
                "params": {"paths": ["conv:fi:turn_tool.files/missing.md"]},
            },
        },
    }

    await ReactRound.execute(react, state)

    assert [event["type"] for event in comm.events] == ["react.tool.call", "react.tool.result"]
    event = comm.events[-1]
    assert event["step"] == "react.tool.result.tc_pull"
    assert event["status"] == "error"
    assert event["data"]["executed"] is True
    assert event["data"]["error_count"] == 1
    assert event["data"]["error_code"] == "pull_failed"
    assert event["data"]["error"]["items"] == [{"code": "pull_failed"}]
    assert event["data"]["result"] == [{"error": {"code": "pull_failed"}}]


@pytest.mark.asyncio
async def test_react_round_preserves_named_service_result_scalar_leaves(
    monkeypatch, tmp_path
):
    async def _fake_tool(*, react, ctx_browser, state, tool_call_id):
        del react, ctx_browser, tool_call_id
        state["last_tool_result"] = [
            {
                "artifact_id": "named_services.list_objects",
                "output": {
                    "items": [
                        {
                            "document_id": "doc-1",
                            "title": "26_006",
                            "owned_by_me": True,
                            "ref": "docs:google:account-1:document:doc-1",
                        }
                    ]
                },
            }
        ]
        return state

    monkeypatch.setattr(round_mod.react_tools, "handle_external_tool", _fake_tool)

    runtime = RuntimeCtx(turn_id="turn_tool", outdir=str(tmp_path), workdir=str(tmp_path))
    ctx = FakeBrowser(runtime)
    comm = _FakeComm()
    react = SimpleNamespace(ctx_browser=ctx, comm=comm)
    state = {
        "last_decision": {
            "tool_call": {
                "tool_id": "named_services.list_objects",
                "tool_call_id": "tc_docs",
                "params": {"namespace": "docs"},
            },
        },
    }

    await ReactRound.execute(react, state)

    item = comm.events[-1]["data"]["result"][0]["output"]["items"][0]
    assert item == {
        "document_id": "doc-1",
        "title": "26_006",
        "owned_by_me": True,
        "ref": "docs:google:account-1:document:doc-1",
    }
    assert comm.events[-1]["data"]["result_truncated"] is False


@pytest.mark.asyncio
async def test_react_round_preserves_nested_named_service_schema(
    monkeypatch, tmp_path
):
    async def _fake_tool(*, react, ctx_browser, state, tool_call_id):
        del react, ctx_browser, tool_call_id
        state["last_tool_result"] = [
            {
                "artifact_id": "named_services.object_schema",
                "output": {
                    "extra": {
                        "schema": {
                            "search": {
                                "filters": {
                                    "account_id": {
                                        "type": "string",
                                        "description": "Connected account id.",
                                    }
                                }
                            },
                            "actions": {
                                "copy": {
                                    "description": "Copy a document.",
                                    "payload": ["title"],
                                }
                            },
                        }
                    }
                },
            }
        ]
        return state

    monkeypatch.setattr(round_mod.react_tools, "handle_external_tool", _fake_tool)

    runtime = RuntimeCtx(turn_id="turn_tool", outdir=str(tmp_path), workdir=str(tmp_path))
    ctx = FakeBrowser(runtime)
    comm = _FakeComm()
    react = SimpleNamespace(ctx_browser=ctx, comm=comm)
    await ReactRound.execute(
        react,
        {
            "last_decision": {
                "tool_call": {
                    "tool_id": "named_services.object_schema",
                    "tool_call_id": "tc_schema",
                    "params": {"namespace": "docs"},
                },
            },
        },
    )

    schema = comm.events[-1]["data"]["result"][0]["output"]["extra"]["schema"]
    assert schema["search"]["filters"]["account_id"]["type"] == "string"
    assert schema["actions"]["copy"]["payload"] == ["title"]


@pytest.mark.asyncio
async def test_react_round_uses_distinct_steps_for_multiple_tool_calls(monkeypatch, tmp_path):
    async def _fake_rg(*, ctx_browser, state, tool_call_id):
        state["last_tool_result"] = [{"ok": True, "call": tool_call_id}]
        return state

    monkeypatch.setattr(round_mod.react_tools, "handle_react_rg", _fake_rg)

    runtime = RuntimeCtx(turn_id="turn_tool", outdir=str(tmp_path), workdir=str(tmp_path))
    ctx = FakeBrowser(runtime)
    comm = _FakeComm()
    react = SimpleNamespace(ctx_browser=ctx, comm=comm)

    for call_id in ("tc_a", "tc_b"):
        await ReactRound.execute(react, {
            "last_decision": {
                "tool_call": {
                    "tool_id": "react.rg",
                    "tool_call_id": call_id,
                    "params": {"query": "x"},
                },
            },
        })

    assert [event["step"] for event in comm.events] == [
        "react.tool.call.tc_a",
        "react.tool.result.tc_a",
        "react.tool.call.tc_b",
        "react.tool.result.tc_b",
    ]
    assert len({event["step"] for event in comm.events}) == 4


@pytest.mark.asyncio
async def test_react_round_emits_rejected_event_for_missing_tool_id(tmp_path):
    runtime = RuntimeCtx(turn_id="turn_tool", outdir=str(tmp_path), workdir=str(tmp_path))
    ctx = FakeBrowser(runtime)
    comm = _FakeComm()
    react = SimpleNamespace(ctx_browser=ctx, comm=comm)

    result = await ReactRound.execute(react, {
        "pending_tool_call_id": "tc_missing",
        "last_decision": {
            "tool_call": {
                "params": {"query": "x"},
            },
        },
    })

    assert result["exit_reason"] == "error"
    assert len(comm.events) == 1
    event = comm.events[0]
    assert event["type"] == "react.tool.rejected"
    assert event["step"] == "react.tool.rejected.tc_missing"
    assert event["status"] == "error"
    assert event["data"]["executed"] is False
    assert event["data"]["error"]["code"] == "missing_tool_id"
