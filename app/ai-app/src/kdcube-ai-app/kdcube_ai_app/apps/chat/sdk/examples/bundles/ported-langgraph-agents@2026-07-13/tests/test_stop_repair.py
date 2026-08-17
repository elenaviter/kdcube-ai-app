# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
"""Leaving a stopped thread in a state the next turn can actually send.

LIVE, on lg-react: a turn was stopped mid-run, and the turn after it failed with

    400 'tool_use' ids were found without 'tool_result' blocks immediately
    after: toolu_01Pnxw65v5FGH5zBByyNca1x

and so did every turn after that — the same history replays each time, so one
stop cost the conversation rather than the turn. The cancellation landed between
the model asking for a tool and the result arriving, and the checkpoint kept the
half it had.
"""
from __future__ import annotations

import asyncio
import importlib.util
import pathlib

import pytest

_MODULE_PATH = (
    pathlib.Path(__file__).resolve().parent.parent / "platform" / "stop_repair.py"
)
_spec = importlib.util.spec_from_file_location("lg_stop_repair", _MODULE_PATH)
stop_repair = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stop_repair)


class _AI:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls


class _Tool:
    tool_calls = None

    def __init__(self, tool_call_id):
        self.tool_call_id = tool_call_id


class _Human:
    tool_calls = None
    tool_call_id = None


class _Snapshot:
    def __init__(self, messages):
        self.values = {"messages": messages}


class _Graph:
    """A graph whose state can be read and appended to, like a checkpointed one."""

    nodes = {"model": object(), "tools": object()}

    def __init__(self, messages):
        self.messages = list(messages)
        self.updates = []
        self.as_nodes = []

    async def aget_state(self, _config):
        return _Snapshot(self.messages)

    async def aupdate_state(self, _config, values, as_node=None):
        self.as_nodes.append(as_node)
        self.updates.append(values)
        self.messages.extend(values["messages"])


def test_a_call_the_stop_interrupted_is_answered():
    graph = _Graph([
        _Human(),
        _AI([{"id": "toolu_01Pnxw", "name": "web_search", "args": {}}]),
    ])

    repaired = asyncio.run(stop_repair.repair_unanswered_tool_calls(graph, {}))

    assert repaired == 1
    added = graph.updates[0]["messages"]
    assert added[0].tool_call_id == "toolu_01Pnxw"
    # Truthful: the model reads that the tool did not run, rather than a result
    # it never produced.
    assert "not run" in added[0].content
    assert stop_repair.unanswered_tool_calls(graph.messages) == []


def test_a_finished_turn_is_left_alone():
    """The repair must be inert on a healthy thread — it runs before EVERY turn."""
    graph = _Graph([
        _Human(),
        _AI([{"id": "toolu_a", "name": "web_search", "args": {}}]),
        _Tool("toolu_a"),
    ])

    assert asyncio.run(stop_repair.repair_unanswered_tool_calls(graph, {})) == 0
    assert graph.updates == []


def test_every_unanswered_call_of_a_parallel_batch_is_answered():
    """One tool_use without its result is enough to be refused, so a batch where
    only some results arrived has to be completed, not skipped."""
    graph = _Graph([
        _Human(),
        _AI([
            {"id": "toolu_a", "name": "web_search", "args": {}},
            {"id": "toolu_b", "name": "run_python", "args": {}},
        ]),
        _Tool("toolu_a"),
    ])

    assert asyncio.run(stop_repair.repair_unanswered_tool_calls(graph, {})) == 1
    assert graph.updates[0]["messages"][0].tool_call_id == "toolu_b"


def test_a_state_that_cannot_be_read_does_not_fail_the_turn():
    class _Broken:
        async def aget_state(self, _config):
            raise RuntimeError("no checkpointer")

    assert asyncio.run(stop_repair.repair_unanswered_tool_calls(_Broken(), {})) == 0


def test_a_repair_that_cannot_be_written_is_reported_not_swallowed(caplog):
    class _ReadOnly(_Graph):
        async def aupdate_state(self, _config, values, as_node=None):
            raise RuntimeError("read-only checkpoint")

    graph = _ReadOnly([_AI([{"id": "toolu_a", "name": "web_search", "args": {}}])])
    with caplog.at_level("WARNING"):
        assert asyncio.run(stop_repair.repair_unanswered_tool_calls(graph, {})) == 0
    assert any("dangling tool call" in record.getMessage() for record in caplog.records)


def test_the_repair_is_written_as_the_tools_node(monkeypatch):
    """LIVE, and the reason the first repair did not land: an unattributed
    update makes LangGraph resume from the interrupted node and re-evaluate its
    conditional edge, which on the prebuilt agent with middleware raises

        KeyError: 'SummarizationMiddleware.before_model'

    A tool result is the tools node's output; saying so follows that node's
    ordinary edge instead of replaying a branch from a run that never finished.
    """
    graph = _Graph([_AI([{"id": "toolu_a", "name": "web_search", "args": {}}])])

    assert asyncio.run(stop_repair.repair_unanswered_tool_calls(graph, {})) == 1
    assert graph.as_nodes == ["tools"]


def test_a_graph_without_a_tools_node_still_writes():
    class _NoTools(_Graph):
        nodes = {"model": object()}

    graph = _NoTools([_AI([{"id": "toolu_a", "name": "web_search", "args": {}}])])
    assert asyncio.run(stop_repair.repair_unanswered_tool_calls(graph, {})) == 1
    assert graph.as_nodes == [None]


def test_a_rejected_attribution_falls_back_rather_than_giving_up():
    """The node names are a guess about somebody else's graph; a wrong guess
    must cost an attempt, not the repair."""
    class _RejectsNode(_Graph):
        async def aupdate_state(self, _config, values, as_node=None):
            if as_node:
                raise ValueError(f"unknown node {as_node}")
            return await _Graph.aupdate_state(self, _config, values)

    graph = _RejectsNode([_AI([{"id": "toolu_a", "name": "web_search", "args": {}}])])
    assert asyncio.run(stop_repair.repair_unanswered_tool_calls(graph, {})) == 1
