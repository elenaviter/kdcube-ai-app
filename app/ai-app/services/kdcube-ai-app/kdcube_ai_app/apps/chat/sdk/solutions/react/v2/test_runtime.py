# SPDX-License-Identifier: MIT

import pytest

from kdcube_ai_app.apps.chat.sdk.solutions.react.v2.runtime import ReactSolverV2


class _LogStub:
    def log(self, *args, **kwargs):
        return None


def _solver_stub() -> ReactSolverV2:
    solver = ReactSolverV2.__new__(ReactSolverV2)
    solver.log = _LogStub()
    return solver


def test_route_after_decision_exits_when_exit_reason_is_set():
    solver = _solver_stub()
    state = {
        "exit_reason": "max_iterations",
        "last_decision": {"action": "call_tool"},
    }

    assert solver._route_after_decision(state) == "exit"


@pytest.mark.asyncio
async def test_decision_node_short_circuits_when_exit_reason_is_set():
    solver = _solver_stub()
    called = {"value": False}

    async def _impl(state, iteration):
        called["value"] = True
        return state

    solver._decision_node_impl = _impl
    state = {
        "exit_reason": "error",
        "iteration": 0,
        "max_iterations": 5,
    }

    out = await solver._decision_node(state)

    assert out is state
    assert called["value"] is False
