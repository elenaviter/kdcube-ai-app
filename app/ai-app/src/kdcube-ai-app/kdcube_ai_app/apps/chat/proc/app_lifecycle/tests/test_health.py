# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi.responses import JSONResponse

from kdcube_ai_app.apps.chat.proc import web_app
from kdcube_ai_app.infra.plugin.app_readiness import (
    ApplicationLifecycleState,
    ApplicationReadinessMode,
    DesiredApplicationState,
    application_readiness_registry,
)


@pytest.fixture
def _health_scope(monkeypatch: pytest.MonkeyPatch):
    tenant = "health-tenant"
    project = "health-project"
    monkeypatch.setattr(
        web_app,
        "get_settings",
        lambda: SimpleNamespace(TENANT=tenant, PROJECT=project),
    )
    web_app.app.state.draining = False
    yield tenant, project
    application_readiness_registry.deactivate_scope(
        tenant=tenant,
        project=project,
        clear=True,
    )


@pytest.mark.asyncio
async def test_independent_application_does_not_block_proc_readiness(_health_scope) -> None:
    tenant, project = _health_scope
    application_readiness_registry.replace_desired(
        tenant=tenant,
        project=project,
        applications={
            "slow@1-0": DesiredApplicationState(
                generation="generation-a",
                readiness=ApplicationReadinessMode.INDEPENDENT,
            )
        },
    )

    response = await web_app.health()

    assert isinstance(response, dict)
    assert response["status"] == "ok"
    assert response["applications_ready"] is True
    assert response["applications"]["slow@1-0"]["state"] == "pending"


@pytest.mark.asyncio
async def test_required_application_blocks_readiness_but_not_liveness(_health_scope) -> None:
    tenant, project = _health_scope
    application_readiness_registry.replace_desired(
        tenant=tenant,
        project=project,
        applications={
            "required@1-0": DesiredApplicationState(
                generation="generation-a",
                readiness=ApplicationReadinessMode.REQUIRED,
            )
        },
    )

    readiness = await web_app.health()
    liveness = await web_app.health_live()

    assert isinstance(readiness, JSONResponse)
    assert readiness.status_code == 503
    payload = json.loads(readiness.body)
    assert payload["blocking_applications"] == ["required@1-0"]
    assert isinstance(liveness, dict)
    assert liveness["status"] == "ok"

    assert application_readiness_registry.transition(
        tenant=tenant,
        project=project,
        application_id="required@1-0",
        generation="generation-a",
        state=ApplicationLifecycleState.READY,
        attempt=1,
    )
    ready = await web_app.health()
    assert isinstance(ready, dict)
    assert ready["status"] == "ok"
