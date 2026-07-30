from __future__ import annotations

import importlib
import importlib.util
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from kdcube_ai_app.apps.chat.sdk.protocol import (
    ExternalEventActor,
    ExternalEventPayload,
    ExternalEventRequest,
    ExternalEventRouting,
    ExternalEventUser,
)
from kdcube_ai_app.apps.chat.sdk.runtime import local_sidecars


def _bundle_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_package(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name,
        path / "__init__.py",
        submodule_search_locations=[str(path)],
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_entrypoint_module():
    root = _bundle_root()
    package_name = "node_bridge_bundle_testpkg"
    _ensure_package(package_name, root)
    module_name = f"{package_name}.entrypoint"
    spec = importlib.util.spec_from_file_location(
        module_name,
        root / "entrypoint.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _ctx() -> ExternalEventPayload:
    return ExternalEventPayload(
        request=ExternalEventRequest(request_id="req-node-bridge"),
        routing=ExternalEventRouting(
            session_id="sid-node-bridge",
            conversation_id="conv-node-bridge",
            turn_id="turn-node-bridge",
            bundle_id="node.bridge.mcp",
        ),
        actor=ExternalEventActor(
            tenant_id="demo-tenant",
            project_id="demo-project",
        ),
        user=ExternalEventUser(
            user_type="registered",
            user_id="user-1",
            username="user@example.com",
            roles=[],
            permissions=[],
            timezone="UTC",
        ),
    )


class _DummyConfig:
    def __init__(self, bundle_id: str):
        self.ai_bundle_spec = SimpleNamespace(id=bundle_id)
        self.log_level = "INFO"
        self.role_models = {}
        self.embedder_config = {}
        self.embedding_model = "text-embedding-3-small"
        self.custom_embedding_endpoint = None
        self.custom_embedding_size = None
        self.openai_api_key = "test-openai-key"
        self.claude_api_key = None
        self.google_api_key = None
        self.custom_model_endpoint = None
        self.custom_model_api_key = None
        self.gemini_cache_enabled = False
        self.gemini_cache_ttl_seconds = None

    def set_role_models(self, models):
        self.role_models = dict(models or {})

    def set_embedding(self, _embedding):
        return None

    def ensure_role(self, _role):
        return {"provider": "openai", "model": "gpt-4o-mini"}

    def __getattr__(self, _name):
        return None


def test_build_node_bridge_mcp_app_returns_streamable_http_app():
    pytest.importorskip("mcp.server.mcpserver")
    mod = _load_entrypoint_module()
    app = mod.node_mcp_tools.build_node_bridge_mcp_app(
        name="node.bridge.test",
        bridge_provider=lambda: None,
    )
    assert app is not None
    assert hasattr(app, "streamable_http_app")


@pytest.mark.asyncio
async def test_node_bridge_bundle_api_smoke(monkeypatch, tmp_path: Path):
    if shutil.which("node") is None:
        pytest.skip("node is not installed")

    entrypoint_mod = importlib.import_module(
        "kdcube_ai_app.apps.chat.sdk.solutions.chatbot.entrypoint"
    )
    monkeypatch.setattr(
        entrypoint_mod,
        "get_settings",
        lambda: SimpleNamespace(TENANT="demo-tenant", PROJECT="demo-project"),
    )
    monkeypatch.setattr(entrypoint_mod, "create_kv_cache_from_env", lambda: None)

    mod = _load_entrypoint_module()
    bundle = mod.NodeBridgeBundle(
        config=_DummyConfig(bundle_id="node.bridge.mcp"),
        comm_context=_ctx(),
    )
    monkeypatch.setattr(bundle, "_bundle_root", lambda: str(_bundle_root()))
    monkeypatch.setattr(bundle, "bundle_storage_root", lambda: tmp_path)

    try:
        status_payload = await bundle.node_status()
        assert status_payload["ok"] is True
        assert status_payload["status"] == 200
        assert status_payload["data"]["runtime"] == "node-sidecar"

        search_payload = await bundle.node_search(query="alpha")
        assert search_payload["ok"] is True
        assert search_payload["data"]["total"] == 1
        assert search_payload["data"]["items"][0]["title"] == "Node match for alpha"
    finally:
        bundle.stop_local_sidecar("node-backend")
        await local_sidecars.shutdown_all_local_sidecars(
            terminate_timeout_sec=1.0,
            kill_timeout_sec=0.5,
        )


@pytest.mark.asyncio
async def test_node_bridge_live_reconfigure_and_startup_restart(monkeypatch, tmp_path: Path):
    if shutil.which("node") is None:
        pytest.skip("node is not installed")

    entrypoint_mod = importlib.import_module(
        "kdcube_ai_app.apps.chat.sdk.solutions.chatbot.entrypoint"
    )
    monkeypatch.setattr(
        entrypoint_mod,
        "get_settings",
        lambda: SimpleNamespace(TENANT="demo-tenant", PROJECT="demo-project"),
    )
    monkeypatch.setattr(entrypoint_mod, "create_kv_cache_from_env", lambda: None)

    mod = _load_entrypoint_module()
    bundle = mod.NodeBridgeBundle(
        config=_DummyConfig(bundle_id="node.bridge.mcp"),
        comm_context=_ctx(),
    )
    monkeypatch.setattr(bundle, "_bundle_root", lambda: str(_bundle_root()))
    monkeypatch.setattr(bundle, "bundle_storage_root", lambda: tmp_path)

    try:
        initial_status = await bundle.node_status()
        assert initial_status["data"]["runtime"] == "node-sidecar"
        handle_one = bundle.get_local_sidecar("node-backend")
        assert handle_one is not None

        bundle.bundle_props = {
            "node_bridge": {
                "runtime_config": {
                    "statusLabel": "node-sidecar-live",
                    "searchPrefix": "Live match for",
                }
            }
        }
        live_search = await bundle.node_search(query="beta")
        handle_two = bundle.get_local_sidecar("node-backend")
        assert handle_two is not None
        assert handle_two.pid == handle_one.pid
        assert live_search["data"]["items"][0]["title"] == "Live match for beta"

        live_status = await bundle.node_status()
        assert live_status["data"]["runtime"] == "node-sidecar-live"

        bundle.bundle_props = {
            "node_bridge": {
                "allowed_prefixes": ["/api/projects", "/api/extra"],
                "runtime_config": {
                    "statusLabel": "node-sidecar-live",
                    "searchPrefix": "Live match for",
                },
            }
        }
        restarted_status = await bundle.node_status()
        handle_three = bundle.get_local_sidecar("node-backend")
        assert handle_three is not None
        assert handle_three.pid != handle_two.pid
        assert restarted_status["ok"] is True
    finally:
        bundle.stop_local_sidecar("node-backend")
        await local_sidecars.shutdown_all_local_sidecars(
            terminate_timeout_sec=1.0,
            kill_timeout_sec=0.5,
        )
