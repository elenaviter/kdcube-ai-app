from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import yaml


REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "AGENTS.md").is_file() and (parent / "agents").is_dir()
)
AGENTS_ROOT = REPO_ROOT / "agents"
ROOT_README = REPO_ROOT / "README.md"
QUICK_START = REPO_ROOT / "app" / "ai-app" / "docs" / "quick-start-README.md"
MCP_CATALOG = REPO_ROOT / "mcp" / "README.md"
WEB_SEARCH_QUICK_START = REPO_ROOT / "mcp" / "web-search" / "README.md"
DIRECT_RECIPE = (
    REPO_ROOT
    / "app"
    / "ai-app"
    / "docs"
    / "recipes"
    / "quickstart"
    / "run-agent-harness-from-python-README.md"
)
ADAPTERS = ("native", "langgraph", "claude")
WEB_SEARCH_TOOL_IDS = {
    "native": "web_tools.web_search",
    "langgraph": "web_search",
    "claude": "mcp__kdcube_web_search__web_search",
}
EXEC_TOOL_IDS = {
    "native": "exec_tools.execute_code_python",
    "langgraph": "execute_python",
    "claude": "mcp__kdcube_harness__execute_python",
}
RENDER_TOOL_IDS = {
    "native": {
        "rendering_tools.write_pdf",
        "rendering_tools.write_docx",
        "rendering_tools.write_pptx",
    },
    "langgraph": {"write_pdf", "write_docx", "write_pptx"},
    "claude": {
        "mcp__kdcube_harness__write_pdf",
        "mcp__kdcube_harness__write_docx",
        "mcp__kdcube_harness__write_pptx",
    },
}
README_SECTIONS = (
    "## What it is",
    "## Run it",
    "## What the demo shows",
    "## Change the demo",
)


def test_primary_agents_are_visible_at_the_agents_root() -> None:
    top_level_directories = {
        path.name
        for path in AGENTS_ROOT.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    }
    assert top_level_directories == set(ADAPTERS)


def test_primary_onboarding_links_agents_and_web_search_directly() -> None:
    root = ROOT_README.read_text(encoding="utf-8")
    quick_start = QUICK_START.read_text(encoding="utf-8")
    agents = (AGENTS_ROOT / "README.md").read_text(encoding="utf-8")
    mcp_catalog = MCP_CATALOG.read_text(encoding="utf-8")

    assert "(agents/README.md)" in root
    assert "(app/ai-app/docs/recipes/quickstart/run-agent-harness-from-python-README.md)" in root
    assert "(mcp/web-search/README.md)" in root
    assert "(../../../agents/README.md)" in quick_start
    assert "(recipes/quickstart/run-agent-harness-from-python-README.md)" in quick_start
    assert "(../../../mcp/web-search/README.md)" in quick_start
    assert "(../mcp/web-search/README.md)" in agents
    assert "(web-search/README.md)" in mcp_catalog
    assert DIRECT_RECIPE.is_file()
    assert WEB_SEARCH_QUICK_START.is_file()


def test_every_agents_readme_answers_the_four_first_use_questions() -> None:
    readmes = sorted(AGENTS_ROOT / adapter / "README.md" for adapter in ADAPTERS)
    assert readmes
    for path in readmes:
        source = path.read_text(encoding="utf-8")
        positions = [source.find(heading) for heading in README_SECTIONS]
        assert all(position >= 0 for position in positions), path
        assert positions == sorted(positions), path


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_each_example_owns_its_runnable_contract(adapter: str) -> None:
    root = AGENTS_ROOT / adapter
    expected = {
        "README.md",
        "agent.py",
        "compose.yaml",
        "config.template.yaml",
        "setup_local.py",
        "descriptors.template",
        "requirements.txt",
        "skills",
    }
    assert expected.issubset({path.name for path in root.iterdir()})
    assert not (root / "web-search.yaml").exists()
    if adapter == "native":
        assert not (root / "tool_plan.py").exists()
        assert not (root / "configuration.py").exists()

    config = yaml.safe_load((root / "config.template.yaml").read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    assert "output" not in config
    assert "web_search" not in config
    assert "infra" not in config
    agent = config["agent"]
    assert agent["instructions"]["profile"] == (
        "lite:core" if adapter == "native" else "workspace-files"
    )
    assert str(agent.get("additional_instructions") or "").strip()
    assert agent["run_directory"] == "./output"
    assert isinstance(agent.get("tools"), list)
    assert agent["tools"]
    assert all(str(item.get("id") or "").strip() for item in agent["tools"])
    configured_by_id = {item["id"]: item for item in agent["tools"]}
    assert configured_by_id[EXEC_TOOL_IDS[adapter]]["runtime"] == "docker"
    assert RENDER_TOOL_IDS[adapter].issubset(configured_by_id)
    assert all(
        configured_by_id[tool_id]["runtime"] == "local"
        for tool_id in RENDER_TOOL_IDS[adapter]
    )
    assert agent.get("skills", {}).get("enabled") == [
        "demo.research-brief"
    ]
    web_rows = [
        item for item in agent["tools"] if item["id"] == WEB_SEARCH_TOOL_IDS[adapter]
    ]
    assert len(web_rows) == 1
    web_policy = web_rows[0]["settings"]
    assert web_policy["filter"]["allowlist"] == ["python.org"]
    assert web_policy["filter"]["blocklist"] == []
    assert web_policy["filter"]["ssrf_guard"] is True
    if adapter == "claude":
        assert agent["adapter"]["model"] == "claude-haiku-4-5-20251001"
    else:
        assert "adapter" not in agent

    requirements = (root / "requirements.txt").read_text(encoding="utf-8")
    assert "../../app/ai-app/src/kdcube-ai-app" in requirements
    assert "sdk/tools/mcp/web_search/requirements.txt" in requirements
    assert "playwright" in requirements
    assert "python-docx" in requirements
    assert "python-pptx" in requirements


def test_every_adapter_uses_kdcube_web_search() -> None:
    native_tools = (AGENTS_ROOT / "native" / "tools.py").read_text(encoding="utf-8")
    langgraph_tools = (AGENTS_ROOT / "langgraph" / "tools.py").read_text(
        encoding="utf-8"
    )
    claude_config = yaml.safe_load(
        (AGENTS_ROOT / "claude" / "config.template.yaml").read_text(encoding="utf-8")
    )
    claude_source = (AGENTS_ROOT / "claude" / "agent.py").read_text(encoding="utf-8")

    for source in (native_tools, langgraph_tools):
        assert "from ddgs import DDGS" not in source
        assert "web_search_server.web_search(" in source

    claude_tools = {
        item["id"]
        for item in claude_config["agent"]["tools"]
        if item.get("enabled", True)
    }
    assert "WebSearch" not in claude_tools
    assert "WebFetch" not in claude_tools
    assert "Bash" not in claude_tools
    assert "mcp__kdcube_web_search__web_search" in claude_tools
    assert "mcp__kdcube_web_search__web_fetch" not in claude_tools
    assert "mcp__kdcube_web_search__allowlist_status" not in claude_tools
    assert "kdcube_ai_app.apps.chat.sdk.tools.mcp.web_search.web_search_server" in claude_source
    assert 'denied_tools=("WebSearch", "WebFetch", "Bash")' in claude_source


@pytest.mark.asyncio
async def test_native_web_search_adapter_calls_kdcube_tool(monkeypatch) -> None:
    from agents.native import tools as native_tools

    search = AsyncMock(
        return_value=[
            {"title": "Python", "text": "Current release", "url": "https://python.org/"}
        ]
    )
    monkeypatch.setattr(native_tools.web_search_server, "web_search", search)

    result = await native_tools.web_search("current Python release", max_results=3)

    assert result["ok"] is True
    assert result["results"][0]["url"] == "https://python.org/"
    search.assert_awaited_once_with(
        queries="current Python release",
        objective="current Python release",
        refinement="none",
        n=3,
        fetch_content=False,
        include_binary_base64=False,
        use_llm=False,
    )


@pytest.mark.asyncio
async def test_langgraph_web_search_adapter_calls_kdcube_tool(
    monkeypatch,
) -> None:
    from agents.langgraph import tools as langgraph_tools

    search = AsyncMock(
        return_value=[
            {"title": "Python", "text": "Current release", "url": "https://python.org/"}
        ]
    )
    monkeypatch.setattr(langgraph_tools.web_search_server, "web_search", search)
    tools = langgraph_tools.build_tools(None, enabled_ids={"web_search"})

    result = json.loads(
        await tools[0].ainvoke({"query": "current Python release", "max_results": 2})
    )

    assert result["ok"] is True
    assert result["results"][0]["url"] == "https://python.org/"
    search.assert_awaited_once_with(
        queries="current Python release",
        objective="current Python release",
        refinement="none",
        n=2,
        fetch_content=False,
        include_binary_base64=False,
        use_llm=False,
    )


@pytest.mark.asyncio
async def test_claude_web_search_mcp_starts_with_operator_policy() -> None:
    from kdcube_ai_app.apps.chat.sdk.runtime.mcp.client import (
        normalize_mcp_tool_result,
        open_mcp_client,
    )

    module = "kdcube_ai_app.apps.chat.sdk.tools.mcp.web_search.web_search_server"
    config = (AGENTS_ROOT / "claude" / "config.template.yaml").resolve()
    async with open_mcp_client(
        transport="stdio",
        command=sys.executable,
        args=[
            "-m",
            module,
            "--transport",
            "stdio",
            "--config",
            str(config),
            "--tool-id",
            "mcp__kdcube_web_search__web_search",
        ],
        env=os.environ.copy(),
        read_timeout_seconds=20,
    ) as client:
        listed = await client.list_tools()
        status = normalize_mcp_tool_result(
            await client.call_tool("allowlist_status", {})
        )["result"]

    assert {tool.name for tool in listed.tools} == {
        "allowlist_status",
        "web_fetch",
        "web_search",
    }
    assert status["allowlist_entries"] == ["python.org"]
    assert status["ssrf_guard"] is True
    assert status["enforced"] is True


@pytest.mark.asyncio
async def test_claude_harness_mcp_exposes_execution_and_rendering_tools(
    tmp_path: Path,
) -> None:
    from kdcube_ai_app.apps.chat.sdk.runtime.mcp.client import open_mcp_client

    module = "kdcube_ai_app.apps.chat.sdk.runtime.direct_hosting.tool_server"
    descriptors = (AGENTS_ROOT / "claude" / "descriptors.template").resolve()
    async with open_mcp_client(
        transport="stdio",
        command=sys.executable,
        args=[
            "-m",
            module,
            "--descriptors",
            str(descriptors),
            "--run-root",
            str(tmp_path / "run"),
            "--events",
            str(tmp_path / "mcp-events.jsonl"),
            "--conversation-id",
            "conversation-demo",
            "--turn-id",
            "turn_demo",
            "--bundle-id",
            "standalone-claude-demo@1-0",
            "--agent-id",
            "claude",
            "--bundle-root",
            str(AGENTS_ROOT / "claude"),
            "--bundle-module",
            "agent",
        ],
        env=os.environ.copy(),
        read_timeout_seconds=20,
    ) as client:
        listed = await client.list_tools()

    assert {tool.name for tool in listed.tools} == {
        "execute_python",
        "write_pdf",
        "write_docx",
        "write_pptx",
    }


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_each_example_owns_a_standard_app_agnostic_descriptor_set(adapter: str) -> None:
    descriptors = AGENTS_ROOT / adapter / "descriptors.template"
    assert {path.name for path in descriptors.iterdir()} == {
        "assembly.yaml",
        "economics.yaml",
        "gateway.yaml",
        "secrets.yaml",
    }
    assembly = yaml.safe_load((descriptors / "assembly.yaml").read_text(encoding="utf-8"))
    secrets = yaml.safe_load((descriptors / "secrets.yaml").read_text(encoding="utf-8"))
    economics = yaml.safe_load((descriptors / "economics.yaml").read_text(encoding="utf-8"))
    assert isinstance(assembly.get("infra", {}).get("redis"), dict)
    assert isinstance(assembly.get("infra", {}).get("postgres"), dict)
    assert assembly.get("storage", {}).get("kdcube") == "../output/kdcube-storage"
    assert assembly["models"]["default_llm_model_id"] == "claude-haiku-4-5-20251001"
    assert (
        assembly["platform"]["services"]["proc"]["exec"]["py_code_exec_image"]
        == "py-code-exec:latest"
    )
    assert assembly.get("secrets", {}).get("provider") == "secrets-file"
    assert secrets["infra"]["redis"]["password"] is None
    assert secrets["infra"]["postgres"]["password"] is None
    if adapter == "claude":
        assert assembly["storage"]["claude_code_session"]["type"] == "git"
        assert str(assembly["storage"]["claude_code_session"]["repo"] or "")
        assert "http_token" in secrets["services"]["git"]
    else:
        assert "claude_code_session" not in assembly["storage"]
        assert "git" not in secrets["services"]
    prices = economics.get("price_tables", {}).get("llm")
    assert prices
    assert prices[0]["model"] == "claude-haiku-4-5-20251001"
    assert prices[0]["provider"] == "anthropic"
    assert yaml.safe_load((descriptors / "gateway.yaml").read_text(encoding="utf-8")) == {}


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_primary_examples_do_not_use_hosted_runtime_client(adapter: str) -> None:
    source = (AGENTS_ROOT / adapter / "agent.py").read_text(encoding="utf-8")
    forbidden = (
        "/sse/chat",
        "RuntimeChatClient",
        "--workdir",
        "KDCUBE_RUNTIME_WORKDIR",
        "agents.integration",
    )
    assert not [marker for marker in forbidden if marker in source]


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_primary_examples_use_sdk_direct_harness(adapter: str) -> None:
    source = (AGENTS_ROOT / adapter / "agent.py").read_text(encoding="utf-8")
    assert "DirectAgentHarness" in source
    assert "harness.turn(" in source
    assert "verify_conversation(" in source


def test_claude_uses_sdk_git_session_store_for_its_transcript() -> None:
    source = (AGENTS_ROOT / "claude" / "agent.py").read_text(encoding="utf-8")
    assembly = yaml.safe_load(
        (AGENTS_ROOT / "claude" / "descriptors.template" / "assembly.yaml").read_text(
            encoding="utf-8"
        )
    )
    secrets = yaml.safe_load(
        (AGENTS_ROOT / "claude" / "descriptors.template" / "secrets.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert "run_claude_code_turn(" in source
    assert "ClaudeCodeSessionStoreConfig(" in source
    assert assembly["storage"]["claude_code_session"]["type"] == "git"
    assert str(assembly["storage"]["claude_code_session"]["repo"] or "")
    assert "http_token" in secrets["services"]["git"]


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_offline_adapter_construction(adapter: str) -> None:
    root = AGENTS_ROOT / adapter
    env = os.environ.copy()
    env.pop("KDCUBE_RUNTIME_WORKDIR", None)
    completed = subprocess.run(
        [
            sys.executable,
            "agent.py",
            "--config",
            "config.template.yaml",
            "--descriptors",
            "descriptors.template",
            "--check",
        ],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "mode: standalone SDK process" in completed.stdout
    assert "tools:" in completed.stdout
    assert "skills: demo.research-brief" in completed.stdout
    assert "instruction profile:" in completed.stdout
    assert "custom instructions: configured" in completed.stdout
    assert "model: anthropic/claude-haiku-4-5-20251001" in completed.stdout
    assert "check: PASS" in completed.stdout


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_offline_adapter_construction_rejects_unknown_instruction_profile(
    adapter: str,
    tmp_path: Path,
) -> None:
    root = AGENTS_ROOT / adapter
    config = yaml.safe_load((root / "config.template.yaml").read_text(encoding="utf-8"))
    config["agent"]["instructions"]["profile"] = "unknown-profile"
    config["agent"]["run_directory"] = str(tmp_path / "output")
    config["agent"]["skills"]["root"] = str(root / "skills")
    config_path = tmp_path / f"{adapter}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "agent.py",
            "--config",
            str(config_path),
            "--descriptors",
            "descriptors.template",
            "--check",
        ],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode != 0
    assert "unknown" in completed.stderr.lower()
    assert "instruction profile" in completed.stderr.lower()


def test_native_model_selection_comes_from_assembly_descriptor(tmp_path: Path) -> None:
    descriptors = tmp_path / "descriptors"
    shutil.copytree(AGENTS_ROOT / "native" / "descriptors.template", descriptors)
    assembly_path = descriptors / "assembly.yaml"
    assembly = yaml.safe_load(assembly_path.read_text(encoding="utf-8"))
    assembly["models"]["default_llm_model_id"] = "gpt-4o"
    assembly_path.write_text(yaml.safe_dump(assembly, sort_keys=False), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "agent.py",
            "--descriptors",
            str(descriptors),
            "--check",
        ],
        cwd=AGENTS_ROOT / "native",
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "model: openai/gpt-4o" in completed.stdout


def test_each_example_owns_its_selected_research_skill() -> None:
    for adapter in ADAPTERS:
        skill = (
            AGENTS_ROOT
            / adapter
            / "skills"
            / "demo"
            / "research-brief"
            / "SKILL.md"
        )
        source = skill.read_text(encoding="utf-8")
        assert "namespace: demo" in source
        assert "name: research-brief" in source
        assert "id: research-brief" in source
        config = yaml.safe_load(
            (AGENTS_ROOT / adapter / "config.template.yaml").read_text(encoding="utf-8")
        )
        assert config["agent"]["skills"]["root"] == "./skills"


def test_native_yaml_can_select_the_isolated_exec_tool() -> None:
    from agents.native.agent import EXEC_TOOL_ID, RENDER_TOOL_IDS, TOOL_SOURCES
    from kdcube_ai_app.apps.chat.sdk.runtime.direct_hosting.native_tool_bindings import (
        resolve_native_tool_bindings,
    )

    config = yaml.safe_load(
        (AGENTS_ROOT / "native" / "config.template.yaml").read_text(encoding="utf-8")
    )
    bindings = resolve_native_tool_bindings(
        config,
        sources=TOOL_SOURCES,
        adapter_name="native direct example",
    )

    assert bindings.enabled_ids == (
        "web_tools.web_search",
        EXEC_TOOL_ID,
        *RENDER_TOOL_IDS,
    )
    assert bindings.tool_runtime[EXEC_TOOL_ID] == "docker"
    assert bindings.allowed_tool_names_by_alias["exec_tools"] == [
        "execute_code_python"
    ]
    assert any(spec.get("alias") == "exec_tools" for spec in bindings.tool_specs)
    assert bindings.allowed_tool_names_by_alias["rendering_tools"] == [
        "write_pdf",
        "write_docx",
        "write_pptx",
    ]


def test_native_tool_binding_rejects_ids_outside_the_host_registry() -> None:
    from agents.native.agent import TOOL_SOURCES
    from kdcube_ai_app.apps.chat.sdk.runtime.direct_hosting.native_tool_bindings import (
        resolve_native_tool_bindings,
    )

    config = yaml.safe_load(
        (AGENTS_ROOT / "native" / "config.template.yaml").read_text(encoding="utf-8")
    )
    config = copy.deepcopy(config)
    config["agent"]["tools"].append(
        {"id": "unregistered.read_everything", "enabled": True, "runtime": "local"}
    )

    with pytest.raises(ValueError, match="does not expose configured tools"):
        resolve_native_tool_bindings(
            config,
            sources=TOOL_SOURCES,
            adapter_name="native direct example",
        )


def test_native_event_evidence_resolves_json_tool_calls_and_results() -> None:
    from agents.native.agent import event_source_ids

    blocks = [
        {
            "type": "react.tool.call",
            "mime": "application/json",
            "text": '{"tool_id":"react.memsearch","tool_call_id":"call-1"}',
        },
        {
            "type": "react.tool.result",
            "meta": {"tool_call_id": "call-1"},
        },
    ]

    assert event_source_ids(blocks) == {"react.memsearch"}


def test_missing_isolated_exec_image_fails_before_agent_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kdcube_ai_app.apps.chat.sdk.runtime.direct_hosting import configuration

    calls: list[list[str]] = []
    monkeypatch.setattr(configuration.shutil, "which", lambda _name: "/usr/bin/docker")

    def fake_run(argv, **_kwargs):
        calls.append(list(argv))
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(configuration.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="build it before enabling code execution"):
        configuration.verify_docker_image({"image": "missing-exec:latest"})
    assert calls == [["/usr/bin/docker", "image", "inspect", "missing-exec:latest"]]


def test_direct_recipe_is_an_executable_configuration_contract() -> None:
    source = DIRECT_RECIPE.read_text(encoding="utf-8")
    required = (
        ".venv/bin/python setup_local.py --provider anthropic",
        "docker compose --env-file .env -f compose.yaml up -d --wait",
        "descriptors.local/",
        "storage:\n  kdcube:",
        "storage.claude_code_session.repo",
        "services.git.http_token",
        "models:\n  default_llm_model_id: claude-haiku-4-5-20251001",
        ".venv/bin/python agent.py --check",
        "agent:\n  topic:",
        "instructions:\n    profile: lite:core",
        "additional_instructions: |",
        "id: web_tools.web_search",
        "id: exec_tools.execute_code_python",
        "skills:\n    root: ./skills",
        "settings:\n        filter:",
        "agent.py`'s `TOOL_SOURCES",
        "Dockerfile_Exec",
        "docker image inspect py-code-exec:latest",
        "demonstration: PASS",
    )
    assert not [fragment for fragment in required if fragment not in source]
    assert "api_key_ref" not in source
    assert "password_ref" not in source
    assert "export OPENAI_API_KEY" not in source
    readmes = (
        AGENTS_ROOT / "README.md",
        *(AGENTS_ROOT / name / "README.md" for name in ADAPTERS),
    )
    for readme in readmes:
        assert "run-agent-harness-from-python-README.md" in readme.read_text(encoding="utf-8")


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_each_example_owns_independent_infrastructure(adapter: str) -> None:
    compose = yaml.safe_load(
        (AGENTS_ROOT / adapter / "compose.yaml").read_text(encoding="utf-8")
    )
    assert set(compose["services"]) == {"postgres", "redis"}
    assert compose["services"]["postgres"]["image"] == "pgvector/pgvector:pg16"
    assert "redis:7" in compose["services"]["redis"]["image"]


def test_configure_materializes_standard_descriptors_without_printing_secrets(
    tmp_path: Path,
) -> None:
    from kdcube_ai_app.apps.chat.sdk.runtime.direct_hosting.local_setup import configure

    shutil.copytree(
        AGENTS_ROOT / "claude" / "descriptors.template",
        tmp_path / "descriptors.template",
    )
    descriptors, compose_env, claude_repo = configure(
        tmp_path,
        provider="openai",
        provider_key="provider-secret",
    )

    secret_document = json.loads((descriptors / "secrets.yaml").read_text(encoding="utf-8"))
    compose_source = compose_env.read_text(encoding="utf-8")
    assert secret_document["services"]["openai"]["api_key"] == "provider-secret"
    assert secret_document["infra"]["postgres"]["password"] in compose_source
    assert secret_document["infra"]["redis"]["password"] in compose_source
    assert (descriptors / "assembly.yaml").is_file()
    assert (descriptors / "economics.yaml").is_file()
    assert (descriptors / "gateway.yaml").is_file()
    assert (descriptors / "secrets.yaml").stat().st_mode & 0o777 == 0o600
    assert compose_env.stat().st_mode & 0o777 == 0o600
    assert (claude_repo / "HEAD").is_file()
    with pytest.raises(FileExistsError, match="local configuration already exists"):
        configure(tmp_path, provider="none", provider_key=None)


def test_infrastructure_projects_standard_descriptor_settings(tmp_path: Path) -> None:
    from kdcube_ai_app.apps.chat.sdk.runtime.direct_hosting.infrastructure import (
        postgres_url,
        redis_url,
        storage_uri,
    )

    settings = SimpleNamespace(
        REDIS_PASSWORD="r:e/d@is",
        REDIS_HOST="localhost",
        REDIS_PORT=56379,
        REDIS_DB=3,
        PGPASSWORD="p:o/s@t",
        PGUSER="demo user",
        PGHOST="localhost",
        PGPORT=55432,
        PGDATABASE="demo/db",
        PGSSL=False,
        STORAGE_PATH="./output/conversation-store",
    )

    assert redis_url(settings) == "redis://:r%3Ae%2Fd%40is@localhost:56379/3"
    assert postgres_url(settings) == (
        "postgresql://demo%20user:p%3Ao%2Fs%40t@localhost:55432/demo%2Fdb?sslmode=disable"
    )
    assert storage_uri(settings, descriptors_dir=tmp_path) == (
        tmp_path / "output" / "conversation-store"
    ).resolve().as_uri()


def test_descriptor_activation_normalizes_storage_for_sdk_consumers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kdcube_ai_app.apps.chat.sdk.runtime.direct_hosting import infrastructure

    for name in infrastructure.DESCRIPTOR_FILENAMES:
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    for env_name in (
        "PLATFORM_DESCRIPTORS_DIR",
        "ASSEMBLY_YAML_DESCRIPTOR_PATH",
        "GLOBAL_SECRETS_YAML",
        "ECONOMICS_YAML_DESCRIPTOR_PATH",
        "GATEWAY_YAML_PATH",
    ):
        monkeypatch.setenv(env_name, "")
    settings = SimpleNamespace(STORAGE_PATH="../output/kdcube-storage")
    monkeypatch.setattr(
        "kdcube_ai_app.apps.chat.sdk.config.get_settings",
        Mock(return_value=settings),
    )
    monkeypatch.setattr(
        "kdcube_ai_app.apps.chat.sdk.config_cache.clear_config_cache",
        Mock(),
    )
    monkeypatch.setattr(
        "kdcube_ai_app.infra.secrets.reset_secrets_manager_cache",
        Mock(),
    )

    activated = infrastructure.activate_platform_descriptors(tmp_path)

    assert activated is settings
    assert settings.STORAGE_PATH == str(
        (tmp_path / "../output/kdcube-storage").resolve()
    )


@pytest.mark.asyncio
async def test_langgraph_postgres_checkpointer_runs_idempotent_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from agents.langgraph.agent import open_postgres_checkpointer

    checkpointer = object()
    setup = AsyncMock()
    checkpointer = type("FakeCheckpointer", (), {"setup": setup})()

    class FakeContextManager:
        async def __aenter__(self):
            return checkpointer

        async def __aexit__(self, *_args):
            return None

    monkeypatch.setattr(
        AsyncPostgresSaver,
        "from_conn_string",
        staticmethod(lambda _url: FakeContextManager()),
    )
    settings = SimpleNamespace(PGHOST="localhost", PGPORT=55432, PGDATABASE="kdcube_agents")

    async with open_postgres_checkpointer(settings, "postgresql://unused") as opened:
        assert opened is checkpointer
    setup.assert_awaited_once_with()


def test_examples_require_model_authored_code_and_kdcube_rendering() -> None:
    for adapter in ADAPTERS:
        source = (AGENTS_ROOT / adapter / "agent.py").read_text(encoding="utf-8")
        assert "openpyxl" in source
        assert "execute_python" in source or "execute_code_python" in source
        assert "write_pdf" in source
        assert "construct pdf bytes" in source.lower()
        assert "in python" in source.lower()

    helper_sources = (
        (AGENTS_ROOT / "native" / "tools.py").read_text(encoding="utf-8"),
        (AGENTS_ROOT / "langgraph" / "tools.py").read_text(encoding="utf-8"),
    )
    for source in helper_sources:
        assert "create_briefing" not in source
        assert "from openpyxl" not in source
        assert "from reportlab" not in source


@pytest.mark.asyncio
async def test_langgraph_file_tools_delegate_to_the_turn_runtime() -> None:
    from agents.langgraph.tools import build_tools

    execute = AsyncMock(return_value={"ok": True, "items": []})
    render = AsyncMock(return_value={"ok": True, "items": []})
    runtime = SimpleNamespace(
        execute_python=execute,
        write_pdf=render,
        write_docx=AsyncMock(return_value={"ok": True, "items": []}),
        write_pptx=AsyncMock(return_value={"ok": True, "items": []}),
        tool_report=lambda result: json.dumps(result),
    )
    tools = {
        item.name: item
        for item in build_tools(runtime, enabled_ids={"execute_python", "write_pdf"})
    }

    execute_result = json.loads(
        await tools["execute_python"].ainvoke(
            {
                "code": "print('agent authored')",
                "artifacts": [
                    {
                        "filepath": "files/research/data.xlsx",
                        "description": "Evidence workbook",
                        "visibility": "external",
                    }
                ],
                "program_name": "Research workbook",
                "timeout_s": 300,
            }
        )
    )
    render_result = json.loads(
        await tools["write_pdf"].ainvoke(
            {
                "source_path": "files/research/brief.html",
                "output_path": "files/research/brief.pdf",
                "title": "Brief",
            }
        )
    )

    assert execute_result["ok"] is True
    assert render_result["ok"] is True
    execute.assert_awaited_once()
    render.assert_awaited_once_with(
        source_path="files/research/brief.html",
        output_path="files/research/brief.pdf",
        title="Brief",
    )
