from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

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
        "configure.py",
        "descriptors.template",
        "requirements.txt",
        "skills",
        "web-search.yaml",
    }
    assert expected.issubset({path.name for path in root.iterdir()})

    config = yaml.safe_load((root / "config.template.yaml").read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    assert isinstance(config.get("output"), dict)
    assert "infra" not in config
    assert str(config.get("agent", {}).get("instructions") or "").strip()
    assert isinstance(config.get("agent", {}).get("tools"), list)
    assert config["agent"]["tools"]
    assert all(str(item.get("id") or "").strip() for item in config["agent"]["tools"])
    assert config.get("agent", {}).get("skills", {}).get("enabled") == [
        "demo.research-brief"
    ]
    assert config.get("web_search", {}).get("config") == "./web-search.yaml"
    web_policy = yaml.safe_load((root / "web-search.yaml").read_text(encoding="utf-8"))
    assert web_policy["filter"]["allowlist"] == ["python.org"]
    assert web_policy["filter"]["ssrf_guard"] is True
    if adapter != "claude":
        assert "model" not in config

    requirements = (root / "requirements.txt").read_text(encoding="utf-8")
    assert "../../app/ai-app/src/kdcube-ai-app" in requirements
    assert "sdk/tools/mcp/web_search/requirements.txt" in requirements


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
    assert "mcp__kdcube_web_search__web_search" in claude_tools
    assert "mcp__kdcube_web_search__web_fetch" not in claude_tools
    assert "mcp__kdcube_web_search__allowlist_status" not in claude_tools
    assert "kdcube_ai_app.apps.chat.sdk.tools.mcp.web_search.web_search_server" in claude_source
    assert 'denied_tools=("WebSearch", "WebFetch")' in claude_source


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
    tmp_path: Path,
    monkeypatch,
) -> None:
    from agents.langgraph import tools as langgraph_tools

    search = AsyncMock(
        return_value=[
            {"title": "Python", "text": "Current release", "url": "https://python.org/"}
        ]
    )
    monkeypatch.setattr(langgraph_tools.web_search_server, "web_search", search)
    tools = langgraph_tools.build_tools(tmp_path, enabled_ids={"web_search"})

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
    config = (AGENTS_ROOT / "claude" / "web-search.yaml").resolve()
    async with open_mcp_client(
        transport="stdio",
        command=sys.executable,
        args=["-m", module, "--transport", "stdio", "--config", str(config)],
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
    assert str(assembly.get("storage", {}).get("kdcube") or "")
    assert assembly["models"]["default_llm_model_id"] == "gpt-4o-mini"
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
    assert economics.get("price_tables", {}).get("llm")
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
    assert "model:" in completed.stdout
    assert "check: PASS" in completed.stdout


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
    from agents.native.configuration import EXEC_TOOL_ID, build_native_tool_plan

    config = yaml.safe_load(
        (AGENTS_ROOT / "native" / "config.template.yaml").read_text(encoding="utf-8")
    )
    config = copy.deepcopy(config)
    for tool in config["agent"]["tools"]:
        if tool["id"] == "demo.create_briefing":
            tool["enabled"] = False
        elif tool["id"] == EXEC_TOOL_ID:
            tool["enabled"] = True

    exec_profile = {
        "mode": "docker",
        "image": "py-code-exec:latest",
        "container_strategy": "split",
        "network_mode": "none",
    }
    plan = build_native_tool_plan(
        config,
        tools_file=AGENTS_ROOT / "native" / "tools.py",
        platform_exec_runtime=exec_profile,
    )

    assert plan.enabled_ids == ("demo.web_search", EXEC_TOOL_ID)
    assert plan.tool_runtime[EXEC_TOOL_ID] == "docker"
    assert plan.allowed_tool_names_by_alias["exec_tools"] == ["execute_code_python"]
    assert plan.exec_runtime == exec_profile
    assert any(spec.get("alias") == "exec_tools" for spec in plan.tools_specs)


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
        ".venv/bin/python configure.py --provider openai",
        "docker compose --env-file .env -f compose.yaml up -d --wait",
        "descriptors.local/",
        "storage:\n  kdcube:",
        "storage.claude_code_session.repo",
        "services.git.http_token",
        "models:\n  default_llm_model_id: gpt-4o-mini",
        ".venv/bin/python agent.py --check",
        "agent:\n  topic:",
        "id: demo.web_search",
        "id: exec_tools.execute_code_python",
        "skills:\n    root: ./skills",
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


def test_native_file_tool_creates_real_pdf_and_xlsx(tmp_path: Path) -> None:
    sys.path.insert(0, str(AGENTS_ROOT / "native"))
    from tools import create_briefing

    from kdcube_ai_app.apps.chat.sdk.runtime.run_ctx import OUTDIR_CV

    token = OUTDIR_CV.set(str(tmp_path))
    try:
        result = create_briefing(
            "Harness <evidence>",
            "A local artifact-generation check with A & B.",
            [
                {
                    "title": "Source <one>",
                    "body": "Finding A & B",
                    "url": "https://example.com?a=1&b=2",
                }
            ],
        )
    finally:
        OUTDIR_CV.reset(token)

    artifact_root = tmp_path / "workdir"
    pdf = artifact_root / "research-brief.pdf"
    xlsx = artifact_root / "research-data.xlsx"
    assert result["ok"] is True
    assert pdf.read_bytes().startswith(b"%PDF-")
    assert xlsx.read_bytes().startswith(b"PK")


def test_langgraph_file_tool_creates_real_pdf_and_xlsx(tmp_path: Path) -> None:
    from agents.langgraph.tools import build_tools

    tools = {item.name: item for item in build_tools(tmp_path)}
    result = tools["create_briefing"].invoke(
        {
            "title": "Harness <evidence>",
            "summary": "A local artifact-generation check with A & B.",
            "findings_json": (
                '[{"title":"Source <one>","body":"Finding A & B",'
                '"url":"https://example.com?a=1&b=2"}]'
            ),
        }
    )

    assert '"ok": true' in result
    assert (tmp_path / "research-brief.pdf").read_bytes().startswith(b"%PDF-")
    assert (tmp_path / "research-data.xlsx").read_bytes().startswith(b"PK")
