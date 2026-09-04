from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import yaml


AGENTS_ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = ("native", "langgraph", "claude")
README_SECTIONS = (
    "## What it is",
    "## Run it",
    "## What the demo shows",
    "## Change the demo",
)


def test_every_agents_readme_answers_the_four_first_use_questions() -> None:
    readmes = sorted(AGENTS_ROOT.rglob("README.md"))
    assert readmes
    for path in readmes:
        source = path.read_text(encoding="utf-8")
        positions = [source.find(heading) for heading in README_SECTIONS]
        assert all(position >= 0 for position in positions), path
        assert positions == sorted(positions), path


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_each_example_owns_its_runnable_contract(adapter: str) -> None:
    root = AGENTS_ROOT / adapter
    expected = {"agent.py", "config.template.yaml", "requirements.txt", "README.md"}
    assert expected.issubset({path.name for path in root.iterdir()})

    config = yaml.safe_load((root / "config.template.yaml").read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    assert isinstance(config.get("output"), dict)
    assert isinstance(config.get("infra", {}).get("redis"), dict)
    assert isinstance(config.get("infra", {}).get("postgres"), dict)
    assert isinstance(config.get("infra", {}).get("storage"), dict)
    assert str(config["infra"]["redis"].get("password_ref") or "")
    assert str(config["infra"]["postgres"].get("password_ref") or "")
    assert str(config["infra"]["storage"].get("uri") or "")

    requirements = (root / "requirements.txt").read_text(encoding="utf-8")
    assert "../../app/ai-app/src/kdcube-ai-app" in requirements


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
def test_primary_examples_use_shared_direct_harness(adapter: str) -> None:
    source = (AGENTS_ROOT / adapter / "agent.py").read_text(encoding="utf-8")
    assert "DirectAgentHarness" in source
    assert "harness.turn(" in source
    assert "verify_conversation(" in source


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_offline_adapter_construction(adapter: str) -> None:
    root = AGENTS_ROOT / adapter
    env = os.environ.copy()
    env.pop("KDCUBE_RUNTIME_WORKDIR", None)
    completed = subprocess.run(
        [sys.executable, "agent.py", "--check"],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "mode: standalone SDK process (no KDCube runtime)" in completed.stdout
    assert "check: PASS" in completed.stdout


def test_independent_infrastructure_compose_contract() -> None:
    compose = yaml.safe_load((AGENTS_ROOT / "compose.yaml").read_text(encoding="utf-8"))
    assert set(compose["services"]) == {"postgres", "redis"}
    assert compose["services"]["postgres"]["image"] == "pgvector/pgvector:pg16"
    assert "redis:7" in compose["services"]["redis"]["image"]


def test_infrastructure_resolves_referenced_secrets_and_storage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from agents.infrastructure import postgres_url, redis_url, storage_uri

    config = {
        "infra": {
            "redis": {
                "host": "localhost",
                "port": 56379,
                "database": 3,
                "password_ref": "REDIS_TEST_SECRET",
            },
            "postgres": {
                "host": "localhost",
                "port": 55432,
                "user": "demo user",
                "database": "demo/db",
                "password_ref": "POSTGRES_TEST_SECRET",
                "sslmode": "disable",
            },
            "storage": {"uri": "./output/conversation-store"},
        }
    }
    monkeypatch.setenv("REDIS_TEST_SECRET", "r:e/d@is")
    monkeypatch.setenv("POSTGRES_TEST_SECRET", "p:o/s@t")

    assert redis_url(config) == "redis://:r%3Ae%2Fd%40is@localhost:56379/3"
    assert postgres_url(config) == (
        "postgresql://demo%20user:p%3Ao%2Fs%40t@localhost:55432/demo%2Fdb?sslmode=disable"
    )
    assert storage_uri(config, config_path=tmp_path / "config.yaml") == (
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
    config = {
        "infra": {
            "postgres": {
                "host": "localhost",
                "port": 55432,
                "database": "kdcube_agents",
            }
        }
    }

    async with open_postgres_checkpointer(config, "postgresql://unused") as opened:
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
            [{"title": "Source <one>", "body": "Finding A & B", "url": "https://example.com?a=1&b=2"}],
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
