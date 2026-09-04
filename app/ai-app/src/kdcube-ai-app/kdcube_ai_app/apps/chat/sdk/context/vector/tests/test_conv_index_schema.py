from __future__ import annotations

from contextlib import asynccontextmanager

import pytest

from kdcube_ai_app.apps.chat.sdk.context.vector.conv_index import ConvIndex


class _Connection:
    def __init__(self) -> None:
        self.statements: list[str] = []

    async def execute(self, statement: str) -> None:
        self.statements.append(statement)


class _Pool:
    def __init__(self) -> None:
        self.connection = _Connection()

    @asynccontextmanager
    async def acquire(self):
        yield self.connection


@pytest.mark.asyncio
async def test_focused_schema_executes_as_one_postgres_script() -> None:
    pool = _Pool()
    index = ConvIndex(pool=pool, schema="kdcube_test_project")  # type: ignore[arg-type]

    await index.ensure_schema()

    assert len(pool.connection.statements) == 1
    statement = pool.connection.statements[0]
    assert "DO $$" in statement
    assert "CREATE TABLE IF NOT EXISTS kdcube_test_project.conv_messages" in statement
    assert "bundle_id" in statement
    assert "search_tsv" in statement
    assert "<SCHEMA>" not in statement


@pytest.mark.asyncio
async def test_focused_schema_resource_is_the_checked_in_sql() -> None:
    index = ConvIndex.__new__(ConvIndex)
    sql = (await index._read_sql()).decode("utf-8")

    assert "CREATE TABLE IF NOT EXISTS <SCHEMA>.conv_messages" in sql
    assert "CREATE TABLE IF NOT EXISTS <SCHEMA>.conv_artifact_edges" in sql


def test_explicit_schema_does_not_read_ambient_platform_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "kdcube_ai_app.apps.chat.sdk.context.vector.conv_index.get_settings",
        lambda: (_ for _ in ()).throw(AssertionError("ambient settings were read")),
    )

    index = ConvIndex(pool=_Pool(), schema="kdcube_direct")  # type: ignore[arg-type]

    assert index.schema == "kdcube_direct"
