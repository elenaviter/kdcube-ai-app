# SPDX-License-Identifier: MIT

"""
Tests for the SQL-backed usage ledger (UsageLedgerStore): per-event cost uses
real per-model rates, and the read-side assembles/sorts per-user spend from the
aggregation view. The DB connection is faked (no Postgres needed).
"""
import asyncio

import kdcube_ai_app.apps.chat.sdk.infra.economics.usage_ledger as ul

SONNET = "claude-sonnet-4-5-20250929"   # $3/1M in, $15/1M out
HAIKU = "claude-haiku-4-5-20251001"     # $1/1M in, $5/1M out


class _Conn:
    def __init__(self, rows):
        self.rows = rows
    async def fetch(self, sql, *a):
        return self.rows
    async def fetchrow(self, sql, *a):
        return self.rows[0] if self.rows else None
    async def execute(self, sql, *a):
        return "OK"


class _Acquire:
    def __init__(self, conn):
        self.conn = conn
    async def __aenter__(self):
        return self.conn
    async def __aexit__(self, *a):
        return False


class FakePool:
    def __init__(self, rows):
        self._conn = _Conn(rows)
    def acquire(self):
        return _Acquire(self._conn)


def _row(user_id, model, cost, inp=0, out=0, reqs=1):
    return {"user_id": user_id, "service_type": "llm", "provider": "anthropic",
            "model": model, "cost_usd": cost, "input_tokens": inp, "output_tokens": out,
            "embedding_tokens": 0, "requests": reqs}


class TestPerEventCost:
    def test_input_output_priced_per_model(self):
        c = ul.cost_usd_for_event(service_type="llm", provider="anthropic", model=SONNET,
                                  usage={"input_tokens": 1_000_000, "output_tokens": 1_000_000})
        assert round(c, 2) == 18.00   # 1M*$3 + 1M*$15, not blended-output $30

    def test_model_specific(self):
        son = ul.cost_usd_for_event(service_type="llm", provider="anthropic", model=SONNET, usage={"output_tokens": 1_000_000})
        hai = ul.cost_usd_for_event(service_type="llm", provider="anthropic", model=HAIKU, usage={"output_tokens": 1_000_000})
        assert round(son, 2) == 15.00 and round(hai, 2) == 5.00 and son > hai

    def test_prefers_provider_reported_cost(self):
        # When the provider reports the billed cost, use it verbatim.
        c = ul.cost_usd_for_event(service_type="llm", provider="anthropic", model=SONNET,
                                  usage={"input_tokens": 1_000_000, "output_tokens": 1_000_000, "cost_usd": 0.42})
        assert c == 0.42

    def test_reported_zero_is_ignored(self):
        # A non-positive reported cost is ignored; we compute from the table.
        c = ul.cost_usd_for_event(service_type="llm", provider="anthropic", model=SONNET,
                                  usage={"output_tokens": 1_000_000, "cost_usd": 0})
        assert round(c, 2) == 15.00


class TestReadSide:
    ROWS = [
        _row("alice", SONNET, 3.0, inp=1_000_000, reqs=2),
        _row("bob", HAIKU, 5.0, out=1_000_000, reqs=3),
        _row("alice", HAIKU, 1.0, out=200_000, reqs=1),
    ]

    def test_cost_by_user_sorted_and_totaled(self):
        store = ul.UsageLedgerStore(FakePool(self.ROWS), tenant="home", project="demo")
        assert store.schema == "home_demo"
        res = asyncio.run(store.cost_by_user(date_from="2026-06-01", date_to="2026-06-09"))
        assert [u["user_id"] for u in res["users"]] == ["bob", "alice"]
        assert round(res["total_cost_usd"], 2) == 9.0
        alice = next(u for u in res["users"] if u["user_id"] == "alice")
        assert round(alice["total_cost_usd"], 2) == 4.0
        assert [m["cost_usd"] for m in alice["by_model"]] == [3.0, 1.0]   # sorted desc
        assert alice["event_count"] == 3                                  # summed requests
        assert alice["tokens"]["input_tokens"] == 1_000_000

    def test_cost_for_user_filters_and_windows(self):
        store = ul.UsageLedgerStore(FakePool([r for r in self.ROWS if r["user_id"] == "alice"]),
                                    tenant="home", project="demo")
        res = asyncio.run(store.cost_for_user(user_id="alice", date_from="2026-06-01", date_to="2026-06-09"))
        assert round(res["total_cost_usd"], 2) == 4.0
        assert res["date_from"] == "2026-06-01" and res["date_to"] == "2026-06-09"

    def test_missing_user_is_zero(self):
        store = ul.UsageLedgerStore(FakePool([]), tenant="home", project="demo")
        res = asyncio.run(store.cost_for_user(user_id="ghost", date_from="a", date_to="b"))
        assert res["total_cost_usd"] == 0.0 and res["by_model"] == []
