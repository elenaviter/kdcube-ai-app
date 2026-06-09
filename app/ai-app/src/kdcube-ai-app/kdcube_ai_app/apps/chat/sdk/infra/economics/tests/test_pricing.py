# SPDX-License-Identifier: MIT

"""
Effective-dated pricing: table_as_of() returns the rate in effect at a given
time, so a price change applies from its effective_from forward while older
events keep the prior price. DB is faked (no Postgres).
"""
import asyncio
from datetime import datetime, timezone

import kdcube_ai_app.apps.chat.sdk.infra.economics.pricing as pricing
import kdcube_ai_app.apps.chat.sdk.infra.economics.usage_ledger as ul

SONNET = "claude-sonnet-4-5-20250929"
T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)   # original price
T1 = datetime(2026, 6, 1, tzinfo=timezone.utc)   # price change


class _Conn:
    def __init__(self, rows):
        self.rows = rows
    async def fetch(self, sql, when):
        # emulate DISTINCT ON (service,provider,model) latest effective_from <= when
        latest = {}
        for r in self.rows:
            if r["effective_from"] <= when:
                k = (r["service_type"], r["provider"], r["model"])
                if k not in latest or r["effective_from"] > latest[k]["effective_from"]:
                    latest[k] = r
        return list(latest.values())
    async def fetchrow(self, sql, *a):
        return self.rows[0] if self.rows else None


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


ROWS = [
    {"service_type": "llm", "provider": "anthropic", "model": SONNET,
     "rates": {"provider": "anthropic", "model": SONNET, "input_tokens_1M": 3.0, "output_tokens_1M": 15.0},
     "effective_from": T0},
    {"service_type": "llm", "provider": "anthropic", "model": SONNET,
     "rates": {"provider": "anthropic", "model": SONNET, "input_tokens_1M": 4.0, "output_tokens_1M": 20.0},
     "effective_from": T1},
]


def _store(rows):
    return pricing.ModelPricingStore(FakePool(rows), tenant="home", project="demo")


def test_rate_before_change():
    store = _store(ROWS)
    tbl = asyncio.run(store.table_as_of(datetime(2026, 3, 1, tzinfo=timezone.utc)))
    cost = ul.cost_usd_for_event(service_type="llm", provider="anthropic", model=SONNET,
                                 usage={"output_tokens": 1_000_000}, pricing_table=tbl)
    assert round(cost, 2) == 15.00


def test_rate_after_change():
    store = _store(ROWS)
    tbl = asyncio.run(store.table_as_of(datetime(2026, 6, 15, tzinfo=timezone.utc)))
    cost = ul.cost_usd_for_event(service_type="llm", provider="anthropic", model=SONNET,
                                 usage={"output_tokens": 1_000_000}, pricing_table=tbl)
    assert round(cost, 2) == 20.00


def test_unseeded_returns_none():
    # No rows -> None, so callers fall back to the in-code price table.
    assert asyncio.run(_store([]).table_as_of(T1)) is None
