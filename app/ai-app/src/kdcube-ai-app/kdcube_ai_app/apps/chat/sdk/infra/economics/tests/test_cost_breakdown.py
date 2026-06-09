# SPDX-License-Identifier: MIT

"""
Tests for the shared true-cost helpers (ingress/opex/cost.py).

Two concerns are covered:

1. Real per-model pricing: spend is computed with each model's OWN input and
   output rates (and is model-specific), NOT the blended
   `equivalent_tokens * reference_output_rate` estimate the rate limiter uses.

2. Assembly/aggregation: cost_for_user self-filters and cost_by_user totals and
   sorts per-user spend descending.

No DB, storage, or real settings are touched: the calculator is faked and the
only settings access inside the cost engine is monkeypatched.
"""

import asyncio

import kdcube_ai_app.apps.chat.ingress.opex.cost as cost_mod

SONNET = "claude-sonnet-4-5-20250929"   # $3/1M in, $15/1M out
HAIKU = "claude-haiku-4-5-20251001"     # $1/1M in, $5/1M out


def _llm_rollup(model, inp, out):
    return [{"service": "llm", "provider": "anthropic", "model": model,
             "spent": {"input": inp, "output": out}}]


class FakeCalc:
    def __init__(self, by_user):
        self._by_user = by_user
        self.calls = []

    async def usage_by_user(self, *, tenant_id, project_id, date_from, date_to, **kw):
        self.calls.append((tenant_id, project_id, date_from, date_to))
        return self._by_user


# ---------------------------------------------------------------------------
# Real per-model pricing
# ---------------------------------------------------------------------------

class TestRealPricing:
    def test_input_and_output_priced_with_distinct_rates(self):
        # 1M input @ $3 + 1M output @ $15 = $18.
        # The old blended estimate would charge 2M tokens @ the $15 output rate = $30.
        res = cost_mod.assemble_cost(_llm_rollup(SONNET, 1_000_000, 1_000_000))
        assert round(res["total_cost_usd"], 2) == 18.00
        assert res["tokens"]["input_tokens"] == 1_000_000
        assert res["tokens"]["output_tokens"] == 1_000_000

    def test_pricing_is_model_specific(self):
        # Same token shape, cheaper model -> strictly cheaper. A single blended
        # reference rate would price these identically.
        sonnet = cost_mod.assemble_cost(_llm_rollup(SONNET, 0, 1_000_000))
        haiku = cost_mod.assemble_cost(_llm_rollup(HAIKU, 0, 1_000_000))
        assert round(sonnet["total_cost_usd"], 2) == 15.00
        assert round(haiku["total_cost_usd"], 2) == 5.00
        assert sonnet["total_cost_usd"] > haiku["total_cost_usd"]

    def test_unknown_model_is_zero_not_reference_priced(self):
        res = cost_mod.assemble_cost(_llm_rollup("totally-unknown-model", 1_000_000, 1_000_000))
        assert res["total_cost_usd"] == 0.0


# ---------------------------------------------------------------------------
# Aggregation / assembly
# ---------------------------------------------------------------------------

class TestAggregation:
    def _stub_cost(self, monkeypatch):
        # Deterministic stub: cost = (input + output) / 1000, so ordering is
        # driven purely by token volume and independent of the price table.
        def _stub(rollup):
            tot = 0
            for it in rollup or []:
                sp = it.get("spent", {}) or {}
                tot += int(sp.get("input", 0)) + int(sp.get("output", 0))
            return {"total_cost_usd": tot / 1000.0,
                    "breakdown": [{"service": it.get("service"), "provider": it.get("provider"),
                                   "model": it.get("model"), "cost_usd": None} for it in (rollup or [])]}
        monkeypatch.setattr(cost_mod, "compute_cost_estimate", _stub)

    def test_cost_for_user_self_filters(self, monkeypatch):
        self._stub_cost(monkeypatch)
        by_user = {
            "alice": {"rollup": _llm_rollup(SONNET, 1000, 2000), "event_count": 4},
            "bob": {"rollup": _llm_rollup(HAIKU, 500, 500), "total": {"event_count": 9}},
        }
        calc = FakeCalc(by_user)
        res = asyncio.run(cost_mod.cost_for_user(
            calc, tenant="t", project="p", user_id="alice",
            date_from="2026-06-01", date_to="2026-06-09"))
        assert res["user_id"] == "alice"
        assert res["total_cost_usd"] == 3.0          # (1000+2000)/1000
        assert res["event_count"] == 4
        assert calc.calls == [("t", "p", "2026-06-01", "2026-06-09")]

    def test_cost_for_user_missing_user_is_zero(self, monkeypatch):
        self._stub_cost(monkeypatch)
        calc = FakeCalc({"alice": {"rollup": _llm_rollup(SONNET, 1000, 0)}})
        res = asyncio.run(cost_mod.cost_for_user(
            calc, tenant="t", project="p", user_id="ghost",
            date_from="2026-06-01", date_to="2026-06-09"))
        assert res["total_cost_usd"] == 0.0
        assert res["by_model"] == []

    def test_cost_by_user_totals_and_sorts_desc(self, monkeypatch):
        self._stub_cost(monkeypatch)
        by_user = {
            "alice": {"rollup": _llm_rollup(SONNET, 1000, 2000), "event_count": 4},   # $3.0
            "bob": {"rollup": _llm_rollup(HAIKU, 5000, 5000), "event_count": 9},       # $10.0
            "carol": {"rollup": _llm_rollup(SONNET, 0, 0), "event_count": 1},          # $0.0
        }
        res = asyncio.run(cost_mod.cost_by_user(
            calc=FakeCalc(by_user), tenant="t", project="p",
            date_from="2026-06-01", date_to="2026-06-09"))
        assert res["total_users"] == 3
        assert [u["user_id"] for u in res["users"]] == ["bob", "alice", "carol"]
        assert res["users"][0]["total_cost_usd"] == 10.0
        assert round(res["total_cost_usd"], 2) == 13.0
        assert res["users"][1]["event_count"] == 4
