# SPDX-License-Identifier: MIT
"""Range-rebuild job: completion is observable state, not a held connection.

Runs against the process-local record store (Redis absent), with the per-day
aggregation stubbed; asserts the record lifecycle the status endpoint serves."""
import asyncio
import time
from datetime import date

import kdcube_ai_app.apps.chat.ingress.opex.routines as routines


async def _no_redis():
    return None


def _isolate(monkeypatch):
    monkeypatch.setattr(routines, "_get_agg_redis", _no_redis)
    monkeypatch.setattr(routines, "_local_range_job", None)
    # Record data-bus broadcasts instead of touching a relay.
    emitted = []

    async def fake_emit(record):
        emitted.append(dict(record))

    monkeypatch.setattr(routines, "_emit_range_job_event", fake_emit)
    return emitted


def test_job_runs_to_done_with_day_progress(monkeypatch):
    emitted = _isolate(monkeypatch)
    days = []

    async def fake_day(run_date, *, recompute=False):
        days.append(run_date.isoformat())

    monkeypatch.setattr(routines, "_run_daily_and_monthly_for_date", fake_day)

    async def scenario():
        record, started = await routines.start_range_job(date(2026, 7, 1), date(2026, 7, 3))
        assert started and record["status"] == "running"
        assert record["days_total"] == 3
        # let the background task drain
        for _ in range(50):
            await asyncio.sleep(0)
            job = await routines.read_range_job()
            if job and job["status"] == "done":
                return job
        raise AssertionError("job never finished")

    job = asyncio.run(scenario())
    assert job["days_done"] == 3
    assert job["finished_at"]
    assert days == ["2026-07-01", "2026-07-02", "2026-07-03"]
    # every state change was broadcast on the data bus: claim + 2/day + done
    assert [e["status"] for e in emitted] == ["running"] * 7 + ["done"]
    assert emitted[-1]["days_done"] == 3


def test_failure_is_recorded_not_swallowed(monkeypatch):
    _isolate(monkeypatch)

    async def boom(run_date, *, recompute=False):
        raise RuntimeError("storage exploded")

    monkeypatch.setattr(routines, "_run_daily_and_monthly_for_date", boom)

    async def scenario():
        await routines.start_range_job(date(2026, 7, 1), date(2026, 7, 2))
        for _ in range(50):
            await asyncio.sleep(0)
            job = await routines.read_range_job()
            if job and job["status"] == "failed":
                return job
        raise AssertionError("failure never recorded")

    job = asyncio.run(scenario())
    assert "storage exploded" in job["error"]


def test_live_job_blocks_second_start_but_stalled_is_replaced(monkeypatch):
    _isolate(monkeypatch)

    async def scenario():
        # a live running record (fresh heartbeat) blocks a new start
        routines._local_range_job = {
            "job_id": "j1", "status": "running", "start_date": "2026-07-01",
            "end_date": "2026-07-05", "days_total": 5, "days_done": 1,
            "current_day": "2026-07-02", "heartbeat": time.time(),
        }
        record, started = await routines.start_range_job(date(2026, 7, 1), date(2026, 7, 5))
        assert not started and record["job_id"] == "j1"

        # a stalled one (old heartbeat) is replaced by a fresh job
        routines._local_range_job["heartbeat"] = time.time() - routines.RANGE_JOB_STALL_SECONDS - 1
        stale = await routines.read_range_job()
        assert stale["stalled"] is True

        async def fake_day(run_date, *, recompute=False):
            pass
        monkeypatch.setattr(routines, "_run_daily_and_monthly_for_date", fake_day)
        record2, started2 = await routines.start_range_job(date(2026, 7, 1), date(2026, 7, 5))
        assert started2 and record2["job_id"] != "j1"

    asyncio.run(scenario())
