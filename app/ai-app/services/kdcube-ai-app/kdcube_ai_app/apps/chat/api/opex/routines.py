# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# # kdcube_ai_app/apps/chat/api/opex/routines.py

import asyncio
from typing import Optional
import uuid, os, logging
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
from croniter import croniter

import redis.asyncio as aioredis

from kdcube_ai_app.apps.chat.sdk.config import get_settings
from kdcube_ai_app.infra.accounting.aggregator import AccountingAggregator
from kdcube_ai_app.storage.storage import create_storage_backend

logger = logging.getLogger("OPEX.Routines")

ACCOUNTING_TZ = ZoneInfo("Europe/Berlin")

_scheduler_task: Optional[asyncio.Task] = None
_aggregator: Optional[AccountingAggregator] = None
_agg_redis: Optional[aioredis.Redis] = None

def _get_aggregator() -> AccountingAggregator:
    """
    Lazily create a shared AccountingAggregator instance for the scheduler.
    Uses the same STORAGE_PATH logic as _get_calculator.
    """
    global _aggregator
    if _aggregator is not None:
        return _aggregator

    _settings = get_settings()
    kdcube_path = _settings.STORAGE_PATH or "file:///tmp/kdcube_data"
    backend = create_storage_backend(kdcube_path)
    _aggregator = AccountingAggregator(backend, base_path="accounting")
    return _aggregator


async def _get_agg_redis() -> Optional[aioredis.Redis]:
    """
    Redis connection for distributed locks between API instances.

    Configure via REDIS_URL, e.g.:
      REDIS_URL=redis://redis:6379/0
    """
    global _agg_redis
    if _agg_redis is not None:
        return _agg_redis

    _settings = get_settings()
    redis_url = _settings.REDIS_URL
    if not redis_url:
        logger.warning(
            "[OPEX Aggregator] REDIS_URL is not set; "
            "scheduled aggregation will run *without* distributed locking."
        )
        _agg_redis = None
        return None

    _agg_redis = aioredis.Redis.from_url(redis_url)
    return _agg_redis

def _get_cron_expression() -> str:
    """
    Return cron expression from settings or env.

    Priority:
      1) settings.OPEX_AGG_CRON if exists
      2) env OPEX_AGG_CRON
      3) default: "0 3 * * *" (daily at 03:00)
    """
    expr = None
    try:
        _settings = get_settings()
        expr = getattr(_settings, "OPEX_AGG_CRON", None)
    except Exception:
        # if settings don’t have it, silently ignore
        expr = None

    expr = expr or os.getenv("OPEX_AGG_CRON")
    if not expr:
        expr = "0 3 * * *"
    return expr


def _compute_next_run(now: datetime) -> datetime:
    """
    Compute next run datetime in ACCOUNTING_TZ from cron expression.
    Falls back to '0 3 * * *' if the configured cron is invalid.
    """
    expr = _get_cron_expression()
    try:
        it = croniter(expr, now)
        return it.get_next(datetime)
    except Exception:
        logger.exception(
            "[OPEX Aggregator] Invalid cron expression '%s', "
            "falling back to '0 3 * * *'",
            expr,
        )
        it = croniter("0 3 * * *", now)
        return it.get_next(datetime)


async def run_aggregation_range(start: date, end: date) -> None:
    """
    Run daily + monthly aggregation for each date in [start, end], inclusive.

    Uses the same Redis-locking logic as the scheduler, so multiple
    API instances can safely call this at the same time.
    """
    if end < start:
        return

    current = start
    while current <= end:
        await _run_daily_and_monthly_for_date(current)
        current += timedelta(days=1)

async def _run_daily_and_monthly_for_date(run_date: date) -> None:
    """
    Compute daily aggregate for run_date and monthly aggregate for its month.

    Uses a Redis lock so only one instance per (tenant, project, date)
    actually does the work.
    """
    agg = _get_aggregator()
    redis = await _get_agg_redis()

    tenant = os.getenv("DEFAULT_TENANT", "home")
    project = os.getenv("DEFAULT_PROJECT_NAME", "demo")

    lock_key = f"acct:agg:{tenant}:{project}:{run_date.isoformat()}"
    token = str(uuid.uuid4())
    lock_ttl_seconds = 4 * 3600  # 4 hours

    # Acquire distributed lock if Redis is available
    if redis:
        got_lock = await redis.set(lock_key, token, ex=lock_ttl_seconds, nx=True)
        if not got_lock:
            logger.info(
                "[OPEX Aggregator] Another instance holds lock %s, "
                "skipping aggregation for %s/%s on %s",
                lock_key, tenant, project, run_date.isoformat()
            )
            return
    else:
        logger.info(
            "[OPEX Aggregator] Redis not configured, running aggregation for %s/%s "
            "on %s without distributed lock",
            tenant, project, run_date.isoformat()
        )

    try:
        date_str = run_date.isoformat()

        # Daily (only this date, skip_existing=True so reruns are safe)
        logger.info(
            "[OPEX Aggregator] Aggregating daily for %s/%s on %s",
            tenant, project, date_str
        )
        await agg.aggregate_daily_range_for_project(
            tenant_id=tenant,
            project_id=project,
            date_from=date_str,
            date_to=date_str,
            skip_existing=True,
        )

        # Monthly
        logger.info(
            "[OPEX Aggregator] Aggregating monthly for %s/%s year=%d month=%d",
            tenant, project, run_date.year, run_date.month
        )
        await agg.aggregate_monthly_from_daily(
            tenant_id=tenant,
            project_id=project,
            year=run_date.year,
            month=run_date.month,
            require_full_coverage=False,
        )

        logger.info(
            "[OPEX Aggregator] Aggregation done for %s/%s on %s",
            tenant, project, run_date.isoformat()
        )

    except Exception:
        logger.exception(
            "[OPEX Aggregator] Failed to aggregate for %s/%s on %s",
            tenant, project, run_date.isoformat()
        )
    finally:
        if redis:
            try:
                current_val = await redis.get(lock_key)
                if current_val is not None and current_val.decode() == token:
                    await redis.delete(lock_key)
            except Exception:
                logger.exception(
                    "[OPEX Aggregator] Failed to release lock %s", lock_key
                )

async def aggregation_scheduler_loop() -> None:
    """
    Background loop:

      - reads cron from OPEX_AGG_CRON (or settings.OPEX_AGG_CRON)
      - waits until the next scheduled time in Europe/Berlin
      - runs aggregation for *yesterday* (Berlin date)
      - repeats forever
    """
    logger.info(
        "[OPEX Aggregator] Scheduler loop started (tz=%s, cron=%s)",
        ACCOUNTING_TZ,
        _get_cron_expression(),
    )

    while True:
        now = datetime.now(ACCOUNTING_TZ)
        next_run = _compute_next_run(now)

        sleep_seconds = (next_run - now).total_seconds()
        logger.info(
            "[OPEX Aggregator] Sleeping %.0f seconds until next run at %s",
            sleep_seconds,
            next_run.isoformat(),
        )

        try:
            await asyncio.sleep(sleep_seconds)
        except asyncio.CancelledError:
            logger.info("[OPEX Aggregator] Scheduler loop cancelled")
            break

        # Semantics: always aggregate "yesterday" in Berlin time.
        run_date = (datetime.now(ACCOUNTING_TZ) - timedelta(days=1)).date()
        logger.info(
            "[OPEX Aggregator] Running scheduled aggregation for date %s (trigger=%s)",
            run_date.isoformat(),
            next_run.isoformat(),
        )
        await _run_daily_and_monthly_for_date(run_date)


