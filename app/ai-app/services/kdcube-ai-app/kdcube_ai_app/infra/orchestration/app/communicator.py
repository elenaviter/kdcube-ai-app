# kdcube_ai_app/infra/orchestration/app/communicator.py
import time
import json
import logging
import os
import contextlib
from typing import Callable, Iterable, Optional, Union, AsyncIterator, List

import redis
import redis.asyncio as aioredis

from dotenv import load_dotenv, find_dotenv

# Load environment
load_dotenv(find_dotenv())

# Logging
logger = logging.getLogger("ServiceCommunicator")


# Configuration
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "")
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = os.environ.get("REDIS_PORT", "6379")
REDIS_URL = os.environ.get("REDIS_URL", f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0")

ORCHESTRATOR_TYPE = os.environ.get("ORCHESTRATOR_TYPE", "dramatiq")
DEFAULT_ORCHESTRATOR_IDENTITY = f"kdcube_orchestrator_{ORCHESTRATOR_TYPE}"
ORCHESTRATOR_IDENTITY = os.environ.get("ORCHESTRATOR_IDENTITY", DEFAULT_ORCHESTRATOR_IDENTITY)

# Queue prefix to match web server expectations
KDCUBE_ORCHESTRATOR_QUEUES_PREFIX = "kdcube_orch_"

class ServiceCommunicator:
    """
    Unified Redis Pub/Sub helper.
    - Sync publish API (used by actors/workers).
    - Async subscribe/listen API (used by Socket.IO service).

    Message schema (JSON):
        {
          "target_sid": str,
          "event": str,
          "data": dict,
          "timestamp": float
        }

    Channels are auto-prefixed with "<ORCHESTRATOR_IDENTITY>." to keep producers/consumers aligned.
    """

    # ---------- construction ----------

    def __init__(
            self,
            redis_url: str = REDIS_URL,
            orchestrator_identity: str = ORCHESTRATOR_IDENTITY,
    ):
        self.redis_url = redis_url
        self.orchestrator_identity = orchestrator_identity

        # sync client for publishing
        self.redis = redis.Redis.from_url(self.redis_url)

        # async client for subscribing
        self._aioredis: Optional[aioredis.Redis] = None
        self._pubsub: Optional[aioredis.client.PubSub] = None
        self._listen_task: Optional["asyncio.Task"] = None
        self._subscribed_channels: List[str] = []

    # ---------- channel helpers ----------

    def _fmt_channel(self, channel: str) -> str:
        """Apply identity prefix."""
        if channel.startswith(self.orchestrator_identity + "."):
            return channel
        return f"{self.orchestrator_identity}.{channel}"

    # ---------- publisher API (sync) ----------

    def pub(self, event: str, target_sid: str, data: dict, channel: str = "kb.process_resource_out"):
        """
        Publish an event to the given channel (sync).
        """
        message = {
            "target_sid": target_sid,
            "event": event,
            "data": data,
            "timestamp": time.time()
        }
        full_channel = self._fmt_channel(channel)
        logger.info(f"Publishing event '{event}' to '{full_channel}' for SID '{target_sid}': {data}")
        self.redis.publish(full_channel, json.dumps(message))

    # ---------- subscriber API (async) ----------

    async def _ensure_async(self):
        if self._aioredis is None:
            self._aioredis = aioredis.Redis.from_url(self.redis_url)

    async def subscribe(self, channels: Union[str, Iterable[str]], *, pattern: bool = False):
        """
        Subscribe (or psubscribe) to one or more channels.
        Call once before listen()/start_listener().
        """
        await self._ensure_async()
        if self._pubsub is None:
            self._pubsub = self._aioredis.pubsub()

        if isinstance(channels, str):
            channels = [channels]

        formatted = [self._fmt_channel(ch) for ch in channels]
        self._subscribed_channels = formatted

        if pattern:
            await self._pubsub.psubscribe(*formatted)
            logger.info(f"Pattern-subscribed to: {formatted}")
        else:
            await self._pubsub.subscribe(*formatted)
            logger.info(f"Subscribed to: {formatted}")

    async def listen(self) -> AsyncIterator[dict]:
        """
        Async iterator yielding decoded payload dicts for 'message'/'pmessage' only.
        Use after subscribe().
        """
        if not self._pubsub:
            raise RuntimeError("Call subscribe() before listen().")

        async for msg in self._pubsub.listen():
            mtype = msg.get("type")
            if mtype not in ("message", "pmessage"):
                # ignore 'subscribe', 'psubscribe', etc.
                continue
            data = msg.get("data")
            if isinstance(data, (bytes, bytearray)):
                try:
                    payload = json.loads(data)
                except Exception as e:
                    logger.error(f"[ServiceCommunicator] JSON decode error: {e} ({data[:200]!r})")
                    continue
            elif isinstance(data, dict):
                payload = data
            else:
                # unexpected
                continue
            yield payload

    async def start_listener(self, on_message: Callable[[dict], "asyncio.Future | None | Any"]):
        """
        Start a background task that invokes on_message(payload) for every message.
        Requires subscribe() called beforehand.
        """
        import asyncio

        if not self._pubsub:
            raise RuntimeError("Call subscribe() before start_listener().")

        async def _loop():
            try:
                async for payload in self.listen():
                    try:
                        res = on_message(payload)
                        if asyncio.iscoroutine(res):
                            await res
                    except Exception as cb_err:
                        logger.error(f"[ServiceCommunicator] on_message error: {cb_err}")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"[ServiceCommunicator] listener error: {e}")

        if self._listen_task and not self._listen_task.done():
            return  # already running

        self._listen_task = asyncio.create_task(_loop(), name="service-communicator-listener")
        logger.info(f"Started listener task on channels: {self._subscribed_channels}")

    async def stop_listener(self):
        """Cancel listener task and close pubsub + connection."""
        import asyncio
        task, self._listen_task = self._listen_task, None
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

        if self._pubsub:
            with contextlib.suppress(Exception):
                if self._subscribed_channels:
                    # attempt to unsubscribe (works for both sub and psub)
                    await self._pubsub.unsubscribe(*self._subscribed_channels)
                    await self._pubsub.punsubscribe(*self._subscribed_channels)
            with contextlib.suppress(Exception):
                await self._pubsub.close()
            self._pubsub = None
            self._subscribed_channels = []

        if self._aioredis:
            with contextlib.suppress(Exception):
                await self._aioredis.close()
            self._aioredis = None
        logger.info("Stopped listener and closed async Redis.")


    # ==============================================================================
    #                           UTILITY FUNCTIONS
    # ==============================================================================

    def get_queue_stats(self):
        """Get queue statistics - matches what the orchestrator interface expects"""
        redis_client = redis.Redis.from_url(REDIS_URL)
        stats = {}

        queues = [
            f"{KDCUBE_ORCHESTRATOR_QUEUES_PREFIX}low_priority",
            f"{KDCUBE_ORCHESTRATOR_QUEUES_PREFIX}high_priority",
            f"{KDCUBE_ORCHESTRATOR_QUEUES_PREFIX}batch",
            "health_check"
        ]

        for queue in queues:
            queue_key = f"dramatiq:default.{queue}"
            length = redis_client.llen(queue_key)
            stats[queue] = length

        return stats

    def get_task_result(self,
                        message_id: str):
        """Get result of a completed task"""
        from dramatiq.results.backends import RedisBackend

        try:
            result_backend = RedisBackend(url=REDIS_URL)
            return result_backend.get_result(message_id, block=False)
        except Exception as e:
            logger.error(f"Failed to get task result for {message_id}: {e}")
            return None