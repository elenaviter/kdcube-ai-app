# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

import os

from kdcube_ai_app.apps.chat.reg import MODEL_CONFIGS, EMBEDDERS


class REDIS:
    class CHAT:
        PROMPT_QUEUE_PREFIX = "kdcube:chat:prompt:queue"

    class INSTANCE:
        HEARTBEAT_PREFIX = "kdcube:heartbeat:instance"

    class PROCESS:
        HEARTBEAT_PREFIX = "kdcube:heartbeat:process"

    SESSION = "kdcube:session"

    class THROTTLING:
        EVENTS_KEY = "kdcube:throttling:events"
        STATS_KEY = "kdcube:throttling:stats"
        SESSION_COUNTERS_KEY = "kdcube:throttling:session_counters"
        TOTAL_REQUESTS_KEY = "kdcube:throttling:total_requests"
        TOTAL_THROTTLED_REQUESTS_KEY = "kdcube:throttling:total_throttled"
        RATE_LIMIT_429 = "kdcube:throttling:rate_limit_429"
        BACKPRESSURE_503 = "kdcube:throttling:backpressure_503"
        HOURLY = "kdcube:throttling:hourly"
        BY_REASON = "kdcube:throttling:by_reason"

    class CIRCUIT_BREAKER:
        """Circuit breaker Redis keys"""
        PREFIX = "kdcube:circuit_breaker"
        STATE_SUFFIX = "state"          # {PREFIX}:{name}:state
        STATS_SUFFIX = "stats"          # {PREFIX}:{name}:stats
        WINDOW_SUFFIX = "window"        # {PREFIX}:{name}:window
        HALF_OPEN_SUFFIX = "half_open_calls"  # {PREFIX}:{name}:half_open_calls

        # Global circuit breaker stats
        GLOBAL_STATS = "kdcube:circuit_breaker:global_stats"
        EVENTS_LOG = "kdcube:circuit_breaker:events"

    class SYSTEM:
        CAPACITY = "kdcube:system:capacity"
        RATE_LIMIT = "kdcube:system:ratelimit"

    class SYNCHRONIZATION:
        LOCK = "kdcube:lock"

    class DISCOVERY:
        REGISTRY = "kdcube:registry"

class CONFIG:
    ID_TOKEN_HEADER_NAME = os.getenv("ID_TOKEN_HEADER_NAME", "X-ID-Token")

    class BUNDLES:
        BUNDLE_MAPPING_KEY_FMT = "kdcube:config:bundles:mapping:{tenant}:{project}"
        UPDATE_CHANNEL = "kdcube:config:bundles:update"

    class AGENTIC:
        DEFAULT_LLM_MODEL_CONFIG = MODEL_CONFIGS.get(os.getenv("DEFAULT_LLM_MODEL_ID"), "gpt-4o-mini")
        DEFAULT_EMBEDDING_MODEL_CONFIG = EMBEDDERS.get(os.getenv("DEFAULT_EMBEDDING_MODEL_ID"), "openai-text-embedding-3-small")

        SUGGESTIONS_PREFIX = "kdcube:agentic:suggestions:{tenant}:{project}:{bundle_id}"


