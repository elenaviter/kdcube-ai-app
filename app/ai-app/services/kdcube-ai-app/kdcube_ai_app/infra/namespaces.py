# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# infra/namespaces.py
import os

from kdcube_ai_app.apps.chat.reg import MODEL_CONFIGS, EMBEDDERS

def tp_prefix(tenant: str | None = None, project: str | None = None) -> str:
    from kdcube_ai_app.apps.chat.sdk.config import get_settings
    s = get_settings()
    t = tenant or s.TENANT
    p = project or s.PROJECT
    return f"{t}:{p}"

def ns_key(base: str, *, tenant: str | None = None, project: str | None = None) -> str:
    return f"{tp_prefix(tenant, project)}:{base}"

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

    class RATE_LIMIT:
        """
        Bundle-level user rate limiting (tokens/requests per user).

        Keys are NOT prefixed with tenant:project since bundles may be
        cross-tenant (or we may add tenant/project prefixing later).

        Format: kdcube:rl:{bundle}:{subject}:*
        where subject = {tenant}:{project}:{user_id} or {tenant}:{project}:{user_id}:{session_id}
        """
        PREFIX = "kdcube:rl"

        # Key patterns (bundle and subject will be interpolated)
        LOCKS = "{prefix}:{bundle}:{subject}:locks"
        REQUESTS_DAY = "{prefix}:{bundle}:{subject}:reqs:day:{ymd}"
        REQUESTS_MONTH = "{prefix}:{bundle}:{subject}:reqs:month:{ym}"
        REQUESTS_TOTAL = "{prefix}:{bundle}:{subject}:reqs:total"
        TOKENS_HOUR = "{prefix}:{bundle}:{subject}:toks:hour:{ymdh}"
        TOKENS_DAY = "{prefix}:{bundle}:{subject}:toks:day:{ymd}"
        TOKENS_MONTH = "{prefix}:{bundle}:{subject}:toks:month:{ym}"
        LAST_TURN_TOKENS = "{prefix}:{bundle}:{subject}:last_turn_tokens"
        LAST_TURN_AT = "{prefix}:{bundle}:{subject}:last_turn_at"


    class BUDGET:
        """
        Application-level budget tracking (USD spending per provider).

        Keys ARE prefixed with tenant:project since budgets are per-tenant/project.

        Format: {tenant}:{project}:kdcube:budget:{bundle}:{provider}:*
        """
        PREFIX = "kdcube:budget"

        # Key patterns (will be prefixed with tenant:project via ns_key)
        SPEND_HOUR = "{prefix}:{bundle}:{provider}:spend:hour:{ymdh}"
        SPEND_DAY = "{prefix}:{bundle}:{provider}:spend:day:{ymd}"
        SPEND_MONTH = "{prefix}:{bundle}:{provider}:spend:month:{ym}"
        LAST_SPEND_USD = "{prefix}:{bundle}:{provider}:last_spend_usd"
        LAST_SPEND_AT = "{prefix}:{bundle}:{provider}:last_spend_at"

    class SYSTEM:
        CAPACITY = "kdcube:system:capacity"
        RATE_LIMIT = "kdcube:system:ratelimit"

    class SYNCHRONIZATION:
        LOCK = "kdcube:lock"

    class DISCOVERY:
        REGISTRY = "kdcube:registry"

class CONFIG:
    ID_TOKEN_HEADER_NAME = os.getenv("ID_TOKEN_HEADER_NAME", "X-ID-Token")
    USER_TIMEZONE_HEADER_NAME = os.getenv("USER_TIMEZONE_HEADER_NAME", "X-User-Timezone")
    USER_UTC_OFFSET_MIN_HEADER_NAME = os.getenv("USER_UTC_OFFSET_MIN_HEADER_NAME", "X-User-UTC-Offset")

    class BUNDLES:
        BUNDLE_MAPPING_KEY_FMT = "kdcube:config:bundles:mapping:{tenant}:{project}"
        UPDATE_CHANNEL = "kdcube:config:bundles:update"

    class AGENTIC:
        DEFAULT_LLM_MODEL_CONFIG = MODEL_CONFIGS.get(os.getenv("DEFAULT_LLM_MODEL_ID"), "gpt-4o-mini")
        DEFAULT_EMBEDDING_MODEL_CONFIG = EMBEDDERS.get(os.getenv("DEFAULT_EMBEDDING_MODEL_ID"), "openai-text-embedding-3-small")

        SUGGESTIONS_PREFIX = "kdcube:agentic:suggestions:{tenant}:{project}:{bundle_id}"

    KDCUBE_STORAGE_PATH = os.environ.get("KDCUBE_STORAGE_PATH") or os.environ.get("STORAGE_PATH")


