# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# sdk/rate_limit/policy.py
from dataclasses import dataclass
from typing import Optional, Dict

"""
User quota policy (track tokens/requests).
"""

@dataclass(frozen=True)
class QuotaPolicy:
    # concurrency (per subject)
    max_concurrent: int = 1
    # request quotas
    requests_per_day: Optional[int] = None
    requests_per_month: Optional[int] = None
    total_requests: Optional[int] = None
    # token quotas (post-paid check; enforced against *previous* committed turns at admit time)
    tokens_per_hour: Optional[int] = None
    tokens_per_day: Optional[int] = None
    tokens_per_month: Optional[int] = None

@dataclass
class PolicyTable:
    by_user_type: Dict[str, QuotaPolicy]
    default: Optional[QuotaPolicy] = None

    def for_user_type(self, user_type: Optional) -> Optional[QuotaPolicy]:
        if user_type and user_type in self.by_user_type:
            return self.by_user_type[user_type]
        return self.default

"""
Application-level budget policies for providers.
Track USD spending per provider across all users.
"""

@dataclass(frozen=True)
class ProviderBudgetPolicy:
    """
    Budget limits for a single provider (anthropic, openai, brave, etc.).

    All limits are in USD.
    None = unlimited for that dimension.
    """
    provider: str

    # Spending limits in USD
    usd_per_hour: Optional[float] = None
    usd_per_day: Optional[float] = None
    usd_per_month: Optional[float] = None

    def __post_init__(self):
        # Validate that at least one limit is set
        if all(x is None for x in [self.usd_per_hour, self.usd_per_day, self.usd_per_month]):
            # All None = unlimited, which is valid
            pass

        # Validate positive values
        for attr in ['usd_per_hour', 'usd_per_day', 'usd_per_month']:
            val = getattr(self, attr)
            if val is not None and val < 0:
                raise ValueError(f"{attr} must be non-negative, got {val}")


@dataclass(frozen=True)
class BudgetInsight:
    """
    Application-level budget status for a provider.

    Similar to QuotaInsight but for USD spending across all users.
    """
    provider: str
    limits: dict[str, Optional[float]]      # {usd_per_hour, usd_per_day, usd_per_month}
    remaining: dict[str, Optional[float]]   # remaining USD in each window
    violations: list[str]                    # e.g., ["usd_per_day", "usd_per_month"]
    retry_after_sec: Optional[int]
    retry_scope: Optional[str]               # "hour" | "day" | "month" | None