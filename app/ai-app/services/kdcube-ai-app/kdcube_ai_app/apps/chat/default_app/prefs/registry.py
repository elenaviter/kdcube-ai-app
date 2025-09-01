# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

from __future__ import annotations
from typing import Dict

from kdcube_ai_app.apps.chat.sdk.context.policy.policy import KeyPolicy

REGISTRY: Dict[str, KeyPolicy] = {
    "budget.": KeyPolicy(min_support=1, avg_decayed=0.6, distinct_days=1, conflict_horizon_days=60, numeric_tolerance=0.02),
    "suggest.": KeyPolicy(min_support=2, avg_decayed=0.7, distinct_days=2),
    "focus.": KeyPolicy(min_support=2, avg_decayed=0.7, distinct_days=2),
    "watering.plan": KeyPolicy(min_support=2, avg_decayed=0.7, distinct_days=2, numeric_tolerance=0.0),
    # potential private keys you don't want to send back to LLM prompts:
    "location.garden_zone": KeyPolicy(send_to_llm=False),
}

def policy_for_key(key: str) -> KeyPolicy:
    best = None
    for prefix, pol in REGISTRY.items():
        if key.startswith(prefix) and (best is None or len(prefix) > len(best[0])):
            best = (prefix, pol)
    return best[1] if best else KeyPolicy()
