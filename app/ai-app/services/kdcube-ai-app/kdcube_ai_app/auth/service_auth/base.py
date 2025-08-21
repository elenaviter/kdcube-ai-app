# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# auth/service_auth/base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Optional, Dict, Any
import time
import jwt  # pyjwt, for exp checks only (no verification here)

@dataclass
class TokenBundle:
    access_token: str
    id_token: str
    refresh_token: Optional[str] = None
    token_type: str = "Bearer"
    # Seconds since epoch; if None we’ll infer from JWT exp
    access_expires_at: Optional[int] = None
    id_expires_at: Optional[int] = None

    def _infer_exp(self, token: str) -> Optional[int]:
        try:
            # decode without verify to read exp
            payload = jwt.decode(token, options={"verify_signature": False, "verify_exp": False})
            return int(payload.get("exp")) if payload.get("exp") else None
        except Exception:
            return None

    def ensure_exp_fields(self) -> None:
        if self.access_token and not self.access_expires_at:
            self.access_expires_at = self._infer_exp(self.access_token)
        if self.id_token and not self.id_expires_at:
            self.id_expires_at = self._infer_exp(self.id_token)

    def is_access_expired(self, skew_sec: int = 60) -> bool:
        self.ensure_exp_fields()
        if not self.access_expires_at:
            # if unknown, be conservative after 10 minutes
            return False
        return time.time() >= (self.access_expires_at - skew_sec)


class ServiceIdP(Protocol):
    """Provider-agnostic service auth client."""
    def authenticate(self) -> TokenBundle:
        """Perform initial sign-in and return tokens."""
        ...

    def refresh(self, tokens: TokenBundle) -> TokenBundle:
        """Refresh tokens. Returns a new bundle."""
        ...

    def close(self) -> None:
        """Cleanup resources, if any."""
        ...

class IdpConfig:
    """Lightweight config bucket for any IdP."""
    def __init__(self, provider: str, **kwargs: Any) -> None:
        self.provider = provider.lower().strip()
        self.kwargs = kwargs

def build_auth_headers(tokens: "TokenBundle", *, id_header_name: str, on_behalf_session_id: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {tokens.access_token}",
        id_header_name: tokens.id_token,
        "User-Session-ID": on_behalf_session_id,
    }
