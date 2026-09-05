"""PKCE values for KDCube browser management ceremonies."""

from __future__ import annotations

import base64
import hashlib
import secrets
from dataclasses import dataclass


@dataclass(frozen=True)
class PKCEParameters:
    code_verifier: str
    code_challenge: str
    state: str


def _base64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def code_challenge(code_verifier: str) -> str:
    return _base64url(hashlib.sha256(code_verifier.encode("ascii")).digest())


def generate_pkce() -> PKCEParameters:
    verifier = secrets.token_urlsafe(64)
    return PKCEParameters(
        code_verifier=verifier,
        code_challenge=code_challenge(verifier),
        state=secrets.token_urlsafe(32),
    )
