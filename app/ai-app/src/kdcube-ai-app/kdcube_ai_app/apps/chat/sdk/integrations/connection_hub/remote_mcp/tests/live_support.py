# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Shared owner and HTTP helpers for opt-in Connection Hub live tests."""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import string
import struct
import time
from typing import Any, Mapping

import boto3
import httpx
from pycognito.aws_srp import AWSSRP


def _totp(secret: str) -> str:
    padded = secret.upper() + "=" * ((8 - len(secret) % 8) % 8)
    key = base64.b32decode(padded)
    counter = int(time.time()) // 30
    digest = hmac.new(key, struct.pack(">Q", counter), hashlib.sha1).digest()
    offset = digest[-1] & 15
    code = struct.unpack(">I", digest[offset : offset + 4])[0] & 0x7FFFFFFF
    return f"{code % 1_000_000:06d}"


class DisposableOwner:
    """One temporary Cognito owner whose credentials never leave this process."""

    def __init__(
        self,
        *,
        region: str,
        pool_id: str,
        client_id: str,
        label: str,
    ) -> None:
        self.region = region
        self.pool_id = pool_id
        self.client_id = client_id
        self.label = label
        self.username = f"ch-live-{int(time.time())}-{secrets.token_hex(3)}"
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        self.password = "Aa9!" + "".join(
            secrets.choice(alphabet) for _ in range(24)
        )
        self.client = boto3.client("cognito-idp", region_name=region)
        self.created = False

    def authenticate(self) -> dict[str, str]:
        self.client.admin_create_user(
            UserPoolId=self.pool_id,
            Username=self.username,
            UserAttributes=[
                {"Name": "email", "Value": f"{self.username}@example.invalid"},
                {"Name": "email_verified", "Value": "true"},
            ],
            MessageAction="SUPPRESS",
        )
        self.created = True
        self.client.admin_set_user_password(
            UserPoolId=self.pool_id,
            Username=self.username,
            Password=self.password,
            Permanent=True,
        )
        first = AWSSRP(
            username=self.username,
            password=self.password,
            pool_id=self.pool_id,
            client_id=self.client_id,
            client=self.client,
        ).authenticate_user()
        if first.get("ChallengeName") != "MFA_SETUP":
            raise RuntimeError(
                "temporary user received unexpected challenge: "
                f"{first.get('ChallengeName')}"
            )
        associated = self.client.associate_software_token(Session=first["Session"])
        verified = self.client.verify_software_token(
            Session=associated["Session"],
            UserCode=_totp(associated["SecretCode"]),
            FriendlyDeviceName=self.label,
        )
        completed = self.client.respond_to_auth_challenge(
            ClientId=self.client_id,
            ChallengeName="MFA_SETUP",
            Session=verified["Session"],
            ChallengeResponses={"USERNAME": self.username},
        )
        tokens = completed.get("AuthenticationResult") or {}
        access_token = str(tokens.get("AccessToken") or "")
        id_token = str(tokens.get("IdToken") or "")
        if not access_token or not id_token:
            raise RuntimeError("temporary user authentication returned no tokens")
        return {
            "Authorization": f"Bearer {access_token}",
            "X-ID-Token": id_token,
            "Accept": "application/json",
        }

    def delete(self) -> None:
        if not self.created:
            return
        self.client.admin_delete_user(
            UserPoolId=self.pool_id,
            Username=self.username,
        )
        self.created = False


class OwnerOperations:
    """Authenticated owner calls against one hosted Connection Hub app."""

    def __init__(
        self,
        *,
        base_url: str,
        tenant: str,
        project: str,
        bundle_id: str,
        headers: Mapping[str, str],
    ) -> None:
        self.base = (
            f"{base_url.rstrip('/')}/api/integrations/bundles/"
            f"{tenant}/{project}/{bundle_id}"
        )
        self.client = httpx.Client(headers=dict(headers), timeout=60)

    @property
    def proxy_url(self) -> str:
        return f"{self.base}/public/mcp/remote_mcp_proxy"

    @property
    def admission_url(self) -> str:
        return f"{self.base}/public/delegated_admission"

    def close(self) -> None:
        self.client.close()

    def wait_ready(self, *, timeout_seconds: float = 60) -> None:
        """Wait until the hosted Connection Hub bundle serves owner reads."""

        deadline = time.monotonic() + max(1.0, float(timeout_seconds))
        last_error = "application_not_ready"
        while time.monotonic() < deadline:
            try:
                result = self.call("GET", "delegated_access_list")
                if result.get("ok") is True:
                    return
                last_error = str(result.get("error") or "not_ready")
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in {409, 503}:
                    raise
                try:
                    body = exc.response.json()
                    last_error = str(body.get("type") or body.get("error") or "not_ready")
                except ValueError:
                    last_error = f"http_{exc.response.status_code}"
            time.sleep(0.25)
        raise RuntimeError(
            f"Connection Hub did not become ready: {last_error}"
        )

    def call(
        self, method: str, operation: str, payload: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        response = self.client.request(
            method,
            f"{self.base}/operations/{operation}",
            json={"data": dict(payload or {})} if method == "POST" else None,
        )
        response.raise_for_status()
        body = response.json()
        result = body.get(operation, body)
        if not isinstance(result, Mapping):
            raise RuntimeError(f"{operation} returned a non-object result")
        return dict(result)


__all__ = ["DisposableOwner", "OwnerOperations"]
