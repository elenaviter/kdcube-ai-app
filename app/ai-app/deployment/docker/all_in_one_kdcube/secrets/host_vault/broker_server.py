# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
"""Deployment-local ``kdcube-secrets`` broker over the host vault.

Same HTTP shape as ``secrets/secrets_server.py`` (``/health``,
``GET /secret/{key}``, ``POST /set``, ``DELETE /secret/{key}``) so the
existing ``SecretsServiceSecretsManager`` client keeps working unchanged
once the provider is switched. Behind that shape there is no store: every
call is one mTLS request to the host vault with the deployment identity, and
a mutation returns ``ok`` only when the vault committed it.

The caller-side headers of the old service (``X-KDCUBE-SECRET-TOKEN``,
``X-KDCUBE-ADMIN-TOKEN``) keep their role as an in-deployment door gate
when configured, and NOTHING more: they are never forwarded, and the vault
never sees them. Workload identity toward the vault is the certificate only.

Environment:
  KDCUBE_HOST_VAULT_ADDR         host:port of the vault
  KDCUBE_HOST_VAULT_SERVER_NAME  name the vault's server certificate carries
                                 (default: the host part of ADDR)
  KDCUBE_HOST_VAULT_IDENTITY_DIR appliance identity mount with
                                 host-vault-client.key|.crt, host-vault-ca.crt
                                 (default /run/kdcube-host-vault-identity)
  KDCUBE_SECRETS_TENANT / KDCUBE_SECRETS_PROJECT
                                 the deployment's canonical namespace
  SECRETS_ADMIN_TOKEN / SECRETS_READ_TOKENS / SECRETS_PORT   as before
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from kdcube_ai_app.infra.secrets.host_vault.broker import SecretsBroker
from kdcube_ai_app.infra.secrets.host_vault.protocol import ErrorCode
from kdcube_ai_app.infra.secrets.host_vault.transport import ClientTLS, HostVaultClient

logging.basicConfig(
    level=os.getenv("SECRETS_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
)
LOGGER = logging.getLogger("kdcube.secrets.broker")

ADMIN_TOKEN = os.getenv("SECRETS_ADMIN_TOKEN")
READ_TOKENS = {
    t.strip() for t in os.getenv("SECRETS_READ_TOKENS", "").split(",") if t.strip()
}
APPLICATION = "kdcube-runtime"

_STATUS = {
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.FORBIDDEN: 403,
    ErrorCode.UNAUTHENTICATED: 503,
    ErrorCode.CONFLICT: 409,
    ErrorCode.TOO_LARGE: 413,
    ErrorCode.INVALID_REQUEST: 400,
}


def _vault_endpoint() -> tuple[str, int]:
    address = os.environ["KDCUBE_HOST_VAULT_ADDR"].strip()
    if "://" in address:
        raise RuntimeError(
            "KDCUBE_HOST_VAULT_ADDR must be host:port without a URL scheme"
        )
    try:
        parsed = urlsplit(f"//{address}")
        port = parsed.port
    except ValueError as exc:
        raise RuntimeError("KDCUBE_HOST_VAULT_ADDR has an invalid port") from exc
    if not parsed.hostname or port is None or parsed.username or parsed.password:
        raise RuntimeError("KDCUBE_HOST_VAULT_ADDR must be host:port")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise RuntimeError("KDCUBE_HOST_VAULT_ADDR must not contain a path")
    return parsed.hostname, port


def _require_identity(identity: Path) -> None:
    for filename in (
        "host-vault-client.crt",
        "host-vault-client.key",
        "host-vault-ca.crt",
    ):
        path = identity / filename
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(
                f"host-vault deployment identity is incomplete: {filename} is missing"
            )


def build_broker() -> SecretsBroker:
    host, port = _vault_endpoint()
    identity = Path(
        os.getenv("KDCUBE_HOST_VAULT_IDENTITY_DIR", "/run/kdcube-host-vault-identity")
    )
    _require_identity(identity)
    client = HostVaultClient(
        host=host,
        port=port,
        tls=ClientTLS(
            identity / "host-vault-client.crt",
            identity / "host-vault-client.key",
            identity / "host-vault-ca.crt",
        ),
        server_hostname=os.getenv("KDCUBE_HOST_VAULT_SERVER_NAME") or host,
    )
    tenant = os.environ["KDCUBE_SECRETS_TENANT"]
    project = os.environ["KDCUBE_SECRETS_PROJECT"]
    return SecretsBroker(transport=client, tenant=tenant, project=project)


app = FastAPI()
BROKER = build_broker()


class SecretItem(BaseModel):
    key: str
    value: str
    expected_generation: int | None = None


class SecretVerification(BaseModel):
    key: str
    sha256: str


def _require_admin(token: Optional[str]) -> None:
    if ADMIN_TOKEN and token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="admin token required")


def _require_read(token: Optional[str]) -> None:
    if READ_TOKENS and token not in READ_TOKENS:
        raise HTTPException(status_code=403, detail="read token required")


def _fail(code: ErrorCode) -> HTTPException:
    return HTTPException(status_code=_STATUS.get(code, 503), detail=code.value)


@app.get("/health")
def health() -> dict[str, Any]:
    vault = BROKER.health()
    if not vault["ok"]:
        raise HTTPException(
            status_code=503, detail=str(vault.get("code") or "backend_unavailable")
        )
    return {"status": "ok", "vault": vault}


@app.get("/secret/{key}")
def get_secret(
    key: str, x_kdcube_secret_token: Optional[str] = Header(default=None)
) -> dict[str, str]:
    _require_read(x_kdcube_secret_token)
    value = BROKER.get(application=APPLICATION, key=key)
    if value is None:
        raise HTTPException(status_code=404, detail="secret not found")
    return {"value": value}


@app.post("/set")
def set_secret(
    item: SecretItem, x_kdcube_admin_token: Optional[str] = Header(default=None)
) -> dict[str, Any]:
    _require_admin(x_kdcube_admin_token)
    result = BROKER.set(
        application=APPLICATION,
        key=item.key,
        value=item.value,
        expected_generation=item.expected_generation,
    )
    if not result.ok:
        raise _fail(result.code)
    return {"status": "ok", "generation": result.generation}


@app.post("/verify")
def verify_secret(
    item: SecretVerification,
    x_kdcube_admin_token: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    _require_admin(x_kdcube_admin_token)
    expected = item.sha256.strip().lower()
    if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
        raise HTTPException(status_code=400, detail="invalid sha256")
    result = BROKER.read(application=APPLICATION, key=item.key)
    if not result.ok:
        if result.code is ErrorCode.NOT_FOUND:
            return {"status": "ok", "state": "missing"}
        raise _fail(result.code)
    actual = hashlib.sha256((result.value or "").encode("utf-8")).hexdigest()
    state = "match" if hmac.compare_digest(actual, expected) else "different"
    return {"status": "ok", "state": state, "generation": result.generation}


@app.delete("/secret/{key}")
def delete_secret(
    key: str, x_kdcube_admin_token: Optional[str] = Header(default=None)
) -> dict[str, Any]:
    _require_admin(x_kdcube_admin_token)
    result = BROKER.delete(application=APPLICATION, key=key)
    if not result.ok:
        raise _fail(result.code)
    return {"status": "ok", "deleted": result.code is ErrorCode.OK}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("SECRETS_PORT", "7777")))
