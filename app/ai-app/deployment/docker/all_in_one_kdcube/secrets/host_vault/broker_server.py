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
  KDCUBE_HOST_VAULT_ADDR         host:port of the vault (default 127.0.0.1:7781)
  KDCUBE_HOST_VAULT_SERVER_NAME  name the vault's server certificate carries
                                 (default: the host part of ADDR)
  KDCUBE_HOST_VAULT_IDENTITY_DIR appliance identity mount with
                                 host-vault-client.key|.crt, host-vault-ca.crt
                                 (default /run/kdcube-host-vault-identity)
  KDCUBE_SECRETS_TENANT / KDCUBE_SECRETS_PROJECT
                                 the deployment's canonical namespace
  KDCUBE_SECRETS_APPLICATION     trusted logical application bound to this
                                 broker (default connection-hub@1-0)
  SECRETS_ADMIN_TOKEN / SECRETS_READ_TOKENS / SECRETS_PORT   as before
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from kdcube_ai_app.infra.secrets.host_vault.broker import SecretsBroker
from kdcube_ai_app.infra.secrets.host_vault.protocol import ErrorCode
from kdcube_ai_app.infra.secrets.host_vault.transport import ClientTLS, HostVaultClient

logging.basicConfig(level=os.getenv("SECRETS_LOG_LEVEL", "INFO").upper(),
                    format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
LOGGER = logging.getLogger("kdcube.secrets.broker")

ADMIN_TOKEN = os.getenv("SECRETS_ADMIN_TOKEN")
READ_TOKENS = {t.strip() for t in os.getenv("SECRETS_READ_TOKENS", "").split(",") if t.strip()}
APPLICATION = os.getenv("KDCUBE_SECRETS_APPLICATION", "connection-hub@1-0")

_STATUS = {
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.FORBIDDEN: 403,
    ErrorCode.UNAUTHENTICATED: 503,
    ErrorCode.CONFLICT: 409,
    ErrorCode.TOO_LARGE: 413,
    ErrorCode.INVALID_REQUEST: 400,
}


def build_broker() -> SecretsBroker:
    host, _, port = os.getenv("KDCUBE_HOST_VAULT_ADDR", "127.0.0.1:7781").rpartition(":")
    identity = Path(os.getenv("KDCUBE_HOST_VAULT_IDENTITY_DIR", "/run/kdcube-host-vault-identity"))
    client = HostVaultClient(
        host=host, port=int(port),
        tls=ClientTLS(identity / "host-vault-client.crt", identity / "host-vault-client.key", identity / "host-vault-ca.crt"),
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
    return {"status": "ok" if vault["ok"] else "degraded", "vault": vault}


@app.get("/secret/{key}")
def get_secret(key: str, x_kdcube_secret_token: Optional[str] = Header(default=None)) -> dict[str, str]:
    _require_read(x_kdcube_secret_token)
    value = BROKER.get(application=APPLICATION, key=key)
    if value is None:
        raise HTTPException(status_code=404, detail="secret not found")
    return {"value": value}


@app.post("/set")
def set_secret(item: SecretItem, x_kdcube_admin_token: Optional[str] = Header(default=None)) -> dict[str, Any]:
    _require_admin(x_kdcube_admin_token)
    result = BROKER.set(application=APPLICATION, key=item.key, value=item.value)
    if not result.ok:
        raise _fail(result.code)
    return {"status": "ok", "generation": result.generation}


@app.delete("/secret/{key}")
def delete_secret(key: str, x_kdcube_admin_token: Optional[str] = Header(default=None)) -> dict[str, Any]:
    _require_admin(x_kdcube_admin_token)
    result = BROKER.delete(application=APPLICATION, key=key)
    if not result.ok:
        raise _fail(result.code)
    return {"status": "ok", "deleted": result.code is ErrorCode.OK}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("SECRETS_PORT", "7777")))
