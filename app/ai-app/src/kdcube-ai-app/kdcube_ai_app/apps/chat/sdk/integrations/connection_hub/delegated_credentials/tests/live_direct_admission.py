# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Live acceptance for a registered external protected service.

This opt-in deployment test starts the App Ecosystem direct-admission example,
creates a disposable owner and delegated caller, and exercises demand-driven
operation consent, one-use authority, replay, repeated authority, and revoke.
The local runtime must already contain the matching catalog, service row, and
secret references. No bearer or secret value is printed.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import httpx
import yaml

from kdcube_ai_app.apps.chat.sdk.integrations.connection_hub.remote_mcp.tests.live_support import (
    DisposableOwner,
    OwnerOperations,
)

RESOURCE = "https://reference.example.test/customers"
OPERATION = "customers.search"
SERVICE_ID = "reference-customers-api"
GRANT = "reference_customers:read"
SERVICE_SECRET_PATH = (
    "connections",
    "delegated_credentials",
    "admission",
    "services",
    SERVICE_ID,
    "signing_secret",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:5173")
    parser.add_argument("--tenant", default="demo-tenant")
    parser.add_argument("--project", default="demo-project")
    parser.add_argument("--bundle-id", default="connection-hub@1-0")
    parser.add_argument("--cognito-region", default="eu-west-1")
    parser.add_argument("--cognito-pool", default="eu-west-1_JrKKhQUNp")
    parser.add_argument(
        "--cognito-client", default="6lgsqqbpatprt44a4i20hveu6u"
    )
    parser.add_argument("--fixture-port", type=int, default=8766)
    parser.add_argument("--app-ecosystem-root", type=Path, required=True)
    parser.add_argument("--runtime-workdir", type=Path, default=None)
    return parser.parse_args()


def _runtime_workdir(args: argparse.Namespace) -> Path:
    if args.runtime_workdir is not None:
        return args.runtime_workdir.expanduser().resolve()
    return (
        Path.home()
        / ".kdcube"
        / "kdcube-runtime"
        / f"{args.tenant}__{args.project}"
    )


def _service_secret(args: argparse.Namespace) -> str:
    descriptor = _runtime_workdir(args) / "config" / "bundles.secrets.yaml"
    payload = yaml.safe_load(descriptor.read_text(encoding="utf-8")) or {}
    items = ((payload.get("bundles") or {}).get("items") or [])
    bundle = next(
        (row for row in items if str(row.get("id") or "") == args.bundle_id),
        None,
    )
    if not isinstance(bundle, Mapping):
        raise RuntimeError(f"bundle secrets missing for {args.bundle_id}")
    value: Any = bundle.get("secrets") or {}
    for part in SERVICE_SECRET_PATH:
        value = value.get(part) if isinstance(value, Mapping) else None
    secret = str(value or "")
    if len(secret.encode("utf-8")) < 32:
        raise RuntimeError("registered direct-admission service secret is unavailable")
    return secret


class DirectServiceFixture:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.process: subprocess.Popen[bytes] | None = None

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.args.fixture_port}/customers/search"

    def start(self, *, admission_url: str, service_secret: str) -> None:
        root = self.args.app_ecosystem_root.expanduser().resolve()
        example_src = root / "examples/connection-hub/direct-admission-service/src"
        package_src = (
            root
            / "products/connection-hub/packages/connection-hub/src"
        )
        if not example_src.is_dir() or not package_src.is_dir():
            raise RuntimeError(f"App Ecosystem source root is invalid: {root}")
        env = os.environ.copy()
        env.update(
            {
                "CONNECTION_HUB_ADMISSION_URL": admission_url,
                "CONNECTION_HUB_SERVICE_ID": SERVICE_ID,
                "CONNECTION_HUB_SERVICE_SECRET": service_secret,
                "CONNECTION_HUB_RESOURCE": RESOURCE,
                "PYTHONPATH": os.pathsep.join(
                    [
                        str(example_src),
                        str(package_src),
                        env.get("PYTHONPATH", ""),
                    ]
                ).rstrip(os.pathsep),
            }
        )
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "reference_service.app:create_app",
                "--factory",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.args.fixture_port),
                "--log-level",
                "warning",
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        health_url = f"http://127.0.0.1:{self.args.fixture_port}/health"
        for _ in range(60):
            if self.process.poll() is not None:
                raise RuntimeError("direct-admission fixture exited during startup")
            try:
                response = httpx.get(health_url, timeout=1)
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
        raise RuntimeError("direct-admission fixture did not become healthy")

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process = None


def _call_service(
    fixture: DirectServiceFixture,
    *,
    bearer: str,
    invocation_id: str,
    query: str,
) -> tuple[httpx.Response, dict[str, Any]]:
    response = httpx.post(
        fixture.endpoint,
        headers={
            "Authorization": f"Bearer {bearer}",
            "Idempotency-Key": invocation_id,
        },
        json={"query": query},
        timeout=30,
    )
    body = response.json()
    if not isinstance(body, Mapping):
        raise RuntimeError("direct-admission fixture returned a non-object")
    return response, dict(body)


def _run(args: argparse.Namespace) -> None:
    owner_identity = DisposableOwner(
        region=args.cognito_region,
        pool_id=args.cognito_pool,
        client_id=args.cognito_client,
        label="Connection Hub direct admission live acceptance",
    )
    owner: OwnerOperations | None = None
    fixture = DirectServiceFixture(args)
    access_id = ""

    try:
        owner = OwnerOperations(
            base_url=args.base_url,
            tenant=args.tenant,
            project=args.project,
            bundle_id=args.bundle_id,
            headers=owner_identity.authenticate(),
        )
        owner.wait_ready()
        fixture.start(
            admission_url=owner.admission_url,
            service_secret=_service_secret(args),
        )

        unauthenticated = httpx.post(
            owner.admission_url,
            headers={
                "Authorization": "Bearer invalid-delegated-bearer",
                "X-Connection-Hub-Service-Id": SERVICE_ID,
                "X-Connection-Hub-Timestamp": str(int(time.time())),
                "X-Connection-Hub-Nonce": "invalid-proof-live-acceptance",
                "X-Connection-Hub-Signature": "invalid",
            },
            json={"resource": RESOURCE, "operation": OPERATION},
            timeout=30,
        )
        assert unauthenticated.status_code == 401, unauthenticated.text
        assert unauthenticated.json()["error"]["code"] == (
            "service_authentication_failed"
        )
        print("PASS unregistered workload proof is denied before caller authority")

        card = owner.call(
            "POST",
            "delegated_access_create",
            {
                "label": "Direct admission live caller",
                "resource_grants": {RESOURCE: [GRANT]},
                "resource_operations": {RESOURCE: []},
                "ttl_seconds": 600,
            },
        )
        assert card.get("ok") is True, card
        delegated_token = str(card.get("access_token") or "")
        access_id = str(card["access"]["access_id"])
        client_id = str(card["access"]["client_id"])
        assert delegated_token and access_id and client_id
        print("PASS direct caller card created without operation authority")

        invocation_id = "direct-search-once"
        denied_response, denial = _call_service(
            fixture,
            bearer=delegated_token,
            invocation_id=invocation_id,
            query="north",
        )
        assert denied_response.status_code == 403, denial
        assert denial["error"]["code"] == "operation_not_consented", denial
        consent = dict(denial.get("consent") or {})
        assert consent.get("agent_client_id") == client_id
        assert consent.get("access_id") == access_id
        assert consent.get("resource") == RESOURCE
        assert consent.get("outer_operation") == OPERATION
        assert consent.get("invocation_change_id") == invocation_id
        assert consent.get("available_choices") == [
            "allow_once",
            "allow_always",
        ]
        grant = dict(consent.get("grant") or {})
        assert grant.get("operation") == "delegated_agent_grant_create"
        grant_payload = dict(grant.get("payload") or {})
        print("PASS direct denial names the exact operation and recovery choices")

        once_grant = owner.call(
            "POST",
            "delegated_agent_grant_create",
            {**grant_payload, "invocation_mode": "once"},
        )
        assert once_grant.get("ok") is True, once_grant
        once_policy = dict(once_grant.get("invocation_policy") or {})
        assert once_policy.get("mode") == "once"
        assert once_policy.get("remaining") == 1

        first_response, first = _call_service(
            fixture,
            bearer=delegated_token,
            invocation_id=invocation_id,
            query="north",
        )
        assert first_response.status_code == 200, first
        assert first.get("admission_replay") is False
        assert first.get("invocation_policy", {}).get("remaining") == 0
        assert [row["id"] for row in first.get("customers") or []] == [
            "customer-101"
        ]
        assert str(first.get("principal", {}).get("sub") or "").startswith(
            "prk_sub_"
        )
        assert owner_identity.username not in str(first)
        assert access_id not in str(first), first

        replay_response, replay = _call_service(
            fixture,
            bearer=delegated_token,
            invocation_id=invocation_id,
            query="north",
        )
        assert replay_response.status_code == 200, replay
        assert replay.get("admission_replay") is True

        exhausted_response, exhausted = _call_service(
            fixture,
            bearer=delegated_token,
            invocation_id="direct-search-next",
            query="contoso",
        )
        assert exhausted_response.status_code == 403, exhausted
        assert exhausted["error"]["code"] == (
            "delegated_invocation_limit_exhausted"
        )
        assert exhausted["ret"]["details"]["available_choices"] == [
            "allow_once",
            "allow_always",
        ]
        print("PASS direct allow once, admission replay, and exhaustion")

        always = owner.call(
            "POST",
            "delegated_invocation_policy_set",
            {
                "access_id": access_id,
                "resource": RESOURCE,
                "operation": OPERATION,
                "mode": "always",
                "expected_revision": once_policy["revision"],
            },
        )
        assert always.get("ok") is True, always
        assert always.get("policy", {}).get("mode") == "always"
        repeated_response, repeated = _call_service(
            fixture,
            bearer=delegated_token,
            invocation_id="direct-search-always",
            query="contoso",
        )
        assert repeated_response.status_code == 200, repeated
        assert [row["id"] for row in repeated.get("customers") or []] == [
            "customer-102"
        ]
        print("PASS direct allow always restores repeated authority")

        revoked = owner.call(
            "POST", "delegated_access_revoke", {"access_id": access_id}
        )
        assert revoked.get("ok") is True, revoked
        access_id = ""
        revoked_response, revoked_body = _call_service(
            fixture,
            bearer=delegated_token,
            invocation_id="direct-search-revoked",
            query="north",
        )
        assert revoked_response.status_code in {401, 403, 503}, revoked_body
        assert revoked_body.get("allowed") is not True
        print("PASS direct caller revoke is enforced on the next admission")
    finally:
        if owner is not None:
            if access_id:
                try:
                    owner.call(
                        "POST", "delegated_access_revoke", {"access_id": access_id}
                    )
                except Exception:
                    pass
            owner.close()
        owner_identity.delete()
        fixture.stop()
        print("PASS disposable owner and direct fixture cleanup")


def main() -> None:
    _run(_arguments())


if __name__ == "__main__":
    main()
