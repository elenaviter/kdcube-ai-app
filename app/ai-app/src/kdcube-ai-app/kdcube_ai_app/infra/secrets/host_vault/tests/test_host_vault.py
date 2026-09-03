# SPDX-License-Identifier: MIT
"""Host vault first-phase proofs (the handoff's required-test list).

Every identity here is FAKE test material minted in memory by the same X.509
code the host CA uses; no real secret value, key, or deployment identity is
read or written. The root-key provider is the labeled in-memory fake."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from kdcube_ai_app.infra.secrets.host_vault import audit, broker, identity, keys, protocol, service, storage, transport
from kdcube_ai_app.infra.secrets.host_vault.protocol import ErrorCode, Operation, SecretNamespace, SecretReference, VaultError, VaultRequest

NS = SecretNamespace("demo-tenant", "demo-project", "connection-hub@1-0")
OTHER_APP = SecretNamespace("demo-tenant", "demo-project", "other-app@1-0")
OTHER_PROJECT = SecretNamespace("demo-tenant", "other-project", "connection-hub@1-0")
OTHER_TENANT = SecretNamespace("other-tenant", "demo-project", "connection-hub@1-0")
KEY = "users.u1.bundles.b1.secrets.token"
CANARY = "CANARY-secret-value-9f3a"
ENROLLED = object()  # Rig.call default: the rig's own enrolled certificate


# ── fixtures ──────────────────────────────────────────────────────────────


class Rig:
    """A host CA, trust registry, store, service, and one enrolled deployment."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.ca = identity.HostIssuingCA.generate()
        self.registry = identity.TrustRegistry(root / "trust.json", ca=self.ca)
        self.keys = keys.FakeInMemoryRootKeyProvider()
        self.store = storage.FileDurableSecretStore(root / "store", self.keys)
        self.audit = audit.MemoryAuditSink()
        self.service = service.HostVaultService(store=self.store, registry=self.registry, audit=self.audit)
        self.key = identity.DeploymentKey.generate()
        ticket = self.registry.mint_ticket(deployment_id="dep-1", namespaces=[NS.path])
        self.cert, self.record = self.registry.enroll(ticket_id=ticket.ticket_id, csr_pem=self.key.csr())

    def request(self, op: Operation, ns: SecretNamespace = NS, key: str = KEY, **kw) -> VaultRequest:
        ref = None if op is Operation.HEALTH else SecretReference(ns, key)
        return VaultRequest.new(op, ref, **kw)

    def call(self, request: VaultRequest, *, cert: Any = ENROLLED, now: float | None = None):
        peer = self.cert if cert is ENROLLED else cert
        return self.service.handle(request.to_wire(), peer_cert_pem=peer, now=now)


@pytest.fixture
def rig(tmp_path: Path) -> Rig:
    return Rig(tmp_path)


# 1. exact namespace authorization ─────────────────────────────────────────


def test_namespace_authorization_is_exact(rig: Rig):
    assert rig.call(rig.request(Operation.SET, value=CANARY)).ok
    assert rig.call(rig.request(Operation.GET)).value == CANARY
    for ns in (OTHER_APP, OTHER_PROJECT, OTHER_TENANT):
        response = rig.call(rig.request(Operation.GET, ns))
        assert response.ok is False and response.code is ErrorCode.FORBIDDEN, ns.path
        assert response.value is None
    # a wildcard ACL covers the project, still not another tenant
    ticket = rig.registry.mint_ticket(deployment_id="dep-wild", namespaces=["demo-tenant/demo-project/*"])
    key2 = identity.DeploymentKey.generate()
    cert2, _ = rig.registry.enroll(ticket_id=ticket.ticket_id, csr_pem=key2.csr())
    assert rig.call(rig.request(Operation.GET, OTHER_APP), cert=cert2).code is ErrorCode.NOT_FOUND
    assert rig.call(rig.request(Operation.GET, OTHER_TENANT), cert=cert2).code is ErrorCode.FORBIDDEN


# 2. certificate identity and live ACL ─────────────────────────────────────


def test_only_enrolled_certificates_identify_a_deployment(rig: Rig):
    stranger_ca = identity.HostIssuingCA.generate()
    stranger_key = identity.DeploymentKey.generate()
    stranger_cert = stranger_ca.issue(stranger_key.csr(), deployment_id="dep-1")  # same id, other CA
    response = rig.call(rig.request(Operation.HEALTH), cert=stranger_cert)
    assert response.code is ErrorCode.UNAUTHENTICATED
    assert rig.call(rig.request(Operation.HEALTH), cert=None).code is ErrorCode.UNAUTHENTICATED
    # the CSR's requested subject is not trusted: the host assigns the id
    ok = rig.call(rig.request(Operation.HEALTH))
    assert ok.ok and ok.extra["deployment_id"] == "dep-1"


def test_enrollment_ticket_is_one_use_and_expires(rig: Rig):
    ticket = rig.registry.mint_ticket(deployment_id="dep-2", namespaces=[NS.path])
    key = identity.DeploymentKey.generate()
    rig.registry.enroll(ticket_id=ticket.ticket_id, csr_pem=key.csr())
    with pytest.raises(VaultError) as exc:
        rig.registry.enroll(ticket_id=ticket.ticket_id, csr_pem=key.csr())
    assert exc.value.code is ErrorCode.UNAUTHENTICATED
    stale = rig.registry.mint_ticket(deployment_id="dep-3", namespaces=[NS.path], ttl_seconds=-1)
    with pytest.raises(VaultError):
        rig.registry.enroll(ticket_id=stale.ticket_id, csr_pem=key.csr())


# 3. revoked and expired identity denial ───────────────────────────────────


def test_revoked_and_expired_identities_are_denied(rig: Rig):
    rig.call(rig.request(Operation.SET, value=CANARY))
    expired_at = rig.record.not_after + 1
    assert rig.call(rig.request(Operation.GET), now=expired_at).code is ErrorCode.UNAUTHENTICATED
    rig.registry.revoke(rig.record.fingerprint)
    assert rig.call(rig.request(Operation.GET)).code is ErrorCode.UNAUTHENTICATED


def test_revocation_by_another_process_lands_on_the_next_identification(rig: Rig):
    operator_view = identity.TrustRegistry(rig.root / "trust.json", ca=rig.ca)
    server_view = identity.TrustRegistry(rig.root / "trust.json")  # identify-only: no CA key
    assert server_view.identify(rig.cert).deployment_id == "dep-1"
    time.sleep(0.01)
    operator_view.revoke(rig.record.fingerprint)
    with pytest.raises(VaultError) as exc:
        server_view.identify(rig.cert)
    assert exc.value.code is ErrorCode.UNAUTHENTICATED
    with pytest.raises(VaultError):  # an identify-only registry cannot issue
        server_view.mint_ticket(deployment_id="x", namespaces=[NS.path])
        server_view.enroll(ticket_id="nope", csr_pem=b"")


def test_rotation_overlaps_then_the_old_certificate_lapses(rig: Rig):
    new_key = identity.DeploymentKey.generate()
    new_cert, new_record = rig.registry.rotate(
        current_fingerprint=rig.record.fingerprint, csr_pem=new_key.csr(), overlap_seconds=60,
    )
    assert new_record.supersedes == rig.record.fingerprint
    assert rig.call(rig.request(Operation.HEALTH), cert=rig.cert).ok  # still inside overlap
    assert rig.call(rig.request(Operation.HEALTH), cert=new_cert).ok
    later = time.time() + 120
    assert rig.call(rig.request(Operation.HEALTH), cert=rig.cert, now=later).code is ErrorCode.UNAUTHENTICATED
    assert rig.call(rig.request(Operation.HEALTH), cert=new_cert, now=later).ok
    # the registry survives a restart with the same trust decisions
    reloaded = identity.TrustRegistry(rig.root / "trust.json", ca=rig.ca)
    assert reloaded.identify(new_cert).deployment_id == "dep-1"


# 4. replay / idempotency ──────────────────────────────────────────────────


def test_replayed_mutation_returns_the_original_result_without_a_second_commit(rig: Rig):
    request = rig.request(Operation.SET, value=CANARY)
    first = rig.call(request)
    again = rig.call(request)
    assert first.ok and again.ok and first.generation == again.generation == 1
    # same request id, different body -> rejected, nothing committed
    forged = VaultRequest(
        operation=Operation.SET, reference=request.reference, request_id=request.request_id,
        issued_at=request.issued_at, value="tampered",
    )
    assert rig.call(forged).code is ErrorCode.REPLAY_REJECTED
    assert rig.call(rig.request(Operation.GET)).value == CANARY
    stale = VaultRequest(operation=Operation.GET, reference=request.reference, request_id="stale-request-1",
                         issued_at=time.time() - 3600)
    assert rig.call(stale).code is ErrorCode.REPLAY_REJECTED


# 5. atomic set / replace / delete and generation conflicts ────────────────


def test_generations_guard_concurrent_replacement(rig: Rig):
    assert rig.call(rig.request(Operation.SET, value="v1")).generation == 1
    assert rig.call(rig.request(Operation.SET, value="v2", expected_generation=1)).generation == 2
    stale = rig.call(rig.request(Operation.SET, value="v3", expected_generation=1))
    assert stale.code is ErrorCode.CONFLICT
    assert rig.call(rig.request(Operation.GET)).value == "v2"
    assert rig.call(rig.request(Operation.ROTATE, value="v4", expected_generation=2)).generation == 3
    assert rig.call(rig.request(Operation.ROTATE, ns=NS, key="never-set", value="x")).code is ErrorCode.NOT_FOUND
    assert rig.call(rig.request(Operation.DELETE, expected_generation=2)).code is ErrorCode.CONFLICT
    assert rig.call(rig.request(Operation.DELETE, expected_generation=3)).generation == 4
    assert rig.call(rig.request(Operation.GET)).code is ErrorCode.NOT_FOUND
    assert rig.call(rig.request(Operation.DELETE)).code is ErrorCode.NOT_FOUND
    assert rig.call(rig.request(Operation.SET, value="v5")).generation == 5  # sequence continues past the tombstone


# 6. crash between candidate write and commit ─────────────────────────────


def test_crash_before_commit_preserves_the_previous_value(rig: Rig, monkeypatch):
    rig.call(rig.request(Operation.SET, value="committed"))

    def crash() -> None:
        raise OSError("disk pulled")

    monkeypatch.setattr(rig.store, "_commit_hook", crash)
    response = rig.call(rig.request(Operation.SET, value="never"))
    assert response.code is ErrorCode.BACKEND_UNAVAILABLE
    monkeypatch.undo()
    assert list(rig.root.rglob("*.candidate")), "the aborted candidate is on disk"
    restarted = storage.FileDurableSecretStore(rig.root / "store", rig.keys)  # recovery on start
    assert not list(rig.root.rglob("*.candidate"))
    record, value = restarted.get(SecretReference(NS, KEY))
    assert value == b"committed" and record.generation == 1


# 7. restart preserves committed values and deletions ─────────────────────


def test_restart_preserves_values_and_deletions(rig: Rig):
    rig.call(rig.request(Operation.SET, value=CANARY))
    rig.call(rig.request(Operation.SET, ns=NS, key="second", value="two"))
    rig.call(rig.request(Operation.DELETE, ns=NS, key="second"))
    fresh = storage.FileDurableSecretStore(rig.root / "store", rig.keys)
    fresh_service = service.HostVaultService(store=fresh, registry=identity.TrustRegistry(rig.root / "trust.json", ca=rig.ca))
    assert fresh_service.handle(rig.request(Operation.GET).to_wire(), peer_cert_pem=rig.cert).value == CANARY
    assert fresh_service.handle(rig.request(Operation.GET, key="second").to_wire(), peer_cert_pem=rig.cert).code is ErrorCode.NOT_FOUND


# 8. corruption fails closed ───────────────────────────────────────────────


def _record_path(rig: Rig) -> Path:
    return next(p for p in (rig.root / "store").rglob("*.json"))


@pytest.mark.parametrize("tamper", ["ciphertext", "metadata", "key_version"])
def test_corrupt_records_fail_closed(rig: Rig, tamper: str):
    rig.call(rig.request(Operation.SET, value=CANARY))
    path = _record_path(rig)
    payload = json.loads(path.read_text())
    if tamper == "ciphertext":
        blob = bytearray(__import__("base64").b64decode(payload["sealed"]["ciphertext"]))
        blob[-1] ^= 0x01
        payload["sealed"]["ciphertext"] = __import__("base64").b64encode(bytes(blob)).decode()
        # re-sign integrity so ONLY the AEAD catches it
        payload.pop("integrity")
        payload["integrity"] = storage.FileDurableSecretStore._digest(payload)
    elif tamper == "metadata":
        payload["generation"] = 99  # integrity digest no longer matches
    else:
        payload["sealed"]["root_key_id"] = "fake-999"
        payload.pop("integrity")
        payload["integrity"] = storage.FileDurableSecretStore._digest(payload)
    path.write_text(json.dumps(payload))
    response = rig.call(rig.request(Operation.GET))
    assert response.code is ErrorCode.CORRUPT_RECORD and response.value is None
    assert "fake-999" not in response.message and "disk" not in response.message


# 9. audit records ─────────────────────────────────────────────────────────


def test_audit_records_carry_identity_and_digest_never_secrets(rig: Rig):
    rig.call(rig.request(Operation.SET, value=CANARY))
    rig.call(rig.request(Operation.GET, OTHER_TENANT))
    serialized = json.dumps([event.to_dict() for event in rig.audit.events])
    assert CANARY not in serialized and KEY not in serialized
    set_event = rig.audit.events[0]
    assert set_event.deployment_id == "dep-1"
    assert set_event.fingerprint == rig.record.fingerprint
    assert set_event.operation == "secret.set" and set_event.code == "ok" and set_event.generation == 1
    assert set_event.reference_digest == SecretReference(NS, KEY).digest
    assert set_event.request_id and set_event.time > 0
    denied = rig.audit.events[1]
    assert denied.code == "forbidden" and denied.application == "connection-hub@1-0"


# 10. adversarial exception text is sanitized ─────────────────────────────


def test_backend_exceptions_with_canaries_never_reach_the_caller(rig: Rig, monkeypatch):
    def explode(*args, **kwargs):
        raise OSError(f"/var/lib/kdcube/vault/{CANARY}/store.json: permission denied")

    monkeypatch.setattr(rig.store, "get", explode)
    response = rig.call(rig.request(Operation.GET))
    assert response.code is ErrorCode.BACKEND_UNAVAILABLE
    assert CANARY not in json.dumps(response.to_wire())

    def boom(*args, **kwargs):
        raise RuntimeError(f"root key {CANARY} unwrap failed")

    monkeypatch.setattr(rig.store, "get", boom)
    response = rig.call(rig.request(Operation.GET))
    assert response.code is ErrorCode.INTERNAL and CANARY not in json.dumps(response.to_wire())


# 11. the broker keeps no authoritative value ──────────────────────────────


class _Direct(broker.VaultTransport):
    """Transport that calls the service in-process with the rig's certificate."""

    def __init__(self, rig: Rig) -> None:
        self.rig = rig
        self.calls: list[VaultRequest] = []

    def call(self, request: VaultRequest):
        self.calls.append(request)
        return self.rig.service.handle(request.to_wire(), peer_cert_pem=self.rig.cert)


def test_broker_is_stateless_and_acknowledges_only_committed_writes(rig: Rig, monkeypatch):
    transport_ = _Direct(rig)
    b = broker.SecretsBroker(transport=transport_, tenant="demo-tenant", project="demo-project")
    assert b.set(application="connection-hub@1-0", key=KEY, value=CANARY).ok
    assert b.get(application="connection-hub@1-0", key=KEY) == CANARY
    assert CANARY not in json.dumps({k: str(v) for k, v in vars(b).items()})
    assert b.get(application="other-app@1-0", key=KEY) is None  # forbidden reads as absent
    # a write the store refused is NOT acknowledged
    monkeypatch.setattr(rig.store, "_commit_hook", lambda: (_ for _ in ()).throw(OSError("crash")))
    result = b.set(application="connection-hub@1-0", key=KEY, value="lost")
    assert result.ok is False and result.code is ErrorCode.BACKEND_UNAVAILABLE
    monkeypatch.undo()
    assert b.get(application="connection-hub@1-0", key=KEY) == CANARY
    assert b.delete(application="connection-hub@1-0", key=KEY).ok
    assert b.delete(application="connection-hub@1-0", key=KEY).code is ErrorCode.NOT_FOUND  # settled
    assert b.get(application="connection-hub@1-0", key=KEY) is None
    # references are derived by the broker: the namespace never comes from the key
    assert all(req.reference.namespace.tenant == "demo-tenant" for req in transport_.calls if req.reference)


# 12. root / data-key rotation ─────────────────────────────────────────────


def test_root_key_rotation_rewraps_without_touching_values(rig: Rig):
    rig.call(rig.request(Operation.SET, value=CANARY))
    before = json.loads(_record_path(rig).read_text())
    old_key_id = before["sealed"]["root_key_id"]
    new_key_id = rig.keys.rotate()
    assert new_key_id != old_key_id
    assert rig.call(rig.request(Operation.GET)).value == CANARY  # old version still opens
    assert rig.store.rewrap_all() == 1
    after = json.loads(_record_path(rig).read_text())
    assert after["sealed"]["root_key_id"] == new_key_id
    assert after["sealed"]["ciphertext"] == before["sealed"]["ciphertext"]  # value bytes untouched
    assert after["sealed"]["wrapped_data_key"] != before["sealed"]["wrapped_data_key"]
    assert rig.call(rig.request(Operation.GET)).value == CANARY
    rig.call(rig.request(Operation.SET, ns=NS, key="fresh", value="new"))
    fresh_payload = [json.loads(p.read_text()) for p in (rig.root / "store").rglob("*.json")]
    assert all(row["sealed"]["root_key_id"] == new_key_id for row in fresh_payload if not row["deleted"])


def test_file_root_key_provider_refuses_readable_keys(tmp_path: Path):
    provider = keys.FileRootKeyProvider(tmp_path / "rootkeys")
    key_id = provider.rotate()
    assert provider.current_key_id() == key_id and len(provider.key(key_id)) == 32
    (tmp_path / "rootkeys" / f"{key_id}.key").chmod(0o644)
    with pytest.raises(VaultError) as exc:
        provider.key(key_id)
    assert exc.value.code is ErrorCode.BACKEND_UNAVAILABLE


# 13. no caller bearer is workload proof (over real mTLS) ──────────────────


@pytest.fixture
def served(rig: Rig):
    skey, scert = rig.ca.issue_server(hostnames=["localhost", "127.0.0.1"])
    for name, data in (("server.key", skey), ("server.crt", scert), ("ca.crt", rig.ca.cert_pem)):
        (rig.root / name).write_bytes(data)
    server = transport.HostVaultServer(
        tls=transport.ServerTLS(rig.root / "server.crt", rig.root / "server.key", rig.root / "ca.crt"),
        handler=lambda body, peer: rig.service.handle(body, peer_cert_pem=peer),
    )
    server.serve_in_thread()
    rig.key.write_identity_files(rig.root / "identity", cert_pem=rig.cert, ca_pem=rig.ca.cert_pem)
    yield server
    server.shutdown()


def _client(rig: Rig, server, *, cert="host-vault-client.crt", key="host-vault-client.key") -> transport.HostVaultClient:
    host, port = server.address
    tls = transport.ClientTLS(rig.root / "identity" / cert, rig.root / "identity" / key, rig.root / "identity" / "host-vault-ca.crt")
    return transport.HostVaultClient(host=host, port=port, tls=tls, server_hostname="localhost")


def test_mtls_round_trip_and_identity_file_modes(rig: Rig, served):
    assert (rig.root / "identity" / "host-vault-client.key").stat().st_mode & 0o777 == 0o400
    client = _client(rig, served)
    b = broker.SecretsBroker(transport=client, tenant="demo-tenant", project="demo-project")
    assert b.health()["deployment_id"] == "dep-1"
    assert b.set(application="connection-hub@1-0", key=KEY, value=CANARY).ok
    assert b.get(application="connection-hub@1-0", key=KEY) == CANARY


def test_bearer_without_client_certificate_is_not_workload_proof(rig: Rig, served):
    import http.client
    import ssl

    host, port = served.address
    ctx = ssl.create_default_context(cafile=str(rig.root / "ca.crt"))
    ctx.check_hostname = False
    connection = http.client.HTTPSConnection(host, port, context=ctx, timeout=5)
    body = json.dumps({**rig.request(Operation.GET).to_wire(), "deployment_id": "dep-1", "bearer": "kst1.copied"})
    with pytest.raises(Exception):  # the handshake itself refuses: no client certificate
        connection.request("POST", transport.VAULT_PATH, body=body, headers={
            "Authorization": "Bearer kst1.copied", "X-KDCUBE-ADMIN-TOKEN": "copied",
        })
        connection.getresponse().read()
    # and the service, handed a body that CLAIMS an identity but no peer certificate, refuses too
    response = rig.service.handle(json.loads(body), peer_cert_pem=None)
    assert response.code is ErrorCode.UNAUTHENTICATED


def test_revoked_certificate_fails_on_the_next_connection(rig: Rig, served):
    client = _client(rig, served)
    b = broker.SecretsBroker(transport=client, tenant="demo-tenant", project="demo-project")
    assert b.health()["ok"]
    rig.registry.revoke(rig.record.fingerprint)
    assert b.health() == {"ok": False, "code": "unauthenticated"}


def test_tls_failures_are_sanitized(rig: Rig, served):
    stranger = identity.HostIssuingCA.generate()
    skey = identity.DeploymentKey.generate()
    (rig.root / "identity" / "stranger.crt").write_bytes(stranger.issue(skey.csr(), deployment_id="dep-1"))
    (rig.root / "identity" / "stranger.key").write_bytes(skey.pem)
    client = _client(rig, served, cert="stranger.crt", key="stranger.key")
    with pytest.raises(VaultError) as exc:
        client.call(rig.request(Operation.HEALTH))
    assert exc.value.code is ErrorCode.BACKEND_UNAVAILABLE
    assert "certificate" not in exc.value.message.lower() and "ssl" not in exc.value.message.lower()


# protocol grammar ─────────────────────────────────────────────────────────


def test_reference_grammar_rejects_arbitrary_paths():
    for bad in ("kdv1:demo-tenant/demo-project/app/../x", "kdv1:a/b", "vault:/etc/passwd", "kdv1:demo tenant/p/a/n"):
        with pytest.raises(VaultError) as exc:
            SecretReference.parse(bad)
        assert exc.value.code is ErrorCode.INVALID_REQUEST
    ref = SecretReference.parse("kdv1:demo-tenant/demo-project/connection-hub@1-0/users.u1.secrets.token")
    assert ref.namespace == NS and ref.name == "users.u1.secrets.token"
    assert SecretReference.derive(namespace=NS, internal_key="users.u1.secrets.token") == ref


def test_values_are_bounded(rig: Rig):
    too_big = "x" * (protocol.MAX_VALUE_BYTES + 1)
    assert rig.call(rig.request(Operation.SET, value=too_big)).code is ErrorCode.TOO_LARGE
