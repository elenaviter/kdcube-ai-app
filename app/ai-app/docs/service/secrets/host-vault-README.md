---
id: repo:kdcube-ai-app/app/ai-app/docs/service/secrets/host-vault-README.md
title: "Host Vault for Provider Secrets"
summary: "Durable, host-owned vault for provider credentials: versioned protocol, deployment workload identity over mTLS, envelope-encrypted store, and the stateless kdcube-secrets broker. First phase, not yet wired into the secrets provider."
tags: ["service", "secrets", "security", "vault", "runtime"]
keywords: ["host vault", "kdcube-host-vault/1", "workload identity", "mTLS", "kdcube-secrets broker", "envelope encryption", "trust registry", "hostvaultctl"]
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/service/secrets/secrets-service-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/arch/security-and-trust-model-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/configuration/secrets-descriptor-README.md
---
# Host Vault for Provider Secrets

Provider credentials a deployment holds on a user's behalf (OAuth refresh
tokens, app passwords, API keys stored through Connection Hub) need a home
that survives service, Docker, and host restarts, that no copied string can
open, and that answers only the deployment they belong to.

The host vault is that home. It is a small service the host operator owns,
outside the KDCube runtime workdir. Deployments reach it over mutual TLS with
a key generated inside their own boundary, and the vault checks every request
against a live trust registry before it touches the store.

This page describes the first phase: the protocol, the modules, the operator
tool, and the proofs. The runtime still selects its secrets backend through
`SECRETS_PROVIDER` as documented in
[secrets-service-README.md](secrets-service-README.md). Nothing on this page
changes that selection yet. The broker here is constructible and testable on
its own.

## 1. Trust model in one pass

```text
KDCube service ──(internal secrets-service HTTP, in-deployment)──▶ kdcube-secrets broker
                                                                        │
                                       deployment certificate + key (mTLS)
                                                                        ▼
                                                              host vault service
                                                                        │
                                                    trust registry: fingerprint → deployment, ACL, status
                                                    envelope-encrypted durable store
                                                    append-only audit log
```

Identity is the presented client certificate and nothing else. The transport
verifies the chain and key possession. The service then looks the
certificate's fingerprint up in the registry: the record must exist, be
active, and be unexpired at that moment. A bearer header, a deployment id in
the body, the socket address, or a process name never establish authority.
The registry's namespace ACL decides which `tenant/project/application`
references that identity may read or write.

Enrollment is a one-use ticket: the deployment generates its private key in
place, hands out only a CSR, and the host CA issues a certificate whose
subject is the deployment id the host assigned (the CSR's own subject is
discarded). Rotation issues a new certificate that overlaps the old for a
bounded interval. Revocation is a registry edit and blocks the next
connection: the running server reloads the registry when the file changes.

## 2. Protocol `kdcube-host-vault/1`

One JSON request per POST to `/v1/vault`. Fields:

| field | meaning |
| --- | --- |
| `protocol` | `kdcube-host-vault/1` |
| `operation` | `health`, `secret.get`, `secret.set`, `secret.delete`, `secret.rotate` |
| `reference` | `kdv1:<tenant>/<project>/<application>/<name>`; each segment matches `[A-Za-z0-9][A-Za-z0-9._@-]{0,127}` |
| `request_id` | 8 to 64 chars of `[A-Za-z0-9-]`, unique per request |
| `issued_at` | Unix seconds; the server rejects requests outside a 300 s skew window |
| `value` | present for set and rotate, at most 64 KiB |
| `expected_generation` | optional optimistic guard for set, rotate, delete |

Responses carry `ok`, `code`, `message`, `request_id`, and for successes
`value` (get) and `generation` (mutations). Codes: `ok`, `invalid_request`,
`unauthenticated`, `forbidden`, `not_found`, `conflict`, `replay_rejected`,
`too_large`, `corrupt_record`, `backend_unavailable`, `internal`. Failure
messages are fixed phrases per code, so backend or TLS exception text never
reaches a caller.

Replay: the service remembers `(deployment_id, request_id)` with the body
digest. The same request again returns the original result without a second
commit. The same id with a different body is `replay_rejected`.

Generations: every committed change to a reference increments its
generation, deletions included (a deletion is a tombstone record). A caller
that passes `expected_generation` gets `conflict` when the store moved on.

## 3. Persistence

Records live under `<home>/store/<digest[:2]>/<digest>.json`, keyed by a
SHA-256 digest of the reference so names never appear on disk. Each record
holds the generation, an integrity digest over its metadata, and the sealed
value: AES-256-GCM under a fresh data key, the data key wrapped by the
current root key, with additional authenticated data binding the record to
its reference digest and generation.

Writes are atomic: candidate file, fsync, `os.replace`, directory fsync. A
crash between candidate and commit leaves the previous value in place, and
the server removes stale candidates on start.

Root keys come from a `RootKeyProvider`. The shipped `FileRootKeyProvider`
keeps raw 32-byte keys as `0400` files in a `0700` directory with a `CURRENT`
marker, and refuses to serve a key file that is group- or other-readable.
Rotation makes a new key current and rewraps every record's data key (the
ciphertext itself is untouched). Old versions remain readable until rewrap
completes. A hardware- or OS-keychain-backed provider can replace this
class behind the same three calls (`current_key_id`, `key`, `rotate`).

Assumptions this phase makes and states: the vault home is owned by a
dedicated service user on the host, the root-key directory is not on a
KDCube-managed volume, and the appliance boundary around the deployment
private key is the operator's host isolation (the code does not claim a
production local boundary until a service-owned appliance or VM is
exercised).

## 4. Modules

```text
kdcube_ai_app/infra/secrets/host_vault/
  protocol.py   references, requests, responses, error codes, sanitizer
  keys.py       RootKeyProvider, FileRootKeyProvider, envelope seal/open/rewrap
  storage.py    FileDurableSecretStore (atomic commit, tombstones, recover)
  audit.py      AuditEvent, MemoryAuditSink, FileAuditSink
  identity.py   HostIssuingCA, DeploymentKey, TrustRegistry, enrollment, rotation, revocation
  service.py    HostVaultService.handle(body, peer_cert_pem)
  transport.py  ServerTLS, ClientTLS, HostVaultServer, HostVaultClient (stdlib ssl)
  broker.py     SecretsBroker: get/set/rotate/delete/health over a VaultTransport
  tests/        the proofs in section 7
```

The broker derives the vault reference from the deployment's canonical
tenant and project plus the trusted logical application the platform binds
(`connection-hub@1-0` by default) and the internal secrets-manager key. No
remote caller names a vault path. The broker caches nothing and returns `ok`
for a mutation only when the vault committed it. Reads that are forbidden or
unreachable come back as `None`, the same shape the existing
`SecretsServiceSecretsManager.get_secret` returns.

## 5. Deployment files

```text
app/ai-app/deployment/docker/all_in_one_kdcube/secrets/host_vault/
  vault_server.py    host service entrypoint (KDCUBE_HOST_VAULT_HOME, _BIND, _PORT)
  hostvaultctl.py    operator tool: init, enroll, rotate-identity, revoke, list,
                     rotate-root-key, deployment-keygen, deployment-install
  broker_server.py   kdcube-secrets broker with the existing HTTP shape
  requirements.txt   cryptography (all), fastapi/uvicorn/pydantic (broker only)
```

Vault home layout (`KDCUBE_HOST_VAULT_HOME`, default
`/var/lib/kdcube-host-vault`):

```text
ca/ca.key        issuing CA private key, 0400, service custody
tls/ca.crt       issuing CA certificate (also shipped to deployments)
tls/server.crt   vault server certificate for the names given at init
tls/server.key   0400
root-keys/       FileRootKeyProvider directory
store/           FileDurableSecretStore root
trust.json       TrustRegistry (atomically replaced on every change)
audit.log        append-only JSON lines
```

Operator flow:

```bash
# host, as the vault user
python hostvaultctl.py init --server-name vault.internal --server-name 10.0.0.5
python vault_server.py

# inside the deployment boundary
python hostvaultctl.py deployment-keygen --dir /run/kdcube-host-vault-identity
# hand ONLY host-vault-client.csr to the host operator

# host
python hostvaultctl.py enroll --deployment-id dep-prod-1 \
  --namespace demo-tenant/demo-project/connection-hub@1-0 \
  --csr host-vault-client.csr --out dep-prod-1.crt

# inside the deployment boundary
python hostvaultctl.py deployment-install --dir /run/kdcube-host-vault-identity \
  --cert dep-prod-1.crt --ca ca.crt
```

The broker reads `KDCUBE_HOST_VAULT_ADDR`, `KDCUBE_HOST_VAULT_SERVER_NAME`,
`KDCUBE_HOST_VAULT_IDENTITY_DIR`, `KDCUBE_SECRETS_TENANT`,
`KDCUBE_SECRETS_PROJECT`, and optionally `KDCUBE_SECRETS_APPLICATION`. It
serves `/health`, `GET /secret/{key}`, `POST /set`, and
`DELETE /secret/{key}` exactly as `secrets/secrets_server.py` does, so the
existing in-deployment client keeps working when the provider is switched
in a later phase. The old `X-KDCUBE-SECRET-TOKEN` and `X-KDCUBE-ADMIN-TOKEN`
headers stay an optional in-deployment door gate. They are never forwarded,
and the vault never sees them.

## 6. Audit

Every handled request appends one event: time, deployment id, certificate
fingerprint, operation, application, reference digest, request id, result
code, generation, and expected generation. Names and values never appear.
`FileAuditSink` opens the log with `O_APPEND` and fsyncs each line.

## 7. Proofs

`kdcube_ai_app/infra/secrets/host_vault/tests/test_host_vault.py` runs with
fake certificates and the labeled in-memory root-key provider and covers:

- exact namespace authorization with cross-tenant, cross-project, and cross-application denial
- certificate identity against the live registry, including a same-id certificate from a foreign CA
- one-use and expiring enrollment tickets
- revoked and expired identities, rotation overlap and lapse
- revocation from another process landing on the next identification
- replay with the same and with a different body, and a stale `issued_at`
- generation conflicts on set, rotate, and delete
- a crash between candidate write and commit
- restart durability of values and deletions
- tampered ciphertext, metadata, and root-key version failing closed
- audit fields without secret bytes
- backend exceptions with canaries sanitized
- a stateless broker that acknowledges only committed writes
- root-key rotation with rewrap, and identity file modes
- a real mTLS round trip, a bearer without a client certificate refused at the handshake and at the service, and sanitized TLS failures

Run from the repository root with the platform venv interpreter:

```bash
PYTHONPATH=app/ai-app/src/kdcube-ai-app \
app/venvs/ai-app/chat-processor/bin/python -m pytest \
  app/ai-app/src/kdcube-ai-app/kdcube_ai_app/infra/secrets/host_vault/tests \
  --import-mode=importlib -q
```

`--import-mode=importlib` matters: `infra/` has no `__init__.py`, and the
default import mode would put `kdcube_ai_app/infra` on `sys.path`, where
the `secrets` package shadows the standard library module.

## 8. What comes after this phase

Wiring the broker under `ISecretsManager(secrets-service)`, descriptor and
compose configuration, migration from the current backend with readback
comparison inside trusted code, Connection Hub connector regression, and
restart durability across service, Docker, and host restarts are separate
integration slices. Generated or isolated code never receives vault
credentials, the deployment private key, unrestricted references, or a
general vault client: trusted supervisor services project only the values
and operations they choose.
