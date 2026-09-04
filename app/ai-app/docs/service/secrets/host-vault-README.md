---
id: repo:kdcube-ai-app/app/ai-app/docs/service/secrets/host-vault-README.md
title: "Host Vault for Provider Secrets"
summary: "Durable, host-owned vault for provider credentials with an opt-in local Compose broker, deployment workload identity over mTLS, envelope-encrypted storage, and explicit migration and hardening gates."
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

This page describes the protocol, local Compose integration, operator tool,
and proofs. The runtime still selects its secrets provider through the
descriptor-owned `secrets.provider` setting documented in
[secrets-service-README.md](secrets-service-README.md). `secrets-file` remains
the shipped default. The host-vault broker may be started beside that file
provider for shadow staging, then selected explicitly as the backing
implementation of `secrets-service` after acceptance. Selecting the broker
never migrates or deletes an existing secret descriptor.

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
references that identity may read or write. KDCube's current secrets manager
is deployment-wide, so its broker binds the exact logical application
`kdcube-runtime`. Platform, bundle, and per-user keys remain distinguished by
their existing internal key grammar inside that namespace. A remote caller
cannot choose either the namespace or a key.

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
| `operation` | `health`, `secret.get`, `secret.list`, `secret.set`, `secret.delete`, `secret.rotate` |
| `reference` | `kdv1:<tenant>/<project>/<application>/<name>`; namespace segments match `[A-Za-z0-9][A-Za-z0-9._@-]{0,127}` and the bounded name uses the protocol's dotted/slashed internal-key grammar |
| `request_id` | 8 to 64 chars of `[A-Za-z0-9-]`, unique per request |
| `issued_at` | Unix seconds; the server rejects requests outside a 300 s skew window |
| `value` | present for set and rotate, at most 64 KiB |
| `expected_generation` | optional optimistic guard for set, rotate, delete |

Responses carry `ok`, `code`, `message`, `request_id`, and for successes
`value` (get), `names` (list), and `generation` (mutations). Codes: `ok`, `invalid_request`,
`unauthenticated`, `forbidden`, `not_found`, `conflict`, `replay_rejected`,
`too_large`, `corrupt_record`, `backend_unavailable`, `internal`. Failure
messages are fixed phrases per code, so backend or TLS exception text never
reaches a caller.

Replay: during one host-vault service process, a bounded cache remembers
`(deployment_id, request_id)` with the body digest. The same cached request
returns the original result without a second commit; the same id with a
different body is `replay_rejected`. This receipt cache is intentionally not a
durable effect ledger and is empty after service restart. Trusted callers use
generation guards for cross-restart mutation conflicts. Card-governed KDCube
management calls additionally use the Redis effect ledger documented in
[Delegated KDCube Management Service](../cicd/delegated-management-service-README.md)
for durable one-effect semantics.

Generations: every committed change to a reference increments its
generation, deletions included (a deletion is a tombstone record). A caller
that passes `expected_generation` gets `conflict` when the store moved on.

## 3. Persistence

Records live under `<home>/store/<digest[:2]>/<digest>.json`, keyed by a
SHA-256 digest of the reference so plaintext names never appear on disk. Each
new live record holds the encrypted name used for scoped inventory, the
generation, an integrity digest over its metadata, and the sealed value:
AES-256-GCM under a fresh data key, the data key wrapped by the current root
key, with additional authenticated data binding the record to its reference
digest and generation. Legacy records without an encrypted name remain
readable; their old `.__keys` metadata is accepted only as a bounded hint and
each referenced value is verified live before it appears in inventory.

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
and the internal secrets-manager key. The shipped HTTP broker always binds
`kdcube-runtime`, because it fronts KDCube's deployment-wide secrets manager;
it does not claim per-application process isolation. No remote caller names a
vault path. The broker caches nothing and returns `ok` for a mutation only
when the vault committed it. Reads that are forbidden or unreachable come
back as `None`, the same shape the existing
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
  --namespace demo-tenant/demo-project/kdcube-runtime \
  --csr host-vault-client.csr --out dep-prod-1.crt

# inside the deployment boundary
python hostvaultctl.py deployment-install --dir /run/kdcube-host-vault-identity \
  --cert dep-prod-1.crt --ca ca.crt
```

The broker reads `KDCUBE_HOST_VAULT_ADDR`, `KDCUBE_HOST_VAULT_SERVER_NAME`,
`KDCUBE_HOST_VAULT_IDENTITY_DIR`, `KDCUBE_SECRETS_TENANT`, and
`KDCUBE_SECRETS_PROJECT`. It preserves `/health`, `GET /secret/{key}`, `POST
/set`, and `DELETE /secret/{key}` from `secrets/secrets_server.py`. The host
vault implementation additionally accepts an optional generation guard on
`POST /set` and exposes admin-only `POST /verify` for secret-free migration
comparison. The old `X-KDCUBE-SECRET-TOKEN` and `X-KDCUBE-ADMIN-TOKEN` headers
stay an in-deployment door gate. They are never forwarded, and the vault never
sees them.

### Local Compose selection

The KDCube descriptor owns the selection. Shadow staging keeps the file
provider authoritative:

```yaml
secrets:
  provider: secrets-file
  service:
    backend: host-vault
    host_vault:
      address: host.docker.internal:7781
      server_name: host.docker.internal
      identity_dir: /absolute/service-owned/path/deployment-identity

platform:
  services:
    proc:
      exec:
        py_code_exec_network_mode: auto
```

After staging and regression acceptance, changing `provider` to
`secrets-service` selects the already populated broker. That cutover is a
separate operator action.

`identity_dir` is a host path outside the KDCube workdir containing exactly:

```text
host-vault-client.crt
host-vault-client.key
host-vault-ca.crt
```

The CLI projects this non-secret topology into Compose. Both maintained local
Compose layouts run the same `kdcube-secrets` image. Its default
`secrets.service.backend` is `ephemeral`, which runs the existing temporary
sidecar. `host-vault` runs the mTLS broker instead. Only that broker receives
read-only mounts for the three identity files and network access to the host;
ingress, proc, metrics, and generated executors receive none of them.

Before `kdcube start`, the CLI verifies that the provider/backend combination
is coherent, the identity directory is outside the workdir, all three files
exist and are regular files, and the private key is owner-only on POSIX.
`host.docker.internal` is mapped to the Docker host on Linux as well as Docker
Desktop. The vault service must bind an address reachable from Docker; mTLS
still authenticates both ends.

The `auto` trusted-runtime network setting keeps host-launched supervisors on
Docker's host network. Under local Docker-in-Docker it shares the processor's
network namespace, which already contains the normal internal-service network
and the private secrets-service network. This gives the trusted supervisor a
route to `kdcube-secrets` without publishing the broker. A split generated-code
executor remains a separate container with `--network none`, no broker token,
no descriptor payload, and no deployment identity. Host-vault preflight rejects
another network mode because a secret-using trusted tool would otherwise fail
only when invoked.

### Shadow-stage existing file secrets

With the host vault enrolled, the broker running, and `secrets-file` still
selected, inspect the destination without writing:

```bash
kdcube secrets host-vault stage \
  --tenant demo-tenant --project demo-project \
  --dry-run --json
```

Then stage and verify all configured non-placeholder values:

```bash
kdcube secrets host-vault stage \
  --tenant demo-tenant --project demo-project \
  --json
```

The CLI reads the owner-only `secrets.yaml` and `bundles.secrets.yaml` in its
trusted process. Values travel to `secretsctl` over stdin and never appear in
process arguments. Verification hashes the candidate in the CLI and compares
it with the stored value inside the broker; neither value is returned. The
result contains counts only.

Staging is idempotent and default-closed:

- every destination value is checked before the first write;
- an existing different value aborts the whole run before writes;
- an absent value is created with `expected_generation: 0`, so a racing or
  previously tombstoned record is not overwritten;
- each create is read back and compared, followed by a complete final pass;
- a failure leaves successful copies in place and leaves the source provider
  untouched, so the next run resumes by accepting exact matches;
- placeholders are counted and skipped;
- no command changes `secrets.provider` or deletes plaintext source files.

The shadow stage establishes destination parity. It is not activation. Keep
`secrets-file` selected until the secret-using runtime regression suite passes
against an explicitly activated test deployment. Plaintext cleanup remains a
later, separately confirmed operator action.

### Activate the staged provider

Check parity and runtime readiness without changing a file or container:

```bash
kdcube secrets host-vault activate \
  --tenant demo-tenant --project demo-project \
  --dry-run --json
```

Then perform the explicit provider switch:

```bash
kdcube secrets host-vault activate \
  --tenant demo-tenant --project demo-project \
  --yes --json
```

Without `--yes`, an interactive terminal asks for confirmation. Automation
and JSON mode require `--yes`. `--wait-seconds` bounds each broker and
consumer readiness check from 5 to 600 seconds; the default is 120.

Activation is a bounded local-Compose transaction:

1. It requires `kdcube-secrets`, `chat-ingress`, and `chat-proc` to be
   running and verifies that every current file-backed value already matches
   the host vault.
2. It durably creates `config/.host-vault-activation.pending.json`. The
   owner-only marker contains only schema, phase, and recovery-mode metadata;
   it contains no value, key name, digest, token, path, or identity material.
3. It quiesces ingress and proc so neither can write a file-backed secret
   while the final inventory is checked.
4. It changes `assembly.secrets.provider` and the two generated consumer
   environments to `secrets-service`.
5. It recreates the broker first and both consumers second with one temporary
   token overlay. Tokens remain out of process arguments and are not persisted
   by activation.
6. It resolves one staged value independently inside ingress and proc and
   compares digests without returning the value. It also rejects a source
   inventory change observed during the switch, then durably removes the
   marker.

Any ordinary failure after quiescing restores the exact prior configuration,
recreates the shadow-mode runtime, and verifies file-backed reads. If the host
vault itself prevents that exact restart, recovery selects the ephemeral
sidecar with `secrets-file`, preserving availability and the plaintext source.

If the CLI process or host stops after the marker is durable and before it is
removed, ordinary `kdcube start` refuses the unresolved transaction. Recover
to a known file-backed state before retrying activation:

```bash
kdcube secrets host-vault recover \
  --tenant demo-tenant --project demo-project \
  --yes --json
```

Recovery is repeatable. It selects `secrets-file` with the ephemeral sidecar,
recreates the broker, ingress, and proc, verifies a real file-backed read in
both consumers, and only then removes the marker. It does not depend on marker
contents to reconstruct configuration and never deletes the plaintext source.
If recovery fails, the marker remains and startup continues to fail closed.
This covers interrupted configuration activation; service, Docker, and host
restart durability of the active vault is still a separate acceptance gate.

Broker verification retries a small number of transient connection and
`502`/`503`/`504` failures. Permission, conflict, and malformed-request errors
remain immediate failures.

This selection is local-Compose-specific. ECS descriptors continue to select
`aws-sm`; no ECS task definition, IAM policy, or Terraform path is changed by
the host-vault switch.

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
- process-lifetime replay with the same and with a different body, a stale
  `issued_at`, and an explicit proof that the receipt cache resets on restart
- generation conflicts on set, rotate, and delete
- partial OS writes and a crash between candidate write and commit
- restart durability of values and deletions
- tampered ciphertext, metadata, and root-key version failing closed,
  including rotation refusing to rewrite a record that fails normal decode
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

## 8. Current gates

The protocol, opt-in local Compose broker, idempotent file-to-vault shadow
stage, operator-confirmed activation, automatic ordinary-failure rollback,
and explicit interrupted-activation recovery are implemented. The remaining
gates are deliberately separate:

- run Connection Hub connector create, invoke, replace, and delete plus Brave
  and connected-account regressions against the broker-backed provider
- prove service, Docker, and host restart durability and deployment-identity
  revocation on a real machine
- prove a complete split-runtime secret-using tool call with the broker enabled;
  parent-network routing and the networkless executor are covered separately
- package the host vault as a dedicated service-owned appliance or VM and add
  its enrollment/install lifecycle
- remove plaintext source values only through a later explicit cleanup action
  after activation and durability acceptance

Generated or isolated code never receives vault credentials, the deployment
private key, unrestricted references, or a general vault client. Trusted
supervisor services project only the values and operations they choose.
