---
id: repo:kdcube-ai-app/app/ai-app/docs/service/secrets/secret-management-cli-README.md
title: "Manage KDCube Secrets"
summary: "Canonical backend-neutral CLI and API flow for exact secret metadata, read, write, delete, owner export, and storage-backend lifecycle."
tags: ["service", "secrets", "cli", "delegation", "operations"]
keywords: ["kdcube secrets", "secret metadata", "secret get", "secret set", "secret delete", "secret export", "backend status", "delegated bearer"]
updated_at: 2026-09-05
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/service/cicd/delegated-management-service-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/service/secrets/secrets-service-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/service/secrets/host-vault-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/configuration/secrets-descriptor-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/arch/security-and-trust-model-README.md
---
# Manage KDCube Secrets

`kdcube secrets` is the canonical operator surface for logical KDCube
secrets. Logical operations name an exact key and scope. KDCube resolves the
deployment-selected `ISecretsManager`; KDCube alone chooses the physical store
for each request.

```text
kdcube secrets metadata|get|set|delete|export
                    |
                    | local workdir or remote KDCube endpoint
                    v
authenticated KDCube management API
                    |
                    | live Card admission for delegated operations
                    v
descriptor-selected ISecretsManager
        +-----------+----------------+------------------+
        |           |                |                  |
  secrets-file  host-vault   AWS Secrets Manager   in-memory
```

Backend inspection and migration live below the same noun:

```text
kdcube secrets backend status
kdcube secrets backend host-vault prepare|stage|activate|recover
```

The former `kdcube secrets host-vault ...` spelling remains a compatibility
alias. New procedures use `kdcube secrets backend host-vault ...`.

## Command Matrix

| Command | Authority | Value handling |
| --- | --- | --- |
| `metadata` | Live Card grant for one exact metadata resource | Returns existence, provider, and writability as metadata only. |
| `get` | Live Card grant for one exact value-read resource | Writes one admitted value to `--output`; the terminal receives a receipt. |
| `set` | Live Card grant for one exact value-write resource | Reads the value from a hidden prompt or stdin; returns a receipt. |
| `delete` | Live Card grant for one exact delete resource | Returns whether an exact value existed as metadata only. |
| `export` | Fresh browser-confirmed, one-use admin transaction | Writes only the explicitly named manifest into a new private directory. |
| `backend status` | Local operator access to the runtime workdir | Reports descriptor-selected storage and migration metadata. |
| `backend host-vault ...` | Local deployment operator | Prepares, stages, activates, or recovers the local durable backend. |

The exact delegated operations are:

```text
kdcube.management.secret.metadata.read
kdcube.management.secret.value.read
kdcube.management.secret.value.write
kdcube.management.secret.delete
```

Each operation is granted against one resource:

```text
urn:kdcube:management:secret:<tenant>:<project>:<scope>:<bundle-or-_>:<key>
```

`Once` and `Always` use the same Card editor and invocation-policy registry as
other delegated operations. `Once` remains a Card selection with a one-use
policy; the server consumes it atomically when it admits the operation.

## Select A Deployment

A local command derives the endpoint and coordinates from an initialized
workdir:

```bash
kdcube secrets metadata services.brave.api_key \
  --scope platform \
  --workdir ~/.kdcube/kdcube-runtime/demo-tenant__demo-project
```

Stored CLI defaults can select the same workdir. An explicit workdir wins over
defaults saved for another deployment.

A remote command names the KDCube origin and coordinates explicitly:

```bash
kdcube secrets metadata services.brave.api_key \
  --scope platform \
  --endpoint https://runtime.example \
  --tenant demo-tenant \
  --project demo-project
```

Remote non-loopback endpoints require HTTPS. The same route works when KDCube
runs on ECS; trusted KDCube code reaches the provider selected by that cloud
deployment, normally AWS Secrets Manager with task IAM. Host Vault lifecycle
commands are scoped to local Compose; ECS infrastructure keeps its declared
AWS path.

## Human And Agent Credentials

Delegated metadata, get, set, and delete requests require a live Card-bound
bearer. In an interactive terminal, the CLI asks for it with a hidden prompt:

```text
Delegated KDCube bearer:
```

For an agent or trusted credential helper, `--credential-stdin` reads the
bearer from the first line of standard input. Hidden prompt and stdin are the
only accepted credential inputs; arguments and KDCube environment variables
remain non-secret configuration surfaces.

For `set --credential-stdin --value-stdin`, stdin has one explicit framing
rule:

```text
<delegated bearer><newline><exact secret value through end of stream>
```

The newline terminates the bearer and is excluded from the value. Every byte
after it is the UTF-8 secret value, including any final newline. This lets an
agent keep both values in a controlled process-to-process pipe and out of
command history and process arguments.

The generic KDCube CLI keeps that bearer process-local. A product CLI can add
native credential custody around the reusable library. Connection
Hub does this for its selected host: `connection-hub host authorize` stores an
OAuth session in the platform credential store, and its `secrets host ...`
commands call the same KDCube management models and transports.

## Exact Operations

Metadata returns existence and provider capability:

```bash
kdcube secrets metadata services.brave.api_key \
  --scope platform \
  --workdir ~/.kdcube/kdcube-runtime/demo-tenant__demo-project
```

Set uses a hidden value prompt by default:

```bash
kdcube secrets set services.brave.api_key \
  --scope platform \
  --workdir ~/.kdcube/kdcube-runtime/demo-tenant__demo-project
```

Get requires an explicit output path. On POSIX the file is created with mode
`0600`; on Windows it inherits the selected parent directory's ACL. Existing,
symlink, directory, and non-file targets fail closed unless a regular file is
explicitly replaced.

```bash
umask 077
kdcube secrets get services.brave.api_key \
  --scope platform \
  --output ./brave-api-key.txt \
  --workdir ~/.kdcube/kdcube-runtime/demo-tenant__demo-project
```

Delete targets the same logical key:

```bash
kdcube secrets delete services.brave.api_key \
  --scope platform \
  --workdir ~/.kdcube/kdcube-runtime/demo-tenant__demo-project
```

Bundle keys add one exact application id:

```bash
kdcube secrets metadata connections.oauth_state_secret \
  --scope bundle \
  --bundle-id connection-hub@1-0 \
  --workdir ~/.kdcube/kdcube-runtime/demo-tenant__demo-project
```

These commands use explicit keys within platform and bundle scopes. Connection
Hub's per-user connector credentials have their own owner surface, which
supports presence, replacement, and removal while keeping values concealed.

## Demand-Driven Consent

An absent exact grant returns a structured denial with a same-origin consent
URL. The request includes an invocation id and is default-closed.

```json
{
  "ok": false,
  "status": 403,
  "error": {
    "code": "missing_operation_grant",
    "retryable": false
  },
  "recovery": {
    "type": "consent_required",
    "choices": ["allow_once", "allow_always"]
  }
}
```

In an interactive terminal the CLI opens the consent page, waits for the user,
and retries the identical request once. `--no-open --no-wait --json` is the
agent-oriented mode: the agent presents the returned URL to the user, then
retries with the same explicit `--invocation-id`. A new invocation id is a new
operation attempt.

## Owner-Only Export

Descriptor export derives authority from a fresh, exact-manifest browser
transaction, independently of delegated bearers and Card grants:

```bash
kdcube secrets export \
  --platform-key services.brave.api_key \
  --bundle-key connection-hub@1-0=connections.oauth_state_secret \
  --output-directory ./kdcube-secret-export-20260905 \
  --endpoint https://runtime.example \
  --tenant demo-tenant \
  --project demo-project
```

The KDCube page authenticates through the authority configured by that
deployment, displays the exact keys and digest, and issues one PKCE-bound code
after explicit approval. The code can be exchanged once. The destination must
be new and receives canonical `secrets.yaml` and `bundles.secrets.yaml` files.

This is the explicit path back to descriptors after a durable backend becomes
authoritative. Ordinary `set` and `delete` operations update the selected
provider; retained bootstrap descriptor files remain rollback snapshots.

## Reusable Management Library

Application-specific CLIs use the installable `kdcube_cli.management` package
for the provider-neutral protocol:

```python
from kdcube_cli.management import (
    HttpxManagementTransport,
    ManagementClient,
    ManagementRequest,
    ManagementTarget,
    management_view,
)

target = ManagementTarget.create(
    public_base_url="https://runtime.example",
    tenant="demo-tenant",
    project="demo-project",
)
request = ManagementRequest.secret_metadata(
    target,
    scope="platform",
    key="services.brave.api_key",
)
client = ManagementClient(transport=HttpxManagementTransport())
```

The product CLI owns how it acquires and safeguards the delegated bearer, then
passes it only to `ManagementClient.execute`. KDCube owns target validation,
request construction, response bounds, consent-recovery validation,
secret-safe projection, private writers, and browser export. Connection Hub
uses this boundary for its native-store OAuth session and selected-host UX.

## Storage Status And Migration

Inspect a local deployment through secret-free metadata:

```bash
kdcube secrets backend status \
  --workdir ~/.kdcube/kdcube-runtime/demo-tenant__demo-project \
  --json
```

The report distinguishes:

- the provider and service backend in staged `assembly.yaml`;
- the configured authoritative store;
- whether plaintext descriptor files still exist;
- Host Vault `shadow-configured`, `active`, or `not-applicable` state;
- incomplete activation requiring recovery.

Its evidence is explicitly `staged-descriptor` and `runtime_verified: false`.
For a live per-key check, use `metadata`; its result comes from the running
management service and selected provider.

The local Host Vault transition is transactional:

```bash
kdcube secrets backend host-vault prepare --tenant T --project P --dry-run
kdcube secrets backend host-vault prepare --tenant T --project P
kdcube secrets backend host-vault stage --tenant T --project P --dry-run
kdcube secrets backend host-vault stage --tenant T --project P
kdcube secrets backend host-vault activate --tenant T --project P --dry-run
kdcube secrets backend host-vault activate --tenant T --project P --yes
```

`stage` copies and verifies existing file-backed values while
`secrets.provider: secrets-file` remains authoritative. `activate` quiesces
consumers, switches to `secrets-service` backed by `host-vault`, recreates the
required services, performs real-read verification, and rolls back ordinary
failures. It retains bootstrap files for explicit rollback acceptance and
leaves their cleanup to an explicit operator step.
