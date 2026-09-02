---
id: repo:kdcube-ai-app/app/ai-app/docs/recipes/connections/protect-external-service-with-connection-hub-README.md
title: "Protect An External Service With Connection Hub"
summary: "Hosts Connection Hub direct admission in KDCube so a registered backend outside KDCube can enforce current delegated card, invocation policy, and pairwise caller identity on each operation."
status: active
tags: ["recipes", "connections", "connection-hub", "connection-hub", "delegated-access", "protected-service"]
keywords: ["direct admission", "external backend", "opaque bearer", "workload proof", "pairwise identity", "invocation policy", "idempotency", "resource server"]
updated_at: 2026-09-02
see_also:
  - https://github.com/elenaviter/app-ecosystem/blob/main/docs/connection-hub/recipes/direct-protected-service.md
  - https://github.com/elenaviter/app-ecosystem/tree/main/examples/connection-hub/direct-admission-service
  - repo:kdcube-ai-app/app/ai-app/docs/arch/delegated-authority-and-admission-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/connections/connection-hub-solution-README.md
---

# Protect an external service with Connection Hub

Use this path when the protected backend runs outside KDCube and therefore
cannot use a KDCube-managed REST or MCP guard. The Connection Hub app remains
the policy-enforcement service; the external backend sends the user's opaque
delegated bearer plus its own independently signed workload proof for one
resource and operation.

The canonical capability, service-registration, signing, response, and
verification contract is [Protect an external backend with Connection
Hub](https://github.com/elenaviter/app-ecosystem/blob/main/docs/connection-hub/recipes/direct-protected-service.md).
This recipe owns only the KDCube host steps.

## 1. Load the Connection Hub app

Select a committed app-ecosystem revision rather than a mutable branch:

```yaml
bundles:
  items:
    - id: connection-hub@1-0
      name: Connection Hub
      repo: https://github.com/elenaviter/app-ecosystem.git
      ref: <APP_ECOSYSTEM_COMMIT_OR_TAG>
      subdir: products/connection-hub/apps/connection-hub@1-0
      module: entrypoint
      singleton: false
      config:
        connections:
          delegated_credentials:
            admission:
              enabled: true
              identity_projection_secret_ref: connections.delegated_credentials.admission.identity_projection_secret
              services: {}
```

Add the catalog grant/resource/operation and the protected-service registration
from the canonical recipe under this same app's `config.connections` node.
Service registration names resource selectors only; the delegated catalog owns
operations and grants.

## 2. Resolve secrets server-side

Put the identity-projection secret and each protected service's signing secret
in `bundles.secrets.yaml` or the deployment's configured app-secret provider.
Descriptor rows contain only `*_ref` paths. The values are at least 32 random
bytes and never enter browser configuration, a delegated bearer, or model
context.

## 3. Apply and load the app

Use the normal descriptor and app lifecycle:

```bash
kdcube bundle config apply
kdcube bundle reload connection-hub@1-0
kdcube bundle status connection-hub@1-0 --json
```

For a local source/platform change, refresh the staged runtime before testing;
the running containers do not execute directly from the KDCube checkout.
When an operator selects a local app source, the seed descriptor may name its
host path, but the staged workdir descriptor must contain the corresponding
container-visible path. With the standard source mount and sibling
`app-ecosystem` checkout, that path is:

```yaml
path: /bundles/products/app-ecosystem/products/connection-hub/apps/connection-hub@1-0
```

`kdcube bundle status connection-hub@1-0 --json` reports both `host_path` and
`runtime_path`; verify that the former exists and the latter is reachable in
the running processor before interpreting an application-readiness failure as
an app defect.

## 4. Configure the external backend

The current KDCube host route is:

```text
POST /api/integrations/bundles/{tenant}/{project}/connection-hub@1-0/public/delegated_admission
```

The host route is discoverable in the Connection Hub OAuth metadata as
`connection_hub_delegated_admission_endpoint`. It is not a permanent Connection Hub product
URL. A stable shorter alias remains a separate KDCube router capability.

Run the [reference protected
service](https://github.com/elenaviter/app-ecosystem/tree/main/examples/connection-hub/direct-admission-service)
against that endpoint. The external service receives only its own workload
secret. KDCube retains app secrets, replay state, current card/catalog storage,
and provider credentials.

For every domain request, the service also supplies one stable invocation id
and a canonical request digest. Connection Hub uses them for `once` policy and
admission replay. A service that changes domain state keeps its own idempotency
ledger under the same id; Connection Hub records the decision but does not
execute or record the external service's effect.

## 5. Verify the live boundary

Exercise the deployed HTTP route, not only package tests:

- a valid fresh request with current delegated authority returns an allow;
- reusing the same nonce returns `409 admission_request_replayed`;
- changing the operation, resource, bearer, or body invalidates the signature
  or current-authority decision;
- narrowing or revoking the card changes the next fresh decision;
- an ungranted operation returns exact recovery for `Allow once` or
  `Allow always`;
- `once` admits one new invocation id, denies a second, and replays the same
  decision for the successful id and digest;
- changing the request under a used invocation id is denied;
- a write fixture applies its provider-side effect once across a retry;
- unavailable replay or authority storage fails closed with a retryable 503;
- the same service/user/profile receives stable pairwise `sub` and `client_id`;
- another caller profile changes only the pairwise caller-profile id, and
  another service correlates neither id;
- no response contains a raw caller id, card access id, provider credential,
  or internal platform user id.
