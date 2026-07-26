---
id: repo:kdcube-ai-app/app/ai-app/docs/recipes/quickstart/expose-governed-service-mcp-README.md
title: "Authenticated MCP From Zero: A Builder Walkthrough"
summary: "Hands-on walkthrough from an empty descriptor to a working authenticated MCP surface: declare the managed door, the delegated resource ceiling, the delegable capabilities, the DCR redirect fence, and the per-operation namespace boundary policy; self-describe the provider's connected-accounts needs in its registration metadata; then verify all three scenarios - a hosted agent consenting in chat, an external MCP connection over OAuth with dynamic client registration, and a bounded automation token."
status: active
tags: ["quickstart", "recipe", "mcp", "connection-hub", "delegated-credentials", "named-services", "oauth", "governance", "app-authoring", "consent", "automation"]
keywords: ["authenticated mcp walkthrough", "mode: managed", "delegated resource", "capabilities", "delegable_roles", "delegable_permissions", "dynamic_client_registration", "named_services.namespaces", "named_services:use", "connected_accounts", "@named_service_provider", "automation access", "claude code mcp"]
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/connections/authenticated-mcp/authenticated-mcp-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/kdcube_for_agents/expose-mcp-service-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/kdcube_for_agents/consume-mcp-service-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/connections/configuring-agent-service-access/configuring-agent-service-access-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/connections/delegated-accounts/delegated-accounts-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/connections/delegated-accounts/custom-oauth-oidc-service-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/connections/delegate-kdcube-service-to-external-client-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/connections/protect-bundle-mcp-with-managed-credentials-README.md
---
# Authenticated MCP From Zero: A Builder Walkthrough

You start with an empty descriptor and end with a working authenticated MCP
surface, verified from three directions: a hosted agent consenting in chat, an
external MCP-speaking app connecting over OAuth with dynamic client
registration, and a bounded automation token. The example app acts on a user's
third-party accounts (Gmail) and on your own OAuth-protected server, and your
code never touches a token.

This walkthrough executes
[Authenticated MCP: The Full Configuration Chain](../../sdk/solutions/connections/authenticated-mcp/authenticated-mcp-README.md)
layer by layer - each step keeps only the hands-on literals here and links to
its layer there for the reference depth.

Current code and descriptors say **bundle** in names such as `bundle_id` and
`bundles.yaml`. Here, **app = bundle**: one deployable KDCube runtime unit.

All `my-service@1-0` / `my_server` literals below are an example app you are
building; the `connection-hub@1-0` paths are the real descriptor paths.

## The route

Two apps own the configuration - yours declares the door and describes itself,
Connection Hub (`connection-hub@1-0`) is the governance store:

```text
Step 0  connection-hub@1-0  delegated_to_kdcube.providers                connectable accounts
Step 1  my-service@1-0      surfaces.as_provider.mcp (mode: managed)     the door
Step 2  connection-hub@1-0  delegated_credentials.oauth.resources        the tool ceiling
Step 3  connection-hub@1-0  delegated_credentials.oauth.capabilities     who may delegate
Step 4  connection-hub@1-0  ...oauth.dynamic_client_registration         the DCR redirect fence
Step 5  connection-hub@1-0  resources[].named_services.namespaces        door claims per operation
Step 6  my-service@1-0      @named_service_provider(metadata=...)        connected-accounts contract
Step 7  runtime             all three scenarios verified
```

## Step 0 - prerequisite: connectable providers

The example realms run on user-connected accounts, so the backing providers
must be connectable under **Delegated to KDCube** first. Gmail uses the
built-in `google.oauth` adapter; your own server registers as a custom
OAuth/OIDC provider:

```yaml
# connection-hub@1-0 · config.connections.delegated_to_kdcube
delegated_to_kdcube:
  enabled: true
  oauth: { public_base_url: "https://<host>" }
  providers:
    google:
      adapter: google.oauth
      enabled: true
      connector_apps:
        gmail:
          client_id: "...apps.googleusercontent.com"
          client_secret_ref: "connections.delegated_to_kdcube.providers.google.connector_apps.gmail.client_secret"
          allowed_claims: [gmail:read, gmail:send]
      claims:
        gmail:read: { label: Read Gmail, provider_scopes: [openid, email, profile, "https://www.googleapis.com/auth/gmail.readonly"] }
        gmail:send: { label: Send Gmail, provider_scopes: [openid, email, profile, "https://www.googleapis.com/auth/gmail.send"] }
    my_server:
      adapter: custom.oauth
      enabled: true
      connector_apps:
        default:
          client_id: "<your-client-id>"
          client_secret_ref: "connections.delegated_to_kdcube.providers.my_server.connector_apps.default.client_secret"
          allowed_claims: [my_server:read, my_server:write]
      claims:
        my_server:read:  { label: "Read your server",  provider_scopes: [read] }
        my_server:write: { label: "Write your server", provider_scopes: [write] }
```

Secrets are `*_ref` pointers into `bundles.secrets.yaml`, never inline.
Provider and account mechanics are
[Delegated Provider Accounts](../../sdk/solutions/connections/delegated-accounts/delegated-accounts-README.md);
the custom-adapter contract (endpoints, claim mapping) is
[Custom OAuth/OIDC Provider Accounts](../../sdk/solutions/connections/delegated-accounts/custom-oauth-oidc-service-README.md).

## Step 1 - declare the managed MCP door

Keep the app modular: token-free domain services, named-service providers over
them, and one thin `@mcp` entrypoint method:

```text
my-service@1-0/
  entrypoint.py                 thin composition root; declares the @mcp door
  services/
    gmail_ops.py                domain logic; receives a resolved credential
    my_server_ops.py            domain logic; receives a resolved credential
  providers/
    mailbox.py                  named-service provider (namespace "mailbox")
    crm.py                      named-service provider (namespace "crm")
  config/ interface/ docs/ tests/
```

```python
# entrypoint.py
@mcp(alias="ops", route="public", transport="streamable-http",
     auth_config="surfaces.as_provider.mcp.ops.auth")
def ops_mcp(self, request=None, **kwargs):
    return build_ops_mcp_app(request=request, ...)
```

```yaml
# my-service@1-0 · config.surfaces.as_provider.mcp
mcp:
  ops:
    auth: { mode: managed, authority_id: delegated_client, selected_tool_grants: true }
```

What `mode: managed`, `authority_id`, and `selected_tool_grants` mean is
[Layer 1 of the chain](../../sdk/solutions/connections/authenticated-mcp/authenticated-mcp-README.md#layer-1---the-door-a-managed-mcp-surface).

## Step 2 - declare the delegated resource and its tool ceiling

In Connection Hub, register the door as a delegable resource with per-tool
grant requirements. This is the ceiling every issuance path stays under:

```yaml
# connection-hub@1-0 · config.connections.delegated_credentials.oauth
resources:
  - resource: '*/api/integrations/bundles/*/*/my-service@1-0/public/mcp/ops*'
    label: My service MCP
    tools:
      named_services_list:
        label: List named services
        description: List configured namespaces and operation grants.
        grants: [named_services:use]
      named_services_search:
        label: Named service search
        description: Search objects in a configured namespace.
        grants: [named_services:use]
      named_services_get:
        label: Named service get
        description: Read objects by ref.
        grants: [named_services:use]
      named_services_action:
        label: Named service action
        description: Run a bounded provider action on one object.
        grants: [named_services:use]
```

The `resource` pattern must byte-match the door URL and every consumer's
`resource` field - it is the grant key. Ceiling semantics are
[Layer 2 of the chain](../../sdk/solutions/connections/authenticated-mcp/authenticated-mcp-README.md#layer-2---the-delegated-resource-the-tool-ceiling).

## Step 3 - declare the delegable capabilities

Every grant the catalog uses (steps 2 and 5) gets a capability row: consent
label plus who may delegate it, by role AND by permission - a user qualifies
through either axis:

```yaml
# connection-hub@1-0 · config.connections.delegated_credentials.oauth
capabilities:
  - grant: named_services:use
    label: Use named services
    description: Admission to the named-services door.
    delegable_roles: [kdcube:role:registered, kdcube:role:paid, kdcube:role:privileged, kdcube:role:super-admin]
    delegable_permissions: [named_services:use]
  - grant: gmail:read
    label: Read Gmail
    description: Search and read messages from connected Gmail accounts.
    delegable_roles: [kdcube:role:registered, kdcube:role:paid, kdcube:role:privileged, kdcube:role:super-admin]
    delegable_permissions: [gmail:read]
  - grant: gmail:send
    label: Send Gmail
    description: Send mail through connected Gmail accounts.
    delegable_roles: [kdcube:role:registered, kdcube:role:paid, kdcube:role:privileged, kdcube:role:super-admin]
    delegable_permissions: [gmail:send]
  - grant: my_server:read
    label: Read your server
    delegable_roles: [kdcube:role:registered, kdcube:role:paid, kdcube:role:privileged, kdcube:role:super-admin]
    delegable_permissions: [my_server:read]
  - grant: my_server:write
    label: Write your server
    delegable_roles: [kdcube:role:privileged, kdcube:role:super-admin]
    delegable_permissions: [my_server:write]
```

A grant with no matching row cannot be delegated and has no consent label.
Filtering and the `delegated_access_grants_not_delegable` failure are
[Layer 3 of the chain](../../sdk/solutions/connections/authenticated-mcp/authenticated-mcp-README.md#layer-3---delegable-capabilities-who-may-delegate-a-grant).

## Step 4 - the DCR redirect allowlist

External MCP-speaking apps not pre-listed in `public_clients` register via
dynamic client registration, which runs before any user authenticates - the
allowlist is the only fence on where an authorization code may be delivered:

```yaml
# connection-hub@1-0 · config.connections.delegated_credentials.oauth
dynamic_client_registration:
  allowed_redirect_uris:
  - https://claude.ai/api/mcp/auth_callback
  - http://localhost/callback
  - http://127.0.0.1/callback
```

Loopback entries match any port; scheme, host, and path match exactly - see
[Layer 4 of the chain](../../sdk/solutions/connections/authenticated-mcp/authenticated-mcp-README.md#layer-4---the-registration-fence-dcr-redirect-allowlist).

## Step 5 - namespace boundary policy: door claims per operation

Step 2 admits the caller to the generic bridge tools; which grants each
namespace OPERATION consumes is the nested boundary tree on the same resource
row, checked on every call:

```yaml
# connection-hub@1-0 · same resources[] row as step 2
resources:
  - resource: '*/api/integrations/bundles/*/*/my-service@1-0/public/mcp/ops*'
    named_services:
      namespaces:
        mailbox:
          label: Mailbox
          authority_id: delegated_client
          tools:
            search:
              operation: object.search
              label: Search mailbox
              grants: [named_services:use, gmail:read]
            get:
              operation: object.get
              label: Read mailbox object
              grants: [named_services:use, gmail:read]
            action:
              operation: object.action
              label: Mailbox action
              operations:
                object.action:
                  grants: [named_services:use, gmail:read, gmail:send]
        crm:
          label: CRM
          authority_id: delegated_client
          tools:
            search:
              operation: object.search
              label: Search CRM
              grants: [named_services:use, my_server:read]
            action:
              operation: object.action
              label: CRM action
              operations:
                object.action:
                  grants: [named_services:use, my_server:write]
```

Hands-on check before you move on: **every operation's `grants` list includes
`named_services:use`** alongside its realm claim(s) - door admission is its own
consent, and each tool row states its complete requirement. Both example
realms are single-provider, so their door claims are the real provider claims
(`gmail:read`, `my_server:write`, ...); the full claim rule, including when a
provider-neutral namespace claim applies and why account-backed claims never
sit in a connection `scope`, is
[Layer 5 of the chain](../../sdk/solutions/connections/authenticated-mcp/authenticated-mcp-README.md#layer-5---namespace-boundary-policy-door-claims-per-operation).

## Step 6 - self-describe the provider: `connected_accounts` metadata

Each provider that runs on connected accounts declares which provider backs it
and which per-account claims each operation needs - in its registration
metadata, on the `@named_service_provider` decorator. Declare it once here;
nothing else hardcodes which provider backs a namespace:

```python
# providers/mailbox.py
MAILBOX_CONNECTED_ACCOUNT_REQUIREMENTS = [
    {
        "provider_id": "google",
        "provider_label": "Google",
        "claims": ["gmail:read", "gmail:send"],
        "claim_labels": {"gmail:read": "read mail", "gmail:send": "send mail"},
        "claims_by_operation": {
            "object.search": ["gmail:read"],
            "object.get": ["gmail:read"],
            "object.action.send": ["gmail:send"],
        },
    }
]

@named_service_provider(
    provider_id="my-service.mailbox",
    namespace="mailbox",
    ...,
    metadata={
        "connected_accounts": MAILBOX_CONNECTED_ACCOUNT_REQUIREMENTS,
    },
)
class MailboxNamedServiceProvider(NamedServiceProvider):
    ...
```

The `crm` provider declares the same shape for `provider_id="my_server"` with
`my_server:read` / `my_server:write`. The full contract - field meanings, the
shipped mail and slack registrations, and every surface that consumes this
metadata - is
[the provider self-description contract](../../sdk/solutions/connections/authenticated-mcp/authenticated-mcp-README.md#the-provider-self-description-contract-connected_accounts).

## Step 7 - verify all three scenarios

### a. Hosted agent, consenting in chat

Declare a resident agent as a consumer of the door. Its connection `scopes`
carry door admission only - read/write on account-backed realms is decided per
account, not per connection:

```yaml
# my-agent@1-0 · config.surfaces.as_consumer.agents.<agent>.tools[]
- name: ops
  kind: mcp
  delegated: true
  url: "https://<host>/api/integrations/bundles/<t>/<p>/my-service@1-0/public/mcp/ops"
  resource: "*/api/integrations/bundles/*/*/my-service@1-0/public/mcp/ops*"
  scopes: [named_services:use]
```

Verify, with a user who has NO Gmail account connected:

1. Ask in chat: "search my mailbox for the invoice thread". The agent attempts
   `mailbox` `object.search`; because the grantor has zero connected accounts
   on the backing provider, the CONNECT demand leads - a chat consent banner
   opens the guided plan on **Delegated to KDCube**, scoped to `gmail:read`
   (this operation's claims per `claims_by_operation`). Why connect leads is
   [the demand-ordering rule](../../sdk/solutions/connections/authenticated-mcp/authenticated-mcp-README.md#demand-ordering-connect-leads-on-zero-accounts).
2. Connect the Gmail account; the plan ends with the hand-off "Continue -
   grant it to \<agent\>", landing on the agent's card under **Delegated by
   KDCube**. Nothing is pre-checked: tick `gmail:read` for the account
   (default-closed per-account binding).
3. Retry the same ask in chat - it succeeds. The two gates this call crossed
   and their full error vocabulary are
   [the two gates at call time](../../sdk/solutions/connections/authenticated-mcp/authenticated-mcp-README.md#the-two-gates-at-call-time).

Also verify the binding is per account: bind read+send on one Gmail account
and read-only on another; a send through the read-only account is refused,
naming the allowed account.

### b. External MCP connection over OAuth/DCR

From an MCP-speaking app - Claude Code as the example:

```bash
# example door URL shape
claude mcp add --transport http ops \
  "https://<host>/api/integrations/bundles/<t>/<p>/my-service@1-0/public/mcp/ops"
```

Verify:

1. The app probes the URL, gets the protected-resource challenge, and
   registers via DCR - accepted only because its callback is on the step 4
   allowlist.
2. The authorize page shows the DCR-issued client id, the step 2 ceiling as
   collapsible capability sections, and per-account default-closed binding for
   `mailbox` and `crm` - tick claims per account, then approve.
3. The connection's card appears under **Delegated by KDCube**, editable and
   revocable; tools run from the external app, and revoking the card stops
   them on the next call.

The full journey and identity-scope choices are
[Delegate A KDCube Service To An External Client](../connections/delegate-kdcube-service-to-external-client-README.md);
the scenario summary is
[the three scenarios, end to end](../../sdk/solutions/connections/authenticated-mcp/authenticated-mcp-README.md#the-three-scenarios-end-to-end).

### c. Bounded automation token

In Connection Hub, **Delegated by KDCube -> Create automation access**:

1. Select the `My service MCP` resource; the panel offers only grants
   delegable to you (step 3 - with the capability rows above, a
   non-privileged user is not offered `my_server:write`).
2. Narrow named-service operations to an exact selection - for example
   `mailbox` `object.search` only - and set a TTL.
3. Call the door with the minted token from a script:

```bash
# example
curl -H "Authorization: Bearer $TOKEN" \
  "https://<host>/api/integrations/bundles/<t>/<p>/my-service@1-0/public/mcp/ops"
```

Verify the token reaches exactly the selected operations: `mailbox` search
succeeds (the connected account remains a separate upstream prerequisite the
panel showed), a `crm` call or a `mailbox` send is refused, and after the TTL
every call is refused.

### Cross-check: no credentials leak

Grep prompts, generated code, logs, and the executor environment for the
Google and `my_server` secret values; they must be absent everywhere - your
code received authorized requests for resolved users, never a credential.

If any verification fails, the symptom-to-fix map is
[the troubleshooting table](../../sdk/solutions/connections/authenticated-mcp/authenticated-mcp-README.md#troubleshooting).

## Read more

- [Authenticated MCP: The Full Configuration Chain](../../sdk/solutions/connections/authenticated-mcp/authenticated-mcp-README.md)
- [Expose An MCP Service From A KDCube App](../kdcube_for_agents/expose-mcp-service-README.md)
- [Connect An MCP Service To A KDCube Agent](../kdcube_for_agents/consume-mcp-service-README.md)
- [Configuring Agent Access To Services And Accounts](../../sdk/solutions/connections/configuring-agent-service-access/configuring-agent-service-access-README.md)
- [How Agents Connect to KDCube](explore-how-agents-connect-to-kdcube-README.md)
