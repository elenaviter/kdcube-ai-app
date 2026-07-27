---
id: repo:kdcube-ai-app/app/ai-app/docs/recipes/connections/integrations/google-gmail-README.md
title: "Google Gmail Integration"
summary: "Recipe for configuring a Google OAuth client for Gmail connected accounts, registering KDCube delegated-to-KDCube redirect URIs, and wiring Gmail tools through Connection Hub claims."
status: active
tags: ["recipes", "connections", "connection-hub", "google", "gmail", "oauth", "connected-accounts", "delegated-to-kdcube"]
updated_at: 2026-07-27
see_also:
  - repo:kdcube-ai-app/app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/examples/bundles/connection-hub@1-0/docs/integrations/google.md
  - repo:kdcube-ai-app/app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/examples/bundles/connection-hub@1-0/docs/integrations/README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/connections/integrations/google-sheets-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/connections/integrations/mail-named-service-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/connections/integrations/slack-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/connections/platform-authority/host-platform-authority-in-bundle-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/connections/connection-hub-solution-README.md
  - repo:kdcube-ai-app/app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/integrations/google/gmail_tools.py
  - repo:kdcube-ai-app/app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/integrations/connected_accounts.py
---
# Google Gmail Integration

Use this recipe when KDCube should let a signed-in user connect their own Gmail
account, then let KDCube tools search or send Gmail on that user's behalf.

This is the **delegated to KDCube** direction:

```text
Google/Gmail user
  -> user consents in Google OAuth
  -> Connection Hub stores the connected account credential
  -> KDCube tool resolves that credential for the current platform user
  -> tool calls Gmail API with the user's delegated Google token
```

This is different from Google as platform login, but it can use the same Google
OAuth client. The difference is in the Google console settings and in how
KDCube uses the client:

- Gmail connected accounts use OAuth code flow and need **Authorized redirect
  URIs** plus a client secret for the server-side code exchange.
- Bundle-hosted Google platform login uses Google Identity Services ID tokens
  and normally needs **Authorized JavaScript origins** for the page that renders
  the Google button.

## Flow

```text
Operator creates Google OAuth client
  Gmail API enabled
  OAuth consent screen configured
  Authorized redirect URIs registered
        |
        v
Operator configures Connection Hub
  provider: google
  connector app: gmail
  claims: gmail:read, gmail:send
        |
        v
Application tool declares required connected-account claims
  search_gmail -> gmail:read
  read_gmail_message -> gmail:read
  download_gmail_attachments -> gmail:read
  send_gmail -> gmail:send
  forward_gmail_message -> gmail:read + gmail:send
        |
        v
User opens Connection Hub -> Delegated to KDCube / Connected accounts
  clicks Gmail connect
  approves Google OAuth consent
        |
        v
Connection Hub callback stores:
  account metadata in user properties
  credential in user secrets, including refresh token when Google returns one
        |
        v
Agent/tool execution
  SDK resolver checks current user, provider, connector app, and claim
  refreshes the Google access token when needed
  Gmail tool calls Gmail API with the resolved Google token
```

## Google Cloud Configuration

1. Open Google Cloud Console.
2. Create or choose the Google Cloud project for this connector app.
3. Open **APIs & Services** -> **Library** and enable **Gmail API**.
4. Open **APIs & Services** -> **OAuth consent screen**.
5. Configure app name, support email, developer contact, and test users if the
   app is still in testing.
6. Open **APIs & Services** -> **Credentials**.
7. Create or open an OAuth 2.0 Client ID of type **Web application**.
8. Add the exact KDCube callback URLs under **Authorized redirect URIs**.

For a local demo-project runtime, use the current public host for that runtime:

```text
https://<LOCAL_PUBLIC_HOST>/api/integrations/bundles/demo-tenant/demo-project/connection-hub@1-0/public/delegated_to_kdcube_oauth_callback
```

For the custom-authority test project:

```text
https://<LOCAL_PUBLIC_HOST>/api/integrations/bundles/demo-tenant/custom-authority/connection-hub@1-0/public/delegated_to_kdcube_oauth_callback
```

For the deployed demo runtime:

```text
https://demo.kdcube.tech/api/integrations/bundles/demo/demo/connection-hub@1-0/public/delegated_to_kdcube_oauth_callback
```

For the deployed dev/staging runtime:

```text
https://dev.kdcube.tech/api/integrations/bundles/demo/demo-march/connection-hub@1-0/public/delegated_to_kdcube_oauth_callback
```

Google compares redirect URIs exactly. The public host, tenant, project, bundle
id, route, and callback path must all match the URL KDCube sends in the OAuth
request. If ngrok changes, add the new callback URL to the Google OAuth client.

9. Copy **Client ID** and **Client Secret**.

## Google OAuth Scopes

The current Gmail SDK tools use these provider scopes:

```text
openid
email
profile
https://www.googleapis.com/auth/gmail.readonly
https://www.googleapis.com/auth/gmail.send
```

`gmail:read` maps to `openid`, `email`, `profile`, and
`https://www.googleapis.com/auth/gmail.readonly`.

`gmail:send` maps to `openid`, `email`, `profile`, and
`https://www.googleapis.com/auth/gmail.send`.

If the OAuth consent screen is in testing mode, add each test user's Google
account under **Test users** or Google will block consent for them.

## Token Lifetime And Reconnect

Google access tokens are short-lived. KDCube therefore requests offline access
when the user connects Gmail:

```text
access_type=offline
prompt=consent
include_granted_scopes=true
```

When Google returns a `refresh_token`, Connection Hub stores it in the user's
secret storage together with the access token. Tool and named-service calls do
not use stale access tokens directly. The delegated-to-KDCube broker checks the
stored credential before returning it to Gmail tools and refreshes the access
token when it is expired or close to expiry.

The Connections widget shows credential health:

- **Connected** means KDCube has a usable credential.
- **Refreshes automatically** means the current access token is expired, but a
  refresh token exists and KDCube will refresh before the next provider call.
- **Reconnect required** means the credential is missing or the access token is
  expired and no refresh token is stored. The user should disconnect/reconnect
  Gmail or start the OAuth connect flow again with the needed claims.

If a user sees a Gmail API error like `invalid authentication credentials`, check
the Connections widget. If it says **Reconnect required**, reconnect the Gmail
account. If it says **Refreshes automatically**, retry the operation; the
backend should refresh the token on use.

## Connection Hub Configuration

Configure the Google provider and Gmail connector app under the Connection Hub
bundle. The `public_base_url` must be the public URL that Google can call back.

`bundles.yaml`:

```yaml
bundles:
  version: "1"
  items:
    - id: connection-hub@1-0
      config:
        connections:
          delegated_to_kdcube:
            enabled: true
            oauth:
              public_base_url: "https://<PUBLIC_HOST>"
            providers:
              google:
                label: Google
                adapter: google.oauth
                enabled: true
                connector_apps:
                  gmail:
                    label: Gmail
                    enabled: true
                    client_id: "<GOOGLE_OAUTH_CLIENT_ID>"
                    client_secret_ref: connections.delegated_to_kdcube.providers.google.connector_apps.gmail.client_secret
                    allowed_claims:
                      - gmail:read
                      - gmail:send
                claims:
                  gmail:read:
                    label: Read Gmail
                    description: Search and read Gmail messages for the approving user.
                    provider_scopes:
                      - openid
                      - email
                      - profile
                      - https://www.googleapis.com/auth/gmail.readonly
                  gmail:send:
                    label: Send Gmail
                    description: Send email through the approving user's Gmail account.
                    provider_scopes:
                      - openid
                      - email
                      - profile
                      - https://www.googleapis.com/auth/gmail.send
```

The snippet shows the Gmail claims relevant to this recipe. Preserve any other
claims already allowed on the same connector app, including `sheets:read` and
`sheets:write` when the [Google Sheets productivity recipe](google-sheets-README.md)
is configured.

`bundles.secrets.yaml`:

```yaml
bundles:
  version: "1"
  items:
    - id: connection-hub@1-0
      secrets:
        connections:
          delegated_to_kdcube:
            oauth_state_secret: "<RANDOM_HEX_32_BYTES>"
            providers:
              google:
                connector_apps:
                  gmail:
                    client_secret: "<GOOGLE_OAUTH_CLIENT_SECRET>"
```

Generate `oauth_state_secret` once per environment and keep it in secrets:

```bash
openssl rand -hex 32
```

## Tool Configuration

Tools declare the connected-account claims they need. The tool does not read
Google secrets directly and does not know where the credential is stored.

Example main-agent tool block:

```yaml
- name: gmail
  kind: python
  module: kdcube_ai_app.apps.chat.sdk.integrations.google.gmail_tools
  alias: gmail
  allowed:
    - search_gmail
    - read_gmail_message
    - download_gmail_attachments
    - send_gmail
    - forward_gmail_message
  tool_traits:
    search_gmail:
      strategy:
        - exploration
    read_gmail_message:
      strategy:
        - exploration
    download_gmail_attachments:
      strategy:
        - exploration
    send_gmail:
      strategy:
        - exploitation
    forward_gmail_message:
      strategy:
        - exploitation
  tool_claims:
    search_gmail:
      connections:
        delegated_to_kdcube:
          connected_accounts:
            - provider_id: google
              claims:
                - gmail:read
    read_gmail_message:
      connections:
        delegated_to_kdcube:
          connected_accounts:
            - provider_id: google
              claims:
                - gmail:read
    download_gmail_attachments:
      connections:
        delegated_to_kdcube:
          connected_accounts:
            - provider_id: google
              claims:
                - gmail:read
    send_gmail:
      connections:
        delegated_to_kdcube:
          connected_accounts:
            - provider_id: google
              claims:
                - gmail:send
    forward_gmail_message:
      connections:
        delegated_to_kdcube:
          connected_accounts:
            - provider_id: google
              claims:
                - gmail:read
                - gmail:send
```

`provider_id` must match the provider under
`connections.delegated_to_kdcube.providers`. Tools declare only the provider and
the claims they need - they do **not** name a connector app. At call time the
broker resolves the user's connected Google account that holds the claim,
whichever connector app it was connected through. If a tool needs several claims
(for example read + send, as `forward_gmail_message` does), it resolves each on
the same account, so the account - and the connector app it was connected
through - must cover them all. See
[Resolve a Connected Credential](resolve-connected-credential-README.md) for the
full resolution mechanism.

## Mail Named-Service Exposure

The same connected Gmail account can also be exposed to external agents through
the provider-neutral `mail` named-service namespace on the generic
`kdcube-services@1-0/public/mcp/named_services` surface.

That adds a separate delegated consent layer:

```text
Claude/external client -> KDCube grants: mail:read, mail:send
KDCube -> Gmail connected-account claims: gmail:read, gmail:send
```

Use [Mail Named Service Over MCP](mail-named-service-README.md) for the exact
namespace refs, MCP operations, Connection Hub boundary config, and Claude test
prompts.

## User Experience

1. The user signs into KDCube.
2. The user opens Connection Hub -> Connections -> Delegated to KDCube /
   Connected accounts.
3. Gmail appears as an available connected-account provider if the descriptor
   config is loaded.
4. The user starts Gmail connection and selects the claims to approve, such as
   Read Gmail or Send Gmail.
5. KDCube opens Google OAuth in a new tab.
6. The user approves the Google consent screen.
7. Google redirects back to Connection Hub.
8. Connection Hub stores the connected account metadata and credential for that
   KDCube platform user.
9. The connected Gmail account appears in the widget.
10. Agent tools can now resolve the account when the same platform user asks to
    search or send through Gmail.

## Test The Flow

After descriptor changes, refresh the runtime and open the Connection Hub
widget. Connect Gmail first, then test from an agent that has the Gmail tools.

Prompts that should reach the tools:

```text
Search Gmail for "KDCube" and show the top 5 matching messages with sender, subject, and date.
```

```text
Send an email to <recipient@example.com> with subject "KDCube Gmail test" and body "Testing KDCube Gmail connection."
```

Attachment/read checks:

```text
Find my latest email from receipts@example.com, read it, list the attachments, and download the attachments as KDCube files.
```

```text
Send an email to <recipient@example.com> with subject "Report" and attach the file at conv:fi:<turn>.files/reports/report.pdf.
```

```text
Forward the Gmail message <message-id> to <recipient@example.com> and include the original attachments.
```

Expected behavior:

- if Gmail is not connected, the tool returns a managed connected-account error
  that the chat UI can surface as a consent/connect action;
- if Gmail is connected with `gmail:read`, `search_gmail` can search;
- if Gmail is connected with `gmail:read`, `read_gmail_message` can read a
  message body and list attachment ids;
- if Gmail is connected with `gmail:read`, `download_gmail_attachments` can
  materialize message attachments as KDCube files;
- if Gmail is connected with `gmail:send`, `send_gmail` can send, including
  attachments passed as KDCube `logical_path` or `physical_path` values;
- if Gmail is connected with both claims, `forward_gmail_message` can forward a
  message and optionally include its original attachments;
- if the account lacks the needed claim, the user must reconnect or upgrade the
  connected account with that claim.

## One Google OAuth Client Or Two?

You can use one Google OAuth client for both:

- KDCube platform login with Google;
- delegated-to-KDCube Gmail connected accounts.

If you do that, configure both sets on the same Google OAuth client:

- **Authorized JavaScript origins** for platform login;
- **Authorized redirect URIs** for Gmail connected-account OAuth callbacks.

KDCube still keeps the two purposes separate in descriptors:

- `authority_registry.authorities.google.accounts.providers.google_oidc.authenticator.client_id`
  uses the client id to verify Google ID tokens for platform login;
- `connections.delegated_to_kdcube.providers.google.connector_apps.gmail`
  uses the client id and client secret to exchange OAuth codes for Gmail user
  credentials.

Using two Google OAuth clients is also valid. That is cleaner when the operator
wants platform login to remain identity-only while Gmail access goes through a
separate OAuth consent screen with sensitive Gmail scopes.

## If The Same Google Client Is Also Used For Platform Login

The Gmail connected-account flow above uses **Authorized redirect URIs**.

Bundle-hosted Google platform login uses Google Identity Services and needs
**Authorized JavaScript origins** for the runtime origins where the login page
opens. For the current environments, those origins are:

```text
https://<LOCAL_PUBLIC_HOST>
https://demo.kdcube.tech
https://dev.kdcube.tech
```

Add origins without paths. Do not put `/platform/chat` or
`/api/integrations/bundles/...` in Authorized JavaScript origins.

## Troubleshooting

### Google Says `redirect_uri_mismatch`

Add the exact callback URL shown in the Google error to **Authorized redirect
URIs** on the OAuth client.

Do not add only the host or only the bundle prefix. The complete callback path
must be registered:

```text
/api/integrations/bundles/<TENANT>/<PROJECT>/connection-hub@1-0/public/delegated_to_kdcube_oauth_callback
```

### Google Says `origin_mismatch`

That is a JavaScript-origin problem, usually from Google Identity Services
login. Add the browser origin to **Authorized JavaScript origins** on the OAuth
client:

```text
https://<PUBLIC_HOST>
```

Do not include a path.

### Gmail API Calls Fail After Consent

Check that:

- Gmail API is enabled in the Google Cloud project;
- the connected account was approved with the needed KDCube claim;
- the OAuth consent screen includes the requested Gmail scopes;
- the user is allowed to use the OAuth app, or is listed as a test user while
  the app is in testing mode.

## Storage Boundary

Connection Hub owns the connected-account registry. Account metadata is stored
as user properties and credentials are stored as user secrets using the platform
property/secret lifecycle. Application tools should use the SDK connected
account resolver and should not read Connection Hub storage or descriptors
directly.
