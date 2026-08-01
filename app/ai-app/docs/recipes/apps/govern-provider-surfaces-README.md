---
id: repo:kdcube-ai-app/app/ai-app/docs/recipes/apps/govern-provider-surfaces-README.md
title: "Govern App API, MCP, and Widget Surfaces"
summary: "Builder recipe for declaring app surfaces in code, governing their enabled state, visibility, authority, and cookie-request CSRF policy from descriptors or the Apps dashboard, and verifying the effective deployment contract."
status: active
tags: ["recipes", "app", "api", "mcp", "widget", "governance", "csrf"]
updated_at: 2026-08-01
keywords: ["surfaces.as_provider", "API visibility roles", "bundle operation CSRF", "Apps dashboard", "surface override"]
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/configuration/bundles-descriptor-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/bundle/bundle-platform-integration-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/bundle/bundle-operation-csrf-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/apps/expose-mcp-service-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/what-i-should-know-about-app-README.md
---

# Govern App API, MCP, and Widget Surfaces

Use this recipe when an app surface already exists, but one deployment needs to
change whether it is active, who may reach it, which authority it requires, or
whether a cookie-authenticated mutation needs one-time CSRF proof.

The outcome is one effective deployment contract with two authors:

```text
app code                        deployment configuration
declares the surface            narrows or overrides policy
and safe defaults               for this tenant/project
          \                     /
           +-> effective surface -> request-time enforcement
```

KDCube still uses **bundle** in literal identifiers such as `bundles.yaml`,
`bundle_id`, and `@bundle_entrypoint`. In builder-facing prose, app and bundle
refer to the same deployable unit.

## 1. Keep The Four Decisions Separate

Do not compress every surface decision into one `protected` or `private` flag.
The runtime enforces distinct questions:

| Question | Contract | Effect |
| --- | --- | --- |
| Does the surface exist? | `@api`, `@mcp`, `@ui_widget`, or another provider decorator | Undeclared code is not remotely callable through that surface family. |
| Is it active in this deployment? | `enabled.bundle` or `enabled.<kind>...` | A disabled app or resource is absent from normal serving and returns `404`. |
| Which user types and roles may reach it? | `visibility.user_types`, `visibility.roles`, and app-level `allowed_roles` | Session-level audience and raw-role checks run before app code. |
| Which authority and grants are required? | the surface `auth` block | The current or delegated authority must satisfy the declared boundary. |
| Does a browser-cookie mutation need request proof? | API-only `csrf` | An opted-in cookie-authenticated POST needs a short-lived, one-time token for that exact operation. |

`enabled` is deployment availability. `visibility` is audience policy. `auth`
is authority policy. `csrf` proves that a cookie-authenticated browser request
was intentionally issued for this operation; it is not a role or a grant.

## 2. Declare The API And Its Defaults In Code

The decorator declares the operation and portable defaults:

```python
from typing import Any

from kdcube_ai_app.infra.plugin.bundle_loader import api


@api(
    method="POST",
    alias="report_publish",
    route="operations",
    user_types=("registered",),
    roles=("kdcube:role:editor",),
    csrf=True,
)
async def report_publish(self, **payload: Any) -> dict[str, Any]:
    return await self.reports.publish(payload)
```

The current API defaults are:

- `method="POST"` and `route="operations"`;
- empty `user_types` and `roles`, meaning no restriction on that dimension;
- `csrf=False`.

When both `user_types` and `roles` are non-empty, both checks must pass.
`user_types` use the ordered platform levels (`anonymous < registered < paid <
privileged`). `roles` compare raw role ids such as
`kdcube:role:finance-team`.

`csrf=True` is valid only for a `POST` operation on the `operations` route.
Public callbacks authenticate through their own provider proof or configured
managed guard; they do not use the browser operation-CSRF exchange.

## 3. Override Only What This Deployment Changes

The exact API method path is:

```text
surfaces.as_provider.api.<route>.<alias>.<METHOD>
```

For the example above, a deployment can replace the role audience, add an
authority grant, and disable or enable CSRF without changing app code:

```yaml
bundles:
  items:
    - id: reporting@1-0
      config:
        surfaces:
          as_provider:
            api:
              operations:
                report_publish:
                  POST:
                    visibility:
                      user_types: [registered]
                      roles: [kdcube:role:publisher]
                    auth:
                      authority_id: platform
                      grants: [reports:publish]
                    csrf: true
```

Method-specific policy wins over the compact alias-level fallback:

```text
surfaces.as_provider.api.<route>.<alias>.<METHOD>.<field>
    wins over
surfaces.as_provider.api.<route>.<alias>.<field>
    wins over
the decorator default
```

Use the method-specific form whenever an alias can exist under more than one
route or HTTP method. An explicit empty visibility list is meaningful: it
removes that visibility restriction for this deployment. A missing value keeps
the code default.

Keep deployment files sparse. Do not copy every decorator default into every
descriptor. Add an override only when that environment intentionally differs.
The Apps dashboard writes a `null` reset marker when a method must resume
following the app's code default.

## 4. Switch A Surface Off Without Removing It

Availability is configured separately:

```yaml
bundles:
  items:
    - id: reporting@1-0
      config:
        enabled:
          bundle: true
          api:
            "operations.report_publish.POST": false
          mcp:
            reports: true
          widget:
            reporting: true
```

Canonical paths are:

| Surface | Enable path |
| --- | --- |
| Whole app | `enabled.bundle` |
| API | `enabled.api["<route>.<alias>.<METHOD>"]` |
| MCP | `enabled.mcp.<alias>` |
| Widget | `enabled.widget.<alias>` |
| Scheduled job | `enabled.cron.<alias>` |

Missing values mean enabled. This block is an override layer, not a second
inventory of app resources.

## 5. Make The Same Change In The Apps Dashboard

The Control Plane Apps view edits the same descriptor-backed properties:

1. Open **Apps** and select the app.
2. Open its API, MCP, widget, or scheduled-job editor.
3. Select the concrete resource. API labels include alias, method, and route.
4. Change the enabled state, visibility, or auth fields supported by that
   surface.
5. For an API `POST` on the `operations` route, use **Cookie request CSRF** to
   require or remove the one-time browser proof.
6. Save the change. Use **Reset to default** to write the method's `null`
   reset marker and return to the decorator value.

For CSRF, the dashboard shows four facts rather than only a checkbox:

```text
effective value
decorator default
descriptor path
whether an explicit override exists
```

This matters when an explicit `true` currently equals a `true` decorator
default: the effective behavior is the same, but Reset still needs to replace
the deployment value with the reset marker.

The dashboard does not show a CSRF control for MCP. MCP uses explicit protocol
credentials and its `auth` contract, not ambient browser-cookie authentication.

## 6. Follow The Browser CSRF Exchange

For a cookie-authenticated operation whose effective policy has `csrf: true`,
the browser performs two requests:

```text
GET  .../operations/report_publish/csrf
  -> {csrf_required: true, csrf_token: "...", expires_in: 600}

POST .../operations/report_publish
X-KDCube-CSRF-Token: <returned token>
  -> token is checked against subject + tenant + project + app
     + operation + method, then consumed once
```

A missing, expired, reused, or context-mismatched token receives `403`. Shared
state unavailability receives `503`; the mutation does not fall back to an
unprotected path.

An explicit bearer or ID-token request remains compatible without this browser
exchange because its non-ambient credential is already the request proof. This
does not bypass `visibility`, `auth`, or product-level authorization.

Do not enable `csrf` speculatively across every POST. A client that uses a
platform session cookie must implement the token exchange before its operation
is opted in. Inventory the app's cookie-capable mutations and make an explicit
protected-or-exempt decision for each one.

## 7. Govern MCP And Widgets By Their Own Contracts

MCP, API, and widget surfaces share descriptor ownership, but not identical
policy fields.

| Surface | Deployment controls |
| --- | --- |
| API | enabled state; `visibility.user_types`; `visibility.roles`; optional `auth`; operation-POST `csrf` |
| MCP | enabled state; transport; `auth.mode` and its app-owned or managed policy |
| Widget | enabled state; `visibility.user_types`; `visibility.roles`; optional `auth` |
| App listing | `enabled.bundle`; `surfaces.as_provider.bundle.visibility.allowed_roles` |

`@mcp(...)` does not accept proc-side `user_types` or `roles`. Use the MCP
`auth` block:

```yaml
surfaces:
  as_provider:
    mcp:
      reports:
        auth:
          mode: managed
          authority_id: delegated_client
          tools:
            report_read:
              grants: [reports:read]
            report_publish:
              grants: [reports:publish]
          selected_tool_grants: true
```

With `mode: managed`, KDCube checks the delegated bearer, concrete resource,
selected tool, and required grants before app MCP code runs. With
`mode: bundle`, the app owns credential verification. An intentionally public
MCP surface omits a credential guard only when its tools are genuinely public.

## 8. Verify The Effective Surface, Not Only The YAML

After applying or saving policy:

1. Reload the app or wait for the live property update to reach every worker.
2. Reopen the Apps editor and confirm the effective value and override marker.
3. Test one permitted and one denied user/role for API or widget visibility.
4. Test authority/grant denial separately from visibility denial.
5. For a CSRF-protected API, verify missing token, valid token, and token reuse.
6. For MCP, run discovery and a real tool call through the selected auth mode.
7. Disable one resource and verify it returns `404` while sibling resources
   remain available.
8. Reset an override and verify the decorator default becomes effective again.

For local source changes, remember that the runtime executes its staged copy of
the platform and app. Refresh or reload the staged source before judging the
result.

## Done Means

- The decorator and app docs declare the surface that actually exists.
- The descriptor contains only intentional deployment differences.
- Enabled state, audience visibility, authority, and request proof remain
  separate decisions.
- API policy is route- and method-specific where ambiguity is possible.
- Cookie-authenticated mutations either implement the CSRF exchange or remain
  explicitly exempt.
- MCP policy uses MCP auth rather than an API CSRF or role shortcut.
- The effective behavior was verified over the real transport.

## Read Next

- [Bundle Descriptor](../../configuration/bundles-descriptor-README.md)
- [Bundle Platform Integration](../../sdk/bundle/bundle-platform-integration-README.md)
- [Bundle Operation CSRF](../../sdk/bundle/bundle-operation-csrf-README.md)
- [Expose an MCP Service](expose-mcp-service-README.md)
- [What I Should Know Before Writing a KDCube App](../what-i-should-know-about-app-README.md)
