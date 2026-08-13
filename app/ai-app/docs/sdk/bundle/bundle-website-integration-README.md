---
id: repo:kdcube/app/ai-app/docs/sdk/bundle/bundle-website-integration-README.md
title: "Bundle Website Integration"
summary: "App package contract for building one main view and optionally registering it as one complete KDCube-hosted website."
status: current
tags: ["sdk", "bundle", "website", "main-view", "site"]
updated_at: 2026-08-13
keywords: ["bundle website", "app main view site", "one website per app", "ui main", "site registration"]
see_also:
  - repo:kdcube/app/ai-app/docs/arch/application-hosted-websites-README.md
  - repo:kdcube/app/ai-app/docs/sdk/solutions/sites/application-sites-README.md
  - repo:kdcube/app/ai-app/docs/recipes/components/website-README.md
  - repo:kdcube/app/ai-app/docs/sdk/bundle/bundle-client-ui-README.md
  - repo:kdcube/app/ai-app/docs/sdk/bundle/ui-components-lifecycle-README.md
---

# Bundle Website Integration

An app can build one primary `ui.main_view` and register that built tree as one
complete website. Site registration adds routes to the existing main-view
artifact; it does not create another frontend build.

## Current Cardinality

```text
app entrypoint
  -> zero or one effective ui.main_view configuration
  -> optional at most one @ui_main surface declaration
  -> optional one ui.main_view.site registration
```

The site registry reads one effective `ui.main_view.site` object. A
configuration-backed website does not require an `@ui_main` method; the
reference website app is built this way. If the app also declares its main UI
as a code surface, the manifest loader permits only one `@ui_main` method.
`hosts` is plural because several names may select the same website. It does
not create several websites inside the app.

To publish independently built websites, use independently addressable apps:

```text
docs app       -> alias docs       -> docs file tree
workspace app  -> alias workspace  -> workspace file tree
admin app      -> alias admin      -> admin file tree
```

## Package Shape

```text
my-site@1/
  entrypoint.py
  ui/site/
    index.html
    ...assets
  config/bundles.template.yaml
  interface/
  docs/
  tests/
```

The effective app configuration supplies `ui.main_view.src_folder` and
`build_command`. This configuration is sufficient for the build and site
registration. Use the optional `@ui_main` declaration only when the app also
exposes a main-UI method in its code surface. Every buildable widget remains a
separate `@ui_widget(alias=...)`; apps may have many widgets even though they
have only one main view.

## Registration

```yaml
ui:
  main_view:
    src_folder: ui/site
    build_command: npm run build -- --outDir <VI_BUILD_DEST_ABSOLUTE_PATH>
    site:
      enabled: true
      alias: workspace
      default: false
      hosts:
        - workspace.localhost
        - workspace.example.com
```

| Field | App contract |
| --- | --- |
| `enabled` | Register this app's already-built main view as a site. |
| `alias` | Unique installation-wide address under `/sites/{alias}/`. |
| `default` | Make this site the one fallback at `/` after no host matches. |
| `hosts` | Exact or wildcard request hosts that select this same site at `/`. |

Use relative frontend asset URLs. For Vite, set `base: './'`; the platform
injects a route-aware base so the same artifact works at root, by alias, and
through the canonical public-static app route.

## Responsibilities

The app owns its files, client router, composition, APIs, and declared site
configuration. KDCube owns building, static storage, catalog validation,
routing, cache headers, platform browser metadata, and session integration.
DNS, TLS, public domains, and tunnels remain deployment responsibilities.

The site should read:

```text
/api/cp-frontend-config  platform route and provider-neutral auth contract
/profile                 current authenticated browser-session truth
app public API           site-specific composition and app identity
```

Serving a public shell does not make protected app data public. Operations,
MCP, files, streaming, and events retain their own surface guards.

## Lifecycle

```text
source or build-affecting props change
  -> normal app deployment/reload lifecycle
  -> main-view signature check and build when needed
  -> atomic UI artifact activation
  -> site catalog reconciliation from current app configuration
  -> alias/host requests resolve the active app target
```

`ui.main_view.site` uses the normal main-view artifact and does not maintain a
second copy. Site descriptor changes do not require application-specific
OpenResty routes because the proxy forwards stable root, clean-path, and
`/sites/*` families to proc.

## Builder Checklist

- [ ] The app has no more than one `@ui_main` declaration when that optional code surface is used.
- [ ] The main-view build writes a complete tree including `index.html`.
- [ ] Assets use relative paths and work under `/sites/{alias}/`.
- [ ] The site alias is unique and is not `_root`.
- [ ] `hosts` is a YAML list when several hosts select the site.
- [ ] At most one app in the deployment has `default: true`.
- [ ] `proxy.route_prefix` is non-root when any site is enabled.
- [ ] Browser auth/config comes from platform endpoints rather than embedded deployment values.
- [ ] Direct, alias, matching-host, unmatched-host, reserved-route, and asset requests are tested through the real proxy.

Follow the [website recipe](../../recipes/components/website-README.md) for the
complete local and public setup. Read
[Application-Hosted Sites](../solutions/sites/application-sites-README.md) for
the catalog and request-time implementation.
