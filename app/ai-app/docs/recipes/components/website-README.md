---
id: repo:kdcube-ai-app/app/ai-app/docs/recipes/components/website-README.md
title: "Application-Hosted Website"
summary: "Build an app-owned website, register it by alias and host, and serve it through the KDCube runtime."
status: current
tags: ["recipe", "website", "application", "main-view", "routing", "authentication"]
updated_at: 2026-08-12
keywords:
  [
    "application hosted website",
    "website app",
    "site registry",
    "host routing",
    "ui main view",
    "sites alias",
  ]
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/sites/application-sites-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/bundle/bundle-client-ui-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/bundle/ui-components-lifecycle-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/configuration/bundles-descriptor-README.md
---

# Recipe: Application-Hosted Website

Use this recipe when an app should own a complete website shell and KDCube
should serve it alongside platform, API, MCP, Event Bus, and widget routes.

An application-hosted website is a normal app `ui.main_view`. The app owns its
HTML, presentation, and composition. The platform owns building, storage,
routing, authentication metadata, and static delivery.

```text
browser
  /                              host match, then default site
  /sites/{alias}                 direct site address
  /sites/{alias}/{path}          site route with SPA fallback
  /platform/*                    platform frontend fixture
  /control/ui/*                  multi-segment platform frontend fixture
  /api/*                         platform and app APIs
          |
          v
OpenResty stable forwarding
          |
          v
proc site registry
  active app registry + authoritative bundles.yaml props
          |
          v
standard ui.main_view build and static-serving lifecycle
```

No website selection belongs in `assembly.yaml`. The CLI stages descriptors but
does not interpret site registration or generate application-specific proxy
routes. `assembly.yaml` still owns `proxy.route_prefix`, and that route prefix
must be non-root when any application-hosted site is enabled.

## 1. Add A Main View

A website app follows the normal app package contract:

```text
website@2026-07-12/
  entrypoint.py
  ui/site/
    index.html
    site.js
    styles.css
  config/
    bundles.template.yaml
    bundles.secrets.template.yaml
  interface/
  docs/
  tests/
```

Declare the main-view source and build command in the entrypoint defaults:

```python
from typing import Any, Dict

from kdcube_ai_app.apps.chat.sdk.solutions.chatbot.entrypoint import BaseEntrypoint
from kdcube_ai_app.infra.plugin.bundle_loader import api, bundle_entrypoint, bundle_id


@bundle_entrypoint(name="website", version="2026.07.12", priority=10)
@bundle_id(id="website@2026-07-12")
class WebsiteEntrypoint(BaseEntrypoint):
    def configuration_defaults(self) -> Dict[str, Any]:
        return {
            "ui": {
                "main_view": {
                    "src_folder": "ui/site",
                    "build_command": (
                        "cp index.html site.js styles.css "
                        "<VI_BUILD_DEST_ABSOLUTE_PATH>/"
                    ),
                    "site": {
                        "enabled": False,
                        "alias": "workspace",
                        "default": False,
                        "hosts": [],
                    },
                }
            }
        }

    @api(method="GET", alias="site_config", route="public")
    async def site_config(self, **kwargs: Any) -> Dict[str, Any]:
        del kwargs
        identity = self.runtime_identity()
        spec = getattr(self.config, "ai_bundle_spec", None)
        application_id = str(getattr(spec, "id", None) or "").strip()
        site = self.bundle_prop("ui.main_view.site", {}) or {}
        return {
            "application_id": application_id,
            "site_alias": str(site.get("alias") or "").strip(),
            "tenant": str(identity.get("tenant") or "").strip(),
            "project": str(identity.get("project") or "").strip(),
            "platform_config_url": "/api/cp-frontend-config",
            "profile_url": "/profile",
        }
```

Read the runtime application id from `config.ai_bundle_spec.id`. Do not create a
second hardcoded app-id constant for runtime behavior.

For a Vite website, use `base: './'` and build into
`<VI_BUILD_DEST_ABSOLUTE_PATH>`. Relative assets continue working when the same
site is served at root, by alias, or through its canonical app static route.

## 2. Register The Site

Enable and route the site in that app's `bundles.yaml` entry:

```yaml
- id: website@2026-07-12
  name: Website
  singleton: false
  config:
    ui:
      main_view:
        src_folder: ui/site
        build_command: >-
          cp index.html site.js styles.css
          <VI_BUILD_DEST_ABSOLUTE_PATH>/
        site:
          enabled: true
          alias: workspace
          default: true
          hosts:
            - workspace.example.com
          title: KDCube Workspace
          scene_application_id: workspace@2026-03-31-13-36
```

| Field | Meaning |
| --- | --- |
| `enabled` | Register the already-built public main view as a site. |
| `alias` | Required unique key used by `/sites/{alias}`. `_root` is reserved. |
| `default` | Use this site at `/` when no host declaration matches. At most one enabled site may be default. |
| `hosts` | Optional exact hosts or wildcard entries such as `*.preview.example.com`. |
| Other fields | App-owned composition data. The runtime ignores it unless the app uses it. |

Many apps may register sites. Duplicate aliases, multiple defaults, or multiple
sites matching one host are invalid registry states. Proc returns `503` instead
of selecting an arbitrary app.

Do not combine an enabled site with `proxy.route_prefix: /`. That descriptor
shape is rejected because root clean paths must have one owner. Use a non-root
control-plane mount such as `/platform` or `/control/ui`, or disable the site.

## 3. Use Platform Authentication Transparently

The website must not embed Cognito, custom-authority, cookie, or login endpoint
configuration. Load the active browser contract from the backend:

```text
website public/site_config
  -> platform_config_url
       -> /api/cp-frontend-config
  -> profile_url
       -> /profile
```

Use `/profile` as session truth. Showing a user from OIDC browser state while
`/profile` reports anonymous creates an incoherent site.

For login:

1. use `auth.loginUrl` from `/api/cp-frontend-config` when present;
2. otherwise open the configured platform frontend, which owns its active login
   flow;
3. pass the current site path as `next` when the login endpoint supports it;
4. re-check `/profile` after the login flow.

For logout, use `auth.logoutUrl` from the same config and then re-check
`/profile`. This keeps one website implementation valid for Cognito and
application-hosted platform authorities.

The website shell is public. User data and actions remain protected by their
API, MCP, widget, and event-surface guards.

## 4. Host A Scene Or Other App Surface

A website may host a scene in an iframe. The reference app reads
`scene_application_id`, mounts that app's `public/static` route, answers the
standard `CONFIG_REQUEST` with `CONFIG_RESPONSE`, and announces session changes
through `kdcube-auth-changed`.

```text
website shell
  fetch site config + platform config + profile
  mount scene iframe
  relay runtime config
    origin
    tenant/project
    scene app id
    active auth contract
  relay authentication changes
```

The website is the host; the scene and its widgets continue to own their
surface behavior.

## 5. Understand Routing And Caching

OpenResty contains no application list. Its stable routes are:

```text
/                  -> proc /api/integrations/site-root
/<path>            -> proc /api/integrations/site-root/<path>
/sites/{alias}/*   -> proc /api/integrations/sites/{alias}/*
```

At startup and after application/config updates, proc validates the current
descriptor declarations and publishes a generated site catalog to Redis. Every
proc subscribes to catalog generations and keeps an immutable copy in memory.
Requests resolve that hot copy and do not access Redis or `bundles.yaml`.
Descriptor reloads can therefore add, remove, or remap sites without
regenerating proxy configuration.

```text
bundles.yaml
    -> versioned Redis projection + update event
    -> proc in-memory SiteCatalog
    -> request-time alias/host lookup
```

For a multipage site, include the complete output tree in the main-view build.
Existing files and directory `index.html` files are served directly; an unknown
path falls back to the root `index.html` for SPA routers.

If no site resolves, root `/` redirects to the configured control-plane chat
route, for example `/platform/chat` or `/control/ui/chat`. A non-root clean path
such as `/guide` returns a controlled `404` when no site resolves; it does not
fall back to the platform frontend.

For a dedicated CDN hostname, add the hostname to `site.hosts` and configure
the CDN origin behavior to preserve that host and rewrite the viewer path:

```text
https://docs.example.com/<path>
  -> /api/integrations/site-root/<path>
  -> host-selected application site
```

The CDN is not a catalog owner and does not query Redis. It only forwards and
caches. Keep HTML revalidating and allow content-hashed `assets/` to use the
immutable cache headers returned by the platform.

The standard cache policy applies:

- entry HTML and root-level non-hashed files: `Cache-Control: no-cache`;
- nested non-hashed files: one hour;
- content-hashed files under `assets/`: one year and `immutable`.

Do not register a website shell with the public-content publication subsystem
merely to obtain caching. Public content and application static delivery solve
different problems.

## 6. Build And Test Locally

Platform and proxy code must be rebuilt the first time this capability is
introduced:

```bash
kdcube refresh \
  --tenant <tenant> \
  --project <project> \
  --path <kdcube-ai-app-repo> \
  --build
```

Then test direct and root routes against the configured proxy port:

```bash
curl -sS -o /dev/null -w '%{http_code}\n' \
  http://localhost:<proxy-port>/sites/workspace

curl -sS -o /dev/null -w '%{http_code}\n' \
  http://localhost:<proxy-port>/

curl -sS -o /dev/null -w '%{http_code}\n' \
  http://localhost:<proxy-port>/sites/does-not-exist
```

Expected results are `200`, `200`, and `404`. Platform UI must remain reachable
at its configured route prefix, for example `/platform/chat`. Repeat the same
check with a multi-segment route prefix such as `/control/ui/chat` when testing
mount-sensitive changes.

After the stable platform routes exist, descriptor-only site changes use the
normal app reload flow. Test host selection without DNS by overriding `Host`:

```bash
curl -sS -H 'Host: workspace.local.test' \
  -o /dev/null -w '%{http_code}\n' \
  http://127.0.0.1:<proxy-port>/
```

## 7. Cloud Deployment

The ECS Cognito and delegated-auth OpenResty templates use the same stable
routes. Terraform continues selecting the proxy template by platform authority
type; it does not need application-specific website logic.

Add site declarations to the environment's `bundles.yaml` and publish the
descriptor through the normal deployment procedure. Domain-based selection
requires those domains to reach the KDCube runtime, appear in the site's
`hosts` list, and preserve the viewer host. For full multipage host routing,
rewrite `/<path>` to `/api/integrations/site-root/<path>` at the CDN behavior,
as described above.

## Diagnostics

| Symptom | Check |
| --- | --- |
| `/` still redirects directly to platform chat | The web-proxy container is using an older image/config; rebuild and recreate it. |
| `/sites/{alias}` returns `404` | Site is disabled, alias differs, or app descriptor was not reloaded. |
| A clean path without a site opens platform chat | Root and clean-path fallback are mixed; clean path without a resolved site should be `404`. |
| `/` returns `503` | Inspect duplicate aliases, multiple defaults, or overlapping host declarations. |
| Catalog rejects the descriptor with `proxy.route_prefix is '/'` | Move the platform frontend to a non-root route prefix before enabling a site. |
| Site HTML loads but assets fail | Build emitted root-relative URLs; use relative URLs or Vite `base: './'`. |
| Site shows authenticated UI but APIs reject the user | Treat `/profile` as truth; do not infer login from client OIDC state alone. |
| Login returns to the wrong site | Pass the current site path through the configured login endpoint's `next` parameter. |
| Recent root-level JavaScript appears stale | Confirm the response has `Cache-Control: no-cache` and reload through the app static lifecycle. |

## Related Documentation

- [Application-Hosted Sites](../../sdk/solutions/sites/application-sites-README.md)
- [Bundle Client UI](../../sdk/bundle/bundle-client-ui-README.md)
- [UI Components Lifecycle](../../sdk/bundle/ui-components-lifecycle-README.md)
- [Bundles Descriptor](../../configuration/bundles-descriptor-README.md)
- [How To Write An App](../../sdk/bundle/build/how-to-write-bundle-README.md)
- [Scene Recipe](scene-README.md)
- Reference app: `sdk/examples/bundles/website@2026-07-12`
