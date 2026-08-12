---
id: website@2026-07-12/docs
title: "Website App Design"
summary: "Ownership and runtime flow for an app-hosted local KDCube website."
status: active
tags: ["design", "website", "scene", "auth"]
---

# Website App Design

## Boundary

The website app owns:

- the static website shell;
- which scene app is mounted;
- website title and presentation;
- browser handling of the standard scene config/auth messages.

The platform owns:

- active authority/provider configuration;
- resolved login and logout endpoints;
- `/profile` session truth;
- `/api/*` and the configured control-plane mount such as `/platform/*` or
  `/control/ui/*`;
- building and serving app main views.

The platform runtime owns the site registry. OpenResty forwards stable root and
alias routes to proc. Proc projects active app descriptors into Redis on
startup/config changes, and every proc worker routes requests from an immutable
in-memory catalog. No proxy regeneration, request-time Redis read, or
request-time descriptor scan is needed when site declarations change.

```text
bundles.yaml
  website.ui.main_view.site
          |
          +--> validated, versioned Redis site projection
          +--> proc hot site catalog: alias + hosts + default
          |
          +--> website public/site_config
                         |
browser ----------------+--> /api/cp-frontend-config
                         +--> /profile
                         +--> configured scene public/static

OpenResty / ---------------------> proc /api/integrations/site-root
OpenResty /{path} ---------------> proc /api/integrations/site-root/{path}
OpenResty /sites/{alias}/{path} -> proc /api/integrations/sites/{alias}/{path}
Site CDN /{path} ---------------> proc /api/integrations/site-root/{path}
```

No site selection or scene composition belongs in `assembly.yaml`. The
`proxy.route_prefix` value still belongs there. When any site is enabled, it
must be non-root so platform frontend files do not compete with site root
paths.

Root resolution is deterministic:

1. select the single site whose `hosts` entry matches the request host;
2. otherwise select the single site with `default: true`;
3. otherwise redirect to the configured platform chat route.

Duplicate aliases, ambiguous host matches, and multiple defaults are invalid
registry states and return `503` rather than selecting an arbitrary site.
If no site resolves for a non-root clean path, proc returns `404`. It does not
redirect that path to the platform frontend.

The website uses the normal main-view cache policy: entry HTML and root-level
non-hashed shell files revalidate, while content-hashed `assets/` files are
immutable. The public-content publication subsystem is not involved.

The CDN preserves `Host` or `X-Forwarded-Host`; proc uses it to select the site.
The CDN stores no catalog and may cache the returned HTML/assets according to
the platform headers.

## Viewport Ownership

The website shell fills its browser or host frame and keeps the configured
Scene in the remaining viewport below the site header. Its HTML carries the
`data-kdcube-resize-reporter="opt-out"` marker, and `#workspace-scene` has a
stylesheet-owned `height: 100% !important`. Nested widget resize messages can
therefore size their local Scene windows while the outer Scene keeps the
website viewport dimensions.
