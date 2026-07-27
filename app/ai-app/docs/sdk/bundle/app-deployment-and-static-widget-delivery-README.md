---
id: repo:kdcube-ai-app/app/ai-app/docs/sdk/bundle/app-deployment-and-static-widget-delivery-README.md
title: "App Deployment And Static Widget Delivery"
summary: "Experimental fleet-coordinated app deployment and policy-aware static widget serving from local or shared filesystem storage."
tags: ["sdk", "bundle", "app", "deployment", "widget", "static-ui", "authorization", "efs"]
keywords: ["on_app_deploy", "static widget delivery mode", "predeployed widget", "policy manifest", "legacy shadow deployed", "local filesystem", "EFS", "role protected widget"]
updated_at: 2026-07-27
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/bundle/ui-components-lifecycle-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/bundle/bundle-lifecycle-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/bundle/bundle-widget-integration-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/bundle/surfaces/as-provider-surfaces-README.md
---
# App Deployment And Static Widget Delivery

KDCube can prepare app (bundle) widget files before a browser requests them and
then serve those files without importing or instantiating the app on the request
path. This path is experimental and runs beside the established request-time
loader.

The design keeps authorization in proc. Built files are not copied to a public
object-store origin, because widget surfaces may require arbitrary roles,
registered user types, or authority grants.

## Select The Mode

Configure the mode in `assembly.yaml`:

```yaml
platform:
  services:
    proc:
      bundles:
        static_widget_delivery_mode: shadow
```

| Mode | Deployment behavior | Request behavior |
|---|---|---|
| `legacy` | No new deployment manifest is produced. | The existing workflow-resolving, source-signature-checking path serves the current build and rebuilds only when output is missing or stale. |
| `shadow` | Startup and live app updates reconcile widget builds and publish manifests. | The existing path still serves every request. Responses carry `X-KDCube-Widget-Delivery: legacy-shadow`. |
| `deployed` | The same deployment pipeline runs. | Proc first serves an authorized current manifest; a missing/stale artifact falls back to the legacy path. |

An absent setting means `legacy`. Use `shadow` to validate a deployment before
switching that environment to `deployed`.

## What This Optimizes

Legacy does **not** rebuild an unchanged widget on every request. It already
stores build output in bundle storage and coordinates actual npm/Vite work by
signature. The normal HTML request nevertheless travels through the app
runtime to prove that stored output is still current. Depending on the app, it
may:

- resolve config and secrets and construct a request communication context;
- import or resolve the app module and instantiate a non-singleton entrypoint;
- perform lifecycle and authority-registration bookkeeping;
- discover the decorated widget interface and apply effective app props;
- walk the widget and shared-source trees to compute the current UI signature;
- enter the process-local and shared-storage build coordinators, which normally
  short-circuit when the signature is unchanged.

Static asset requests skip the source-signature walk, but they still enter the
workflow-resolution and policy-discovery path before serving the file.

`deployed` moves app execution, source inspection, and build coordination to
startup or app-update deployment. A current request reads the small manifest,
checks it against the active registry and descriptor-props generation, applies
the recorded authorization policy, and serves the built file. This reduces
per-request CPU, source/EFS metadata reads, non-singleton construction, and the
tail risk that a browser request becomes the owner or waiter of reconciliation
work.

The deployed path is not a public zero-cost file server. It still reads current
registry/config state and enforces authorization in proc. Its local median gain
may be small; the principal goal is a shorter, bounded request path and moving
generation work away from page load.

## Deployment Flow

```text
registry generation or props update
              |
              v
     import app + on_bundle_load       per proc process
              |
              v
 shared filesystem generation lock    local filesystem or EFS
              |
              v
         on_app_deploy                 once per shared generation
              |
              v
 reconcile configured widget UI builds
              |
              v
 atomically publish policy manifest
```

`on_app_deploy` is an optional async lifecycle hook. It is for idempotent work
that must finish once for one source/configuration generation before static
surfaces become current. `on_bundle_load` remains the per-process initialization
hook. Request-local state belongs in neither hook.

The coordinator uses the existing shared-filesystem lock and heartbeat
primitive. A crashed owner leaves no current signature; another proc can retry.

## Published State

Each app receives this generated state under its normal bundle storage root:

```text
<bundle-storage>/<tenant>/<project>/<app-id>/
  ui/widgets/<alias>/...
  .kdcube.app-deployment/
    static-widget-surfaces.v1.json
    static-widget-surfaces.v1.signature
```

Local deployments use their configured local bundle-storage path. ECS
deployments use the shared EFS mount already configured as bundle storage. This
feature has no S3 dependency.

Proc-facing deployment storage and descriptor operations are async. Local
filesystem and EFS calls run outside the proc event loop because Python does not
provide native asynchronous filesystem operations.

The JSON manifest contains:

- app source generation and descriptor-props fingerprint;
- resolved widget `user_types`, `roles`, and authority/grant policy;
- enabled/static state and the relative artifact directory;
- build and deployment signatures.

It contains no app secrets, user credentials, or provider tokens.

## Deployed Request Path

For `static_widget_delivery_mode: deployed`, proc handles a widget request in
this order:

1. Resolve the app in the active tenant/project registry.
2. Read the small generated manifest from bundle storage.
3. Match its source generation and descriptor-props fingerprint to current
   authority state.
4. Enforce app-level roles plus the widget user-type, role, enabled, and
   authority-grant policy used by the legacy route.
5. Serve `index.html` or a built asset from the declared widget directory.

A policy denial returns `403` or `404`; it does not fall through. A missing or
stale manifest falls back to the legacy route, which resolves the app and
serves the current build or performs a coordinated build when the output is
missing/stale. The response identifies the path used:

```text
X-KDCube-Widget-Delivery: deployed
X-KDCube-Widget-Delivery: legacy-fallback
X-KDCube-Widget-Delivery: legacy-shadow
X-KDCube-Widget-Delivery: legacy
```

Successful deployed responses also carry a short
`X-KDCube-App-Deployment` signature.

Authenticated widget assets use private browser caching so a shared proxy
cannot replay one caller's role-authorized response to another caller. Only
the explicit `/public/widgets/...` route emits public cache directives.

## Startup And Live Changes

`shadow` and `deployed` imply startup preload even when
`bundles_preload_on_start` is false. The deployment pipeline also runs after:

- a `bundles.update` event changes an app source definition;
- a `bundles.props.update` event changes effective app configuration;
- an explicit reload republishes the active descriptor state.

The preload generation includes the app source fingerprint, descriptor-props
fingerprint, and runtime release identity (`PLATFORM_REF`, `APP_IMAGE_TAG`, or
`IMAGE_TAG` when supplied). A refreshed local source tree, changed app policy,
or new runtime image therefore cannot reuse an older deployment completion
marker merely because the app id and path stayed the same.

Before a live source reload, proc removes the published manifest pointer. Built
files remain intact, so legacy fallback stays available while the new
generation is prepared. Publication is atomic after the build and lifecycle
hook succeed.

Unlike the legacy HTML route, `deployed` does not poll the source tree for edits
on every browser request. A local source edit under an unchanged app path
becomes current through the normal app reload or runtime refresh, which starts
a new deployment generation. Descriptor and registry updates use their
existing update events. This is the deliberate exchange: deterministic
deployment invalidation replaces request-time source scanning.

## Verification

1. Set `shadow`, refresh the runtime, and load a widget. Confirm the header is
   `legacy-shadow` and the manifest exists under bundle storage.
2. Set `deployed`, refresh, and load the same widget. Confirm the header is
   `deployed` and the request does not trigger workflow construction,
   source-signature scanning, or build coordination.
3. Test a widget restricted to a custom role with an allowed and a denied user.
   The denied user must receive `403` in both modes.
4. Change widget visibility or app props and apply them. Until the new manifest
   is current, the request may say `legacy-fallback`; afterward it says
   `deployed` and enforces the new policy.
5. Set `legacy` to return immediately to the established serving path.
