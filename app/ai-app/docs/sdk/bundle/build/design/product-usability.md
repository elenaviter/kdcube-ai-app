# Terminology:

- `tenant/project` = one KDCube environment
- an environment is one isolated deployment snapshot

What that isolation includes:
- its own staged descriptors
- its own platform snapshot/version
- its own bundle props and bundle secrets
- its own user-scoped bundle state
- its own Postgres/Redis runtime data

How to interpret it:
- do not think of `tenant/project` as “one bundle”
- think of it as “one whole environment that can host many bundles”

Typical uses:
- lifecycle stages:
    - `dev`
    - `staging`
    - `prod`
- parallel isolated environments:
    - `product-a/dev`
    - `product-a/prod`
    - `product-b/dev`
    - `product-b/prod`

Why we use this snapshot model:
- one environment can stay on a known-good platform/config state
- another environment can move forward independently
- changes in one environment do not mutate runtime data or bundle state in another one

Local-machine rule:
- one machine may contain many environment snapshots on disk
- but local compose-backed runtime should still be treated as one active environment at a time by default

If you want, I can also fold this into the previous note to the CLI engineer.


# Content creator/configurator/deployer/user usability notes, links, runtime model
Main points:

1. Docs location changed
- configuration docs now live under:
    - https://github.com/kdcube/kdcube-ai-app/tree/main/app/ai-app/docs/configuration
- CLI docs now live under:
    - https://github.com/kdcube/kdcube-ai-app/blob/main/app/ai-app/docs/service/cicd/cli-README.md
    - https://github.com/kdcube/kdcube-ai-app/blob/main/app/ai-app/docs/service/cicd/design/cli--as-control-plane-README.md
    - https://github.com/kdcube/kdcube-ai-app/blob/main/app/ai-app/src/kdcube-ai-app/kdcube_cli/README.md

2. Bundle-facing runtime docs that must stay aligned with the new CLI
- current local runtime contract:
    - https://github.com/kdcube/kdcube-ai-app/blob/main/app/ai-app/docs/sdk/bundle/build/how-to-configure-and-run-bundle-README.md
- planned new CLI bundle entrypoint:
    - https://github.com/kdcube/kdcube-ai-app/blob/main/app/ai-app/docs/sdk/bundle/build/how-to-configure-and-run-bundle-new-cli-README.md
- Tier 1 navigation entry:
    - https://github.com/kdcube/kdcube-ai-app/blob/main/app/ai-app/docs/sdk/bundle/build/how-to-navigate-kdcube-docs-README.md
- bundle configuration/secrets contract:
    - https://github.com/kdcube/kdcube-ai-app/blob/main/app/ai-app/docs/configuration/bundle-runtime-configuration-and-secrets-README.md
- runtime storage/authority details:
    - https://github.com/kdcube/kdcube-ai-app/blob/main/app/ai-app/docs/configuration/runtime-configuration-and-secrets-store-README.md

3. Important CLI behavior
- one machine may hold many deployment snapshots on disk
- `tenant/project` is the environment boundary
- but local compose-backed development should default to one active local KDCube deployment at a time
- conv:so:
    - many workdirs on disk: yes
    - many concurrently running local KDCubes by default: no
- if another local deployment is already running, `start` should refuse and tell the operator what is active and what to stop first
- if we ever support true concurrent local runtimes, that should be an explicit advanced mode with separate compose naming and port allocation

4. Default environment behavior
- the design already supports defaults
- if a default deployment is configured, the user should not need to restate tenant/project explicitly
- explicit `--tenant/--project` or `--workdir` should still override defaults
- if no default exists and target resolution is ambiguous, the CLI should refuse and ask for an explicit target
- this applies to `start`, `stop`, `reload`, and `--info` style deployment-targeted commands


# Req to the target agent who we built tier 1 for - which roles with explanation we expect from it that it can execute thanks to our plugin
Target agent profile for the Build-with-KDCube plugin:

This is one planning agent, not a set of separate personas.

The plugin should let the same agent combine these task facets in one flow:

- creator
  Builds a new bundle from scratch from a product idea or feature description.

- integrator
  Wraps an existing backend, frontend, webhook, cron job, MCP server, or other user code into a KDCube bundle without rewriting the business logic unnecessarily.

- configurator
  Maps application settings into the correct KDCube scopes:
  platform/global settings and secrets, deployment-scoped bundle props and secrets, and user-scoped bundle state.

- deployer
  Wires the bundle into one KDCube environment through descriptors or CLI, understands `tenant/project` as the environment boundary, and can start, stop, inspect, and reload the
  local runtime.

- local QA
  Runs and interprets local validation:
  syntax/import checks, shared bundle suite, bundle-local tests, and direct checks of helpers/builders.

- integration QA
  Validates the bundle inside a real KDCube runtime:
  widget/browser behavior, API behavior, MCP behavior, reload/reconcile behavior, and cron/runtime-path behavior.

- document reader
  Navigates the KDCube docs efficiently, chooses the right next document, and does not waste time reading random deep docs before Tier 1.

Important:
- these are task facets, not separate agents
- one real task often combines several of them in sequence
- typical flow is:
  document reader -> local QA expectations -> creator/integrator -> configurator -> deployer -> integration QA

This is the capability set we expect the plugin to enable through the Tier 1 doc pack.
