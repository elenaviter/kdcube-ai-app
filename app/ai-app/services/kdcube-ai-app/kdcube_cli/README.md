# KDCube CLI

![KDCube CLI](https://raw.githubusercontent.com/kdcube/kdcube-ai-app/main/app/ai-app/services/kdcube-ai-app/kdcube_cli/pixel-cubes.png)

Bootstrap installer for the KDCube platform stack. This package clones the
repository (if needed) and launches the guided setup wizard.

----

## Prerequisites

### macOS
- Python 3.9+ (Homebrew recommended)
- Git (Xcode Command Line Tools or Homebrew)
- Docker Desktop (includes Docker Compose)

### Windows
- Python 3.9+
- Git for Windows
- Docker Desktop (enable WSL2 backend)

### Linux
- Python 3.9+
- Git
- Docker Engine + Docker Compose plugin

## Install

```bash
pipx install kdcube-cli
```

Alternative (pip):

```bash
python -m pip install --user kdcube-cli
```

## Run

```bash
kdcube-setup
```

### CLI options (common)
| Option | Purpose |
|---|---|
| `--repo <url>` | Git repo URL (default: official kdcube repo). |
| `--path <repo>` | Use a local repo checkout for builds (skip cloning). |
| `--workdir <path>` | Use a specific workdir instead of `~/.kdcube/kdcube-runtime`. |
| `--reset-config` | Re‑prompt for config values without deleting files. |
| `--reset` | Alias for `--reset-config`. |
| `--clean` | Clean local Docker cache and unused KDCube images. |
| `--secrets-prompt` | Prompt for LLM keys and inject them at runtime (sidecar). |
| `--secrets-set KEY=VALUE` | Inject a secret value without prompting (repeatable). |

### Host bundle descriptor
You can point the CLI to a **bundle descriptor YAML** that defines external bundles
to preload (git repositories, refs, module entrypoints). This is useful when you
want a default bundle set different from the built‑in registry.

The wizard prompts for this as **Host bundle descriptor path**.
It is written into the workdir envs and mounted into proc at runtime.

Example:
```yaml
bundles:
  default_bundle_id: "react@2026-02-10-02-44"
  items:
    - id: "app@2-0"
      name: "Customer App"
      repo: "git@github.com:org/customer-repo.git"
      ref: "bundle-v2026.02.22"
      subdir: "service/bundles"
      module: "app@2-0.entrypoint"
```

Reference:
- https://github.com/kdcube/kdcube-ai-app/blob/main/app/ai-app/docs/sdk/bundle/bundle-ops-README.md
- https://github.com/kdcube/kdcube-ai-app/blob/main/app/ai-app/docs/service/cicd/release-descriptor-README.md

## What it installs (default)
- Repo clone: `~/.kdcube/kdcube-ai-app`
- Workdir: `~/.kdcube/kdcube-runtime`
- Docker images: pulled (release) or built (upstream)

### Quick start (new users)

1) Run `kdcube-setup`
2) Choose **release-latest** (pull prebuilt images)
3) Answer **yes** to “Run docker compose now?”

That will bring up the stack with no local build required.

### Use a local checkout (dev)

```bash
kdcube-setup --path /Users/you/src/kdcube/kdcube-ai-app
```

### Manual compose (advanced)

If you want to run compose manually, use the workdir env file:

```bash
docker compose --env-file ~/.kdcube/kdcube-runtime/config/.env up -d --build
```

Note: `--env-file` is a **Docker Compose** option (not a CLI flag).

At “Install source”:
- **upstream** → build images from your local repo
- **skip** → keep the repo as-is (no pull) and use it for build/run

If you choose **upstream**, answer **yes** to “Build core platform images?”
to rebuild from your local changes.

Re-run prompts (edit existing values):

```bash
kdcube-setup --reset
```

Clean local Docker images/cache:

```bash
kdcube-setup --clean
```

Tip: if `kdcube-setup` is not on your PATH, run:

```bash
python -m kdcube_cli.cli
```

## What the wizard does (today)

When you run `kdcube-setup`, the **wizard** performs the steps below:
1) Creates a **workdir** with `config/`, `data/`, and `logs/` folders.
2) Writes compose env files into `config/` (only if missing; it won’t overwrite existing files).
3) Copies nginx configs into `config/` for runtime overrides:
   - `nginx_ui.conf`
   - runtime proxy config (based on selected auth mode)
4) Selects **auth mode** (simple or cognito) and writes:
   - `AUTH_PROVIDER` in `.env.ingress` + `.env.proc`
   - Cognito fields when applicable (see below)
5) Generates frontend runtime config (hardcoded or cognito).
5) Creates local data folders for Postgres/Redis/exec workspace/bundle storage.
6) Optionally builds images and starts `docker compose up -d`.

### Authentication modes
The wizard prompts for an auth mode and updates both backend and frontend config.

**Simple (hardcoded)**
- `AUTH_PROVIDER=simple`
- Frontend config: `frontend.config.hardcoded.json`
- Uses a hardcoded admin token in config (local dev only)

**Cognito**
- `AUTH_PROVIDER=cognito`
- Frontend config: `frontend.config.cognito.json`
- Required fields:
  - `COGNITO_REGION`
  - `COGNITO_USER_POOL_ID`
  - `COGNITO_APP_CLIENT_ID`
  - `COGNITO_SERVICE_CLIENT_ID`
- The frontend `authority` is composed as:
  - `https://cognito-idp.<COGNITO_REGION>.amazonaws.com/<COGNITO_USER_POOL_ID>`

**Delegated**
- Shown in the selector but **disabled** for now (falls back to cognito).

### Routes prefix & nginx proxy
The frontend config includes `routesPrefix` (default: `/chatbot`).
The wizard patches the **runtime proxy config** in `config/` so nginx uses the
same prefix. This keeps `/chatbot` (or any custom prefix) consistent between UI and proxy.

### Secrets (third services tokens)
The wizard **does not** write OpenAI/Anthropic/Brave keys to `.env` files.
If you provide them during setup, they are injected at runtime into the
`kdcube-secrets` sidecar (in‑memory only). If you restart the stack, you’ll
be prompted again to re‑inject keys.

Order (automatic):
1) Start `kdcube-secrets`
2) Wait for it to be ready
3) Inject keys
4) Start/restart ingress + proc (they fetch secrets)

Manual re‑inject:

```bash
kdcube-setup --secrets-prompt --workdir ~/.kdcube/kdcube-runtime
```

Or pass explicit values:

```bash
kdcube-setup --secrets-set OPENAI_API_KEY=... --secrets-set ANTHROPIC_API_KEY=...
```

You can also override the git HTTPS token this way:

```bash
kdcube-setup --secrets-set GIT_HTTP_TOKEN=...
```

Note: re‑inject will **restart** `kdcube-secrets`, `chat-ingress`, and `chat-proc`
to refresh per‑run tokens (and the web proxy to keep upstreams in sync).
Per‑run tokens are generated by the CLI and are **not stored** in `config/`.

If you set LLM keys in env files (managed infra / custom setups), those env
values still work and take precedence. The secrets sidecar is only used when
env keys are missing.

### Git HTTPS token (private bundles)
If you choose **https-token** auth for git bundles, the token is treated as a
secret and is **not stored** in `.env.proc`. It is injected at runtime (same
flow as LLM keys). You will be prompted again on the next run unless you pass
it via `--secrets-set GIT_HTTP_TOKEN=...`.

More details:
https://github.com/kdcube/kdcube-ai-app/blob/main/app/ai-app/docs/ops/local/local-setup-README.md

Current scope: the wizard is **optimized for docker‑compose** (all‑in‑one).
It creates a workdir (default: `~/.kdcube/kdcube-runtime`) and lets you:
- generate config/data/log folders
- choose **release** (pull from DockerHub) or **upstream** (build locally)
- start `docker compose` (optional)

Install source options:
- `release-latest`: pull prebuilt images for the latest release
- `release-installed`: pull prebuilt images for the last installed release (if known)
- `release-tag`: pull prebuilt images for a specific version (platform.ref)
- `upstream`: build images from the current git checkout
- `skip`: keep current repo/workdir without pulling or changing versions

Tip: you can select the install source using the ↑/↓ arrow keys and Enter.

Example workdir layout:

```
~/.kdcube/kdcube-runtime
├─ config/
│  ├─ .env
│  ├─ .env.ingress
│  ├─ .env.proc
│  ├─ .env.metrics
│  ├─ .env.postgres.setup
│  ├─ .env.proxylogin
│  ├─ frontend.config.hardcoded.json
│  ├─ nginx_ui.conf
│  └─ nginx_proxy.conf
├─ data/
│  ├─ postgres/
│  ├─ redis/
│  ├─ kdcube-storage/
│  │  ├─ cb/
│  │  │  └─ tenants/<tenant>/projects/<project>/
│  │  │     ├─ conversation/<user>/<conversation>/<turn>/
│  │  │     └─ executions/<user>/<conversation>/<turn>/<exec_id>/
│  │  ├─ accounting/<tenant>/project/<YYYY.MM.DD>/<service>/<bundle_id>/
│  │  └─ analytics/<tenant>/project/accounting/{daily,weekly,monthly}/
│  ├─ exec-workspace/
│  └─ bundle-storage/
└─ logs/
   ├─ chat-ingress/
   └─ chat-proc/
```

## Where data is stored
- **Config:** `workdir/config/` (env files, nginx config, UI config)
- **Data:** `workdir/data/` (postgres/redis storage, bundle storage, exec workspace)
- **Logs:** `workdir/logs/`

Infra credentials (Postgres/Redis) are stored in `config/.env*` for local compose.
LLM keys are **not** stored in files; they live only in the secrets sidecar.

## Compose usage (recommended)

1) Run the wizard and choose a workdir (example: `/srv/kdcube-local`).
2) It will generate:
   - `/srv/kdcube-local/config/.env`
   - `/srv/kdcube-local/config/.env.ingress`
   - `/srv/kdcube-local/config/.env.proc`
   - `/srv/kdcube-local/config/.env.metrics`
   - `/srv/kdcube-local/config/.env.postgres.setup`
   - `/srv/kdcube-local/config/.env.proxylogin`
   - `/srv/kdcube-local/config/frontend.config.hardcoded.json`
   - `/srv/kdcube-local/config/nginx_ui.conf`
   - `/srv/kdcube-local/config/nginx_proxy.conf`
3) Start compose from `deployment/docker/all_in_one_kdcube`:

```bash
docker compose --env-file /srv/kdcube-local/config/.env up -d --build
```

Open the UI:
- `http://localhost:${KDCUBE_UI_PORT}/chatbot/chat` (via proxy, omit `:${KDCUBE_UI_PORT}` if it is `80`)

## Notes

- The wizard **does not overwrite** existing config files in your workdir. It only fills
  placeholders in newly created files.
- Use `kdcube-setup --reset` to re-enter values without deleting files.
- Config upgrades/migrations will be added later when configs are versioned.

Tip: you can edit `workdir/config/nginx_ui.conf` and `workdir/config/nginx_proxy.conf`
without rebuilding images (they are mounted into the containers at runtime).

## UI config source of truth

The web UI loads its runtime config from `/config.json` inside the `web-ui`
container. Docker compose mounts the host file defined by
`PATH_TO_FRONTEND_CONFIG_JSON` to:

`/usr/share/nginx/html/config.json`

## Clean / reset
Clean local Docker cache and unused KDCube images:
```bash
kdcube-setup --clean
```

Reset prompts without deleting files:
```bash
kdcube-setup --reset
```

Full reset (delete workdir):
```bash
rm -rf ~/.kdcube/kdcube-runtime
```

If the UI is calling the wrong tenant/project, check:
- `PATH_TO_FRONTEND_CONFIG_JSON` in the generated `.env`
- `curl http://localhost:<ui_port>/config.json`

See the full local setup flow on GitHub:
https://github.com/kdcube/kdcube-ai-app/blob/main/app/ai-app/docs/ops/local/local-setup-README.md
