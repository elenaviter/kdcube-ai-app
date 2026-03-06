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

Re-run prompts (edit existing values):

```bash
kdcube-setup --reset
```

Tip: if `kdcube-setup` is not on your PATH, run:

```bash
python -m kdcube_cli.cli
```

## What the wizard does (today)

- Creates a **workdir** with `config/`, `data/`, and `logs/` folders
- Writes the compose env files into `config/` (only if missing; it won’t overwrite existing files)
- Copies nginx configs into `config/` for runtime overrides:
  - `nginx_ui.conf`
  - `nginx_proxy.conf`
- Generates frontend runtime config
- Creates local data folders for Postgres/Redis/exec workspace/bundle storage
- Optionally builds images and runs `docker compose up -d --build`

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
│  ├─ exec-workspace/
│  └─ bundle-storage/
└─ logs/
   ├─ chat-ingress/
   └─ chat-proc/
```

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

See the dev‑host guide on GitHub:
https://github.com/kdcube/kdcube-ai-app/blob/main/app/ai-app/docs/service/environment/setup-dev-env-README.md
