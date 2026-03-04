# KDCube CLI

![pixel-cubes.svg](pixel-cubes.svg)

Bootstrap installer for the KDCube platform stack. This package clones the
repository (if needed) and launches the guided setup wizard.

CLI source: `services/kdcube-ai-app/kdcube_ai_app/ops/cli`

## Install

```bash
pipx install kdcube-apps-cli
```

## Run

```bash
kdcube-apps-cli
```

## What the wizard does

- Creates a **workdir** with `config/`, `data/`, and `logs/` folders
- Writes the compose env files into `config/`
- Generates frontend runtime config
- Creates local data folders for Postgres/Redis/exec workspace/bundle storage

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
3) Start compose from `deployment/docker/all_in_one_kdcube`:

```bash
docker compose --env-file /srv/kdcube-local/config/.env up -d --build
```

## Dev‑host usage (run services on host)

You can still use the CLI to bootstrap a workdir and then run services locally:

1) Run the wizard and choose a workdir.
2) Point your IDE/run configs to the generated env files in `workdir/config`.
3) The CLI also writes a dev UI config to:
   - `app/ai-app/ui/chat-web-app/public/private/config.hardcoded.json`
4) Start local infra via `deployment/docker/local-infra-stack` if needed.

See `app/ai-app/docs/service/environment/setup-dev-env-README.md` for the full
dev‑host flow.
