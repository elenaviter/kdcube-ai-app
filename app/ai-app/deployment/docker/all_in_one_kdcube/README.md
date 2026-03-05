# All-in-one KDCube (local dev)

This compose stack runs a full local KDCube environment with **managed local infra + UI + proxy** so you can develop and evaluate bundles quickly.

**Runs**
- Postgres + pgvector (`postgres-db`)
- Postgres schema bootstrap (`postgres-setup`)
- Redis (`redis`)
- ClamAV (`clamav`)
- Chat ingress (`chat-ingress`)
- Chat processor (`chat-proc`)
- Metrics service (`metrics`)
- Web UI (`web-ui`)
- OpenResty reverse proxy (`web-proxy`)
- Optional `proxylogin` (commented out by default)

## Quick start

1. Copy sample envs and edit as needed (recommended layout: `config/`):

```bash
mkdir -p ./config
cp sample_env/.env ./config/.env
cp sample_env/.env.postgres.setup ./config/.env.postgres.setup
cp sample_env/.env.ingress ./config/.env.ingress
cp sample_env/.env.proc ./config/.env.proc
cp sample_env/.env.metrics ./config/.env.metrics
# Nginx configs (mounted into containers at runtime)
cp nginx/conf/nginx_ui.conf ./config/nginx_ui.conf
cp nginx/conf/nginx_proxy.conf ./config/nginx_proxy.conf
# Optional (if you enable proxylogin):
# cp sample_env/.env.proxylogin ./config/.env.proxylogin
```

2. Ensure bundle/exec paths are set (used by `chat-proc`):

- `HOST_BUNDLES_PATH` = host folder that contains all bundles
- `AGENTIC_BUNDLES_ROOT` = bundle root inside container (usually `/bundles`)
- `HOST_BUNDLE_STORAGE_PATH` = host folder for shared bundle local storage
- `BUNDLE_STORAGE_ROOT` = shared bundle local storage root inside container (e.g. `/bundle-storage`)
- `HOST_EXEC_WORKSPACE_PATH` = host exec workspace

3. Pick the OpenResty config you want (set in `.env`):

- `nginx/conf/nginx_proxy.conf` (no TLS, no proxylogin)
- `nginx/conf/nginx_proxy_ssl.conf` (TLS, no proxylogin)
- `nginx/conf/nginx_proxy_ssl_delegated_auth.conf` (TLS + proxylogin)

You can swap the proxy config by copying the desired file into
`./config/nginx_proxy.conf` (no rebuild required).

4. Frontend config examples live in:

- `frontend/config.json` (simple)
- `frontend/config.cognito.json`
- `frontend/config.hardcoded.json`

These are mounted at runtime as `/usr/share/nginx/html/config.json` via
`PATH_TO_FRONTEND_CONFIG_JSON` in `.env`.

5. Start the stack:

```bash
docker compose --env-file ./config/.env up -d --build
```

Open the UI:
- `http://localhost/chatbot/chat` (via proxy)
- `http://localhost:5173/chatbot/chat` (direct web-ui)

## Prepare data directories

```shell
mkdir -p ./data/{postgres,redis,clamav-db,neo4j/{data,logs,plugins,import},bundle-storage}
```

```shell
chmod -R 0777 data
chmod -R 0777 logs
```

## Ports (defaults)

- Ingress API: `8010`
- Processor API: `8020`
- Metrics: `8090` (bound to localhost)
- Web UI: `5173`
- OpenResty: `80` / `443`

## Bundles

`chat-proc` **mounts bundles from the host**:

- `HOST_BUNDLES_PATH` (host) → `AGENTIC_BUNDLES_ROOT` (container)

**Knowledge space storage (doc/knowledge bundles):**

- `HOST_BUNDLE_STORAGE_PATH` (host) → `BUNDLE_STORAGE_ROOT` (container)

This is how bundle code becomes available to the processor in this setup.

**Optional release descriptor (recommended):**

- Set `HOST_BUNDLE_DESCRIPTOR_PATH` in `.env` (host path to `release.yaml`)
- Inside container it mounts to `/config/release.yaml`
- In `.env.proc`, set:
  - `AGENTIC_BUNDLES_JSON=/config/release.yaml`

If you leave `HOST_BUNDLE_DESCRIPTOR_PATH` unset, `/dev/null` is mounted and the loader
falls back to inline `AGENTIC_BUNDLES_JSON` or the Redis registry.

## Notes

- `postgres-setup` runs once after Postgres is healthy and creates schemas.
- `pgadmin` requires `PGADMIN_DEFAULT_EMAIL` and `PGADMIN_DEFAULT_PASSWORD`
  in `.env.postgres.setup` (sample env provides defaults).
- Data persists under `./data/*`.
- Proxylogin is disabled by default in compose; enable it if you use delegated auth.


```bash
alias dc-infra='docker compose -f docker-compose.yaml'
```

```shell
docker compose stop postgres-setup && docker compose rm postgres-setup -f && docker compose build postgres-setup --no-cache && docker compose up postgres-setup -d
```

```shell
docker compose stop proxylogin && docker compose rm proxylogin -f && docker compose build proxylogin --no-cache && docker compose up proxylogin -d
```

```shell
docker compose stop redis && docker compose rm redis -f && docker compose up redis -d --build
```

```shell
docker compose stop proxylogin && docker compose rm proxylogin -f && docker compose build --no-cache && docker compose up proxylogin -d
```

```bash
# Rebuild ingress (no deps)
 dc-infra build chat-ingress && dc-infra up -d --no-deps chat-ingress

# Rebuild processor (no deps)
 dc-infra build chat-proc && dc-infra up -d --no-deps chat-proc

# Rebuild UI (no deps)
 dc-infra build web-ui && dc-infra up -d --no-deps web-ui

# Rebuild proxylogin (no deps)
 dc-infra build proxylogin --no-cache && dc-infra up -d --no-deps proxylogin

# Rebuild proxy (no deps)
 dc-infra build web-proxy && dc-infra up -d --no-deps web-proxy

# Logs
 dc-infra logs -f chat-proc
```