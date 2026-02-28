# ECS Deployment Templates

This folder contains **environment templates** for ECS tasks/services.
They are intended as starting points for task definitions or parameter stores.

## Templates
- `deployment/ecs/ingress/env.template` — Ingress (SSE/API entrypoint).
- `deployment/ecs/proc/env.template` — Processor (queue worker + integrations API).
- `deployment/ecs/metrics/env.template` — Metrics service (CloudWatch/Prometheus export).
- `deployment/ecs/frontend/env.template` — UI service (runtime config injection).

## Required shared settings
All services must set:
- `REDIS_URL` (ingress/proc/metrics)
- `GATEWAY_CONFIG_JSON` (must include `tenant` + `project`)
- `GATEWAY_COMPONENT` (`ingress` | `proc` | `metrics`)

## AWS runtime (recommended on ECS)
If running on ECS/EC2 with IAM roles:
- `AWS_REGION` or `AWS_DEFAULT_REGION`
- `AWS_EC2_METADATA_DISABLED=false`
- `NO_PROXY=169.254.169.254,localhost,127.0.0.1` (only if proxy is used)

## UI Runtime Config (ECS)
The UI container supports runtime config injection. Use **one** of:

- `FRONTEND_CONFIG_JSON` — JSON string for `config.json`
- `FRONTEND_CONFIG_S3_URL` — S3 or HTTPS URL to fetch on startup

Use one of them in the frontend task definition.

**Example (inline JSON):**
```
FRONTEND_CONFIG_JSON='{"auth":{"authType":"delegated","apiBase":"/auth/"},"tenant":"<TENANT>","project":"<PROJECT>","routesPrefix":"/chatbot/api"}'
```

**Example (S3):**
```
FRONTEND_CONFIG_S3_URL=s3://<bucket>/<path>/config.json
AWS_REGION=eu-west-1
AWS_EC2_METADATA_DISABLED=false
NO_PROXY=169.254.169.254,localhost,127.0.0.1
```

## Component docs
- Metrics scheduled task example: `docs/ops/ecs/components/metric-server-README.md`

## Bundles from Git (proc)

For git‑defined bundles, ensure:

- `git` binary is available in the proc image (included by default).
- `BUNDLE_GIT_RESOLUTION_ENABLED=1`
- `BUNDLE_GIT_REDIS_LOCK=1` (each replica pulls once)
- `AGENTIC_BUNDLES_JSON` can point to a JSON/YAML file path mounted into the task (recommended for readability).
  Mount it to `/config/release.yaml` and set:
  ```
  AGENTIC_BUNDLES_JSON=/config/release.yaml
  ```

**Rule:** set `subdir` to the **parent bundles directory** and use `module: "<bundle_folder>.entrypoint"`.

**Option A — EFS (recommended for git pulls)**

- Mount EFS to `/bundles`
- Use an EFS Access Point with:
  - `posix_user.uid = 1000`
  - `posix_user.gid = 1000`
- Set:
  ```
  AGENTIC_BUNDLES_ROOT=/bundles
  BUNDLE_GIT_RESOLUTION_ENABLED=1
  ```

**Private repos (SSH):**
Provide these envs and mount the key/known_hosts into the container:

```
GIT_SSH_KEY_PATH=/run/secrets/git_ssh_key
GIT_SSH_KNOWN_HOSTS=/run/secrets/git_known_hosts
GIT_SSH_STRICT_HOST_KEY_CHECKING=yes
```

**Bundles root:**

Set `AGENTIC_BUNDLES_ROOT=/bundles` in the proc task definition.  
Avoid setting `HOST_BUNDLES_PATH` in ECS unless the path is valid inside the container.

## Notes
- These templates intentionally use **placeholders**. Replace them in your task
  definition or inject via parameter store/secrets manager.
