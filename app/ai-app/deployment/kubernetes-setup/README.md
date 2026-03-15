# KDCube Kubernetes Setup (Compose Parity)

This folder contains a Kubernetes/Helm deployment equivalent of the active services from `app/ai-app/deployment/docker/all_in_one_kdcube/docker-compose.yaml`.

Validated in this workspace on:
- k3d cluster `k3d-k3s-dev`
- namespace `kdcube-ai-app`
- published image tag `2026.3.15.435`

Included services:
- `postgres-db` (pgvector)
- `postgres-setup` (one-shot bootstrap job)
- `redis`
- `kdcube-secrets`
- `chat-ingress`
- `chat-proc`
- `metrics`
- `web-ui`
- `web-proxy`

Ignored on purpose (per request):
- `clamav`
- `pgadmin`
- commented-out compose services (`kb`, `dramatiq`, `neo4j`, `proxylogin`)

## Layout

- `charts/postgres-db`: pgvector PostgreSQL chart
- `charts/redis`: Redis chart with password auth
- `charts/postgres-setup`: bootstrap job chart
- `charts/kdcube-platform`: app services chart
- `manifests/*.yaml`: external config files mounted by app services
- `values/*.yaml`: install-time values files

Important values files:
- `values/kdcube-platform-values.yaml`: shared platform baseline
- `values/kdcube-platform-k3d-values.yaml`: k3d/containerd override that disables the processor Docker socket mount

## Prerequisites

- Kubernetes cluster (k3d/k3s/minikube/EKS/etc)
- `kubectl`
- `helm`
- Access to Docker Hub images under `kdcube/*`

## Installation Order

Use this exact order.

1. Create namespace:

```bash
kubectl create namespace kdcube-ai-app
```

2. Apply external config manifests (outside charts):

```bash
kubectl apply -n kdcube-ai-app -f manifests/backend-configmap.yaml
kubectl apply -n kdcube-ai-app -f manifests/frontend-runtime-configmap.yaml
kubectl apply -n kdcube-ai-app -f manifests/frontend-nginx-configmap.yaml
kubectl apply -n kdcube-ai-app -f manifests/nginx-proxy-configmap.yaml
kubectl apply -n kdcube-ai-app -f manifests/descriptors-configmap.yaml
```

3. Install PostgreSQL chart:

```bash
helm upgrade --install postgres-db charts/postgres-db \
  -n kdcube-ai-app \
  -f values/postgres-db-values.yaml
```

4. Install Redis chart:

```bash
helm upgrade --install redis charts/redis \
  -n kdcube-ai-app \
  -f values/redis-values.yaml
```

5. Run Postgres bootstrap job:

```bash
helm upgrade --install postgres-setup charts/postgres-setup \
  -n kdcube-ai-app \
  -f values/postgres-setup-values.yaml
```

6. Install platform services:

```bash
helm upgrade --install kdcube-platform charts/kdcube-platform \
  -n kdcube-ai-app \
  -f values/kdcube-platform-values.yaml \
  -f values/kdcube-platform-k3d-values.yaml
```

## External Files To Maintain

These are intentionally outside Helm charts because they are environment-specific and change often.

- `manifests/backend-configmap.yaml`
: Runtime backend env (`GATEWAY_CONFIG_JSON`, auth mode, storage paths, ports, model defaults).

- `manifests/frontend-runtime-configmap.yaml`
: UI runtime `config.json` mounted into `web-ui` as `/usr/share/nginx/html/config.json`.

- `manifests/frontend-nginx-configmap.yaml`
: UI nginx config mounted into `web-ui` as `/etc/nginx/nginx.conf`.

- `manifests/nginx-proxy-configmap.yaml`
: Proxy nginx config mounted into `web-proxy` as `/usr/local/openresty/nginx/conf/nginx.conf.template`.

- `manifests/descriptors-configmap.yaml`
: `assembly.yaml`, `bundles.yaml`, `gateway.yaml`, `secrets.yaml` mounted into `chat-proc` at `/config/*`.

## Data and Mounts

`kdcube-platform` chart creates PVCs by default:
- `bundles`
- `kdcube-storage`
- `bundle-storage`
- `exec-workspace`
- `kdcube-logs`

Adjust sizes and `storageClassName` in `values/kdcube-platform-values.yaml`.

For k3d specifically, keep the install order above and include `values/kdcube-platform-k3d-values.yaml` when installing or upgrading the platform chart.

If you need host paths for local development:
- disable/replace PVC usage in `charts/kdcube-platform/templates/*.yaml`
- or pre-bind PVs to local paths in your cluster.

## Secrets and Credentials

- DB credentials are created by `postgres-db` chart secret: `postgres-db-credentials`
- Redis password secret is created by `redis` chart secret: `redis-auth`
- App/API keys are in `kdcube-platform` chart secret (`kdcube-platform-secrets`)

Before install, set:
- `OPENAI_API_KEY`
- `HUGGING_FACE_API_TOKEN`
- `ANTHROPIC_API_KEY`
- `BRAVE_API_KEY`
- `SECRETS_*` tokens

in `values/kdcube-platform-values.yaml`.

## k3d Note

`chat-proc` is running on k3d with the Docker socket mount disabled.

That means the processor service starts correctly, but any runtime feature that requires launching Docker-based execution containers from inside the pod will need a different execution strategy for containerd-based clusters.

## Validation

```bash
kubectl get pods -n kdcube-ai-app
kubectl get svc -n kdcube-ai-app
kubectl logs -n kdcube-ai-app deploy/chat-ingress
kubectl logs -n kdcube-ai-app deploy/chat-proc
```

If using local port-forward for web access:

```bash
kubectl -n kdcube-ai-app port-forward svc/web-proxy 8080:80
```

Then open `http://localhost:8080/chatbot/chat`.
