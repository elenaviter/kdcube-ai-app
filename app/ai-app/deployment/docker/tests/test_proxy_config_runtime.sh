#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
IMAGE="${1:-kdcube-proxy-contract-test}"
TEMPLATE="$ROOT_DIR/deployment/docker/tests/fixtures/nginx.conf.template"
STATE_DIR="$(mktemp -d)"
trap 'rm -rf "$STATE_DIR"' EXIT

if command -v sha256sum >/dev/null 2>&1; then
  EXPECTED_SHA="$(sha256sum "$TEMPLATE" | awk '{print $1}')"
else
  EXPECTED_SHA="$(shasum -a 256 "$TEMPLATE" | awk '{print $1}')"
fi
CONFIG_B64="$(base64 < "$TEMPLATE" | tr -d '\n')"
COMMON_ENV=(
  -e APP_DOMAIN=example.test
  -e ALB_CIDR=10.0.0.0/16
  -e ROUTE_PREFIX=/platform
  -e NGINX_CONFIG_SHA256="$EXPECTED_SHA"
)

docker run --rm \
  "${COMMON_ENV[@]}" \
  -e NGINX_CONFIG_B64="$CONFIG_B64" \
  -v "$STATE_DIR:/nginx-config" \
  "$IMAGE" \
  /usr/local/bin/kdcube-nginx-config activate

docker run --rm \
  "${COMMON_ENV[@]}" \
  -v "$STATE_DIR:/nginx-config" \
  "$IMAGE" \
  /usr/local/bin/kdcube-nginx-config render

docker run --rm \
  "${COMMON_ENV[@]}" \
  -v "$STATE_DIR:/nginx-config" \
  "$IMAGE" \
  /bin/sh -c '
    /usr/local/bin/kdcube-nginx-config render &&
    openresty &&
    test "$(wget -qO- http://127.0.0.1/forwarded-proto)" = "http" &&
    test "$(wget -qO- --header="X-Forwarded-Proto: https" http://127.0.0.1/forwarded-proto)" = "https" &&
    test "$(wget -qO- --header="X-Forwarded-Proto: http, https" http://127.0.0.1/forwarded-proto)" = "https" &&
    test "$(wget -qO- --header="X-Forwarded-Proto: ftp" http://127.0.0.1/forwarded-proto)" = "http"
  '

set +e
docker run --rm \
  "${COMMON_ENV[@]}" \
  -e NGINX_CONFIG_B64="$CONFIG_B64" \
  -v "$STATE_DIR:/nginx-config" \
  "$IMAGE" \
  /usr/local/bin/kdcube-nginx-config activate
UNCHANGED_EXIT=$?

docker run --rm \
  -e APP_DOMAIN=example.test \
  -e ALB_CIDR=10.0.0.0/16 \
  -e ROUTE_PREFIX=/platform \
  -e NGINX_CONFIG_SHA256=ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff \
  -v "$STATE_DIR:/nginx-config" \
  "$IMAGE" \
  /usr/local/bin/kdcube-nginx-config render
MISMATCH_EXIT=$?

rm -f "$STATE_DIR/nginx.conf.sha256"
docker run --rm \
  "${COMMON_ENV[@]}" \
  -v "$STATE_DIR:/nginx-config" \
  "$IMAGE" \
  /usr/local/bin/kdcube-nginx-config render
MISSING_MARKER_EXIT=$?
set -e

if [[ "$UNCHANGED_EXIT" -ne 10 ]]; then
  echo "expected unchanged activation exit 10, got $UNCHANGED_EXIT" >&2
  exit 1
fi

if [[ "$MISMATCH_EXIT" -ne 20 ]]; then
  echo "expected task-definition mismatch exit 20, got $MISMATCH_EXIT" >&2
  exit 1
fi

if [[ "$MISSING_MARKER_EXIT" -ne 21 ]]; then
  echo "expected missing-marker exit 21, got $MISSING_MARKER_EXIT" >&2
  exit 1
fi

LOCAL_SERVICE_HOSTS=(
  --add-host chat-ingress:127.0.0.1
  --add-host chat-proc:127.0.0.1
  --add-host metrics:127.0.0.1
  --add-host proxylogin:127.0.0.1
  --add-host web-ui:127.0.0.1
)
LOCAL_CONFIGS=(
  deployment/docker/all_in_one_kdcube/nginx/conf/nginx_proxy.conf
  deployment/docker/all_in_one_kdcube/nginx/conf/nginx_proxy_delegated.conf
  deployment/docker/custom-ui-managed-infra/nginx/conf/nginx_proxy.conf
  deployment/docker/custom-ui-managed-infra/nginx/conf/nginx_proxy_delegated.conf
)

for relative_config in "${LOCAL_CONFIGS[@]}"; do
  for source in request trusted_x_forwarded_proto; do
    candidate="$STATE_DIR/$(echo "$relative_config-$source" | tr '/' '_')"
    if [[ "$source" == "request" ]]; then
      cp "$ROOT_DIR/$relative_config" "$candidate"
    else
      sed \
        -e 's/KDCUBE_FORWARDED_PROTO_SOURCE: request/KDCUBE_FORWARDED_PROTO_SOURCE: trusted_x_forwarded_proto/' \
        -e 's/map \$scheme \$forwarded_proto_last/map \$http_x_forwarded_proto \$forwarded_proto_last/' \
        -e 's/default                        \$scheme;/default                        \$http_x_forwarded_proto;/' \
        "$ROOT_DIR/$relative_config" > "$candidate"
    fi

    docker run --rm \
      "${LOCAL_SERVICE_HOSTS[@]}" \
      -v "$candidate:/usr/local/openresty/nginx/conf/nginx.conf:ro" \
      "$IMAGE" \
      /usr/local/bin/kdcube-nginx-config render
  done
done

echo "proxy config runtime contract verified"
