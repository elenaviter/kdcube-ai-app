from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from urllib.parse import quote

SECRETS_URL = os.getenv("SECRETS_URL", "http://127.0.0.1:7777")
ADMIN_TOKEN = os.getenv("SECRETS_ADMIN_TOKEN")


def _post_json(path: str, payload: dict[str, object]) -> dict[str, object]:
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if ADMIN_TOKEN:
        headers["X-KDCUBE-ADMIN-TOKEN"] = ADMIN_TOKEN
    req = urllib.request.Request(
        f"{SECRETS_URL}{path}",
        data=data,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        if resp.status >= 400:
            raise RuntimeError(f"Request failed: {resp.status}")
        parsed = json.loads(resp.read().decode("utf-8") or "{}")
    if not isinstance(parsed, dict):
        raise RuntimeError("Invalid response")
    return parsed


def _post_set(
    key: str,
    value: str,
    *,
    expected_generation: int | None = None,
) -> None:
    payload: dict[str, object] = {"key": key, "value": value}
    if expected_generation is not None:
        payload["expected_generation"] = expected_generation
    _post_json("/set", payload)


def _verify(key: str, sha256: str) -> str:
    payload = _post_json("/verify", {"key": key, "sha256": sha256})
    state = str(payload.get("state") or "")
    if state not in {"match", "missing", "different"}:
        raise RuntimeError("Invalid verification response")
    return state


def _delete(key: str) -> None:
    headers = {}
    if ADMIN_TOKEN:
        headers["X-KDCUBE-ADMIN-TOKEN"] = ADMIN_TOKEN
    req = urllib.request.Request(
        f"{SECRETS_URL}/secret/{quote(key, safe='')}",
        headers=headers,
        method="DELETE",
    )
    with urllib.request.urlopen(req) as resp:
        if resp.status >= 400:
            raise RuntimeError(f"Request failed: {resp.status}")


def main() -> int:
    try:
        if len(sys.argv) in {4, 5} and sys.argv[1] == "set" and sys.argv[3] == "--stdin":
            create_only = len(sys.argv) == 5 and sys.argv[4] == "--if-absent"
            if len(sys.argv) == 5 and not create_only:
                return _usage()
            key, value = sys.argv[2], sys.stdin.read()
            _post_set(key, value, expected_generation=0 if create_only else None)
            print("ok")
            return 0
        if len(sys.argv) == 4 and sys.argv[1] == "set":
            key, value = sys.argv[2], sys.argv[3]
            _post_set(key, value)
            print("ok")
            return 0
        if len(sys.argv) == 4 and sys.argv[1] == "verify" and sys.argv[3] == "--sha256-stdin":
            state = _verify(sys.argv[2], sys.stdin.read().strip())
            print(state)
            return {"match": 0, "missing": 3, "different": 4}[state]
        if len(sys.argv) == 3 and sys.argv[1] == "delete":
            _delete(sys.argv[2])
            print("ok")
            return 0
    except urllib.error.HTTPError as exc:
        print("secrets operation failed", file=sys.stderr)
        return 5 if exc.code in {502, 503, 504} else 2
    except urllib.error.URLError:
        print("secrets operation failed", file=sys.stderr)
        return 5
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
        print("secrets operation failed", file=sys.stderr)
        return 2
    return _usage()


def _usage() -> int:
    print(
        "Usage: secretsctl.py set KEY --stdin [--if-absent] | "
        "set KEY VALUE | verify KEY --sha256-stdin | delete KEY"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
