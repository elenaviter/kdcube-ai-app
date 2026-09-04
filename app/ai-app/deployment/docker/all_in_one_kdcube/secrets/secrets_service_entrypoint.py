# SPDX-License-Identifier: MIT
from __future__ import annotations

import os
import sys
from typing import Optional


SERVER_PATHS = {
    "ephemeral": "/app/secrets_server.py",
    "host-vault": "/app/host_vault/broker_server.py",
}


def selected_server_path(raw_backend: Optional[str] = None) -> str:
    backend = (
        str(
            raw_backend
            if raw_backend is not None
            else os.getenv("KDCUBE_SECRETS_SERVICE_BACKEND", "ephemeral")
        )
        .strip()
        .lower()
        .replace("_", "-")
    )
    if backend in {"memory", "transient", "ephemeral-memory"}:
        backend = "ephemeral"
    try:
        return SERVER_PATHS[backend]
    except KeyError as exc:
        supported = ", ".join(sorted(SERVER_PATHS))
        raise ValueError(
            f"unsupported secrets service backend; expected one of: {supported}"
        ) from exc


def main() -> int:
    try:
        server_path = selected_server_path()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    os.execv(sys.executable, [sys.executable, server_path])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
