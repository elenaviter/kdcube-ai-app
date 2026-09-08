#!/usr/bin/env python3
"""Create local standard descriptors and matching Compose credentials."""

from __future__ import annotations

import argparse
import copy
import getpass
import json
import secrets
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlparse

import yaml


PROVIDER_PATHS = {
    "openai": ("openai", "api_key"),
    "anthropic": ("anthropic", "api_key"),
    "google": ("google", "api_key"),
    "openrouter": ("openrouter", "api_key"),
}


def _secret_descriptor(
    template: Mapping,
    *,
    provider: str,
    provider_key: str | None,
    postgres: str,
    redis: str,
) -> dict:
    document = copy.deepcopy(dict(template))
    platform = document.setdefault("platform", {})
    services = platform.setdefault("services", {})
    infra = platform.setdefault("infra", {})
    infra.setdefault("postgres", {})["password"] = postgres
    infra.setdefault("redis", {})["password"] = redis
    if provider_key:
        service, field = PROVIDER_PATHS[provider]
        services.setdefault(service, {})[field] = provider_key
    return document


def configure(
    root: Path,
    *,
    provider: str,
    provider_key: str | None,
) -> tuple[Path, Path, Path | None]:
    descriptors = root / "descriptors.local"
    compose_env = root / ".env"
    assembly_template = (
        yaml.safe_load(
            (root / "descriptors.template" / "assembly.yaml").read_text(
                encoding="utf-8"
            )
        )
        or {}
    )
    session_config = (assembly_template.get("storage") or {}).get(
        "claude_code_session"
    ) or {}
    session_repo = str(session_config.get("repo") or "").strip()
    parsed_repo = urlparse(session_repo)
    claude_repo = None
    if (
        str(session_config.get("type") or "").strip().lower() == "git"
        and session_repo
        and not parsed_repo.scheme
        and not session_repo.startswith("git@")
    ):
        claude_repo = (descriptors / session_repo).resolve()
    candidates = (descriptors, compose_env, claude_repo)
    existing = [path for path in candidates if path is not None and path.exists()]
    if existing:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"local configuration already exists: {rendered}")
    git = shutil.which("git") if claude_repo is not None else None
    if claude_repo is not None and git is None:
        raise RuntimeError("git is required to initialize the Claude transcript store")

    postgres_password = secrets.token_urlsafe(24)
    redis_password = secrets.token_urlsafe(24)
    shutil.copytree(root / "descriptors.template", descriptors)
    assembly = (
        yaml.safe_load((descriptors / "assembly.yaml").read_text(encoding="utf-8"))
        or {}
    )
    infra = assembly.get("infra") or {}
    postgres_config = infra.get("postgres") or {}
    redis_config = infra.get("redis") or {}
    secrets_path = descriptors / "secrets.yaml"
    secrets_template = yaml.safe_load(secrets_path.read_text(encoding="utf-8")) or {}
    secrets_path.write_text(
        json.dumps(
            _secret_descriptor(
                secrets_template,
                provider=provider,
                provider_key=provider_key,
                postgres=postgres_password,
                redis=redis_password,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    secrets_path.chmod(0o600)
    compose_env.write_text(
        "\n".join(
            (
                f"AGENT_DEMO_POSTGRES_PASSWORD={postgres_password}",
                f"AGENT_DEMO_REDIS_PASSWORD={redis_password}",
                "AGENT_DEMO_POSTGRES_USER="
                f"{postgres_config.get('user') or 'kdcube_agents'}",
                "AGENT_DEMO_POSTGRES_DATABASE="
                f"{postgres_config.get('database') or 'kdcube_agents'}",
                "AGENT_DEMO_POSTGRES_PORT="
                f"{int(postgres_config.get('port') or 55432)}",
                f"AGENT_DEMO_REDIS_PORT={int(redis_config.get('port') or 56379)}",
                "",
            )
        ),
        encoding="utf-8",
    )
    compose_env.chmod(0o600)
    if claude_repo is not None:
        claude_repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [str(git), "init", "--bare", str(claude_repo)],
            check=True,
            capture_output=True,
            text=True,
        )
    return descriptors, compose_env, claude_repo


def main(*, root: Path | None = None, default_provider: str = "openai") -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=("none", *PROVIDER_PATHS),
        default=default_provider,
        help=(
            "Model credential to place in secrets.yaml; use none for an "
            "unprotected on-host model gateway or an existing Claude CLI login."
        ),
    )
    args = parser.parse_args()
    provider_key = None
    if args.provider != "none":
        provider_key = getpass.getpass(f"{args.provider} API key: ").strip()
        if not provider_key:
            raise SystemExit("provider key cannot be empty")
    descriptors, compose_env, claude_repo = configure(
        (root or Path.cwd()).resolve(),
        provider=args.provider,
        provider_key=provider_key,
    )
    print(f"created {descriptors}")
    print(f"created {compose_env}")
    if claude_repo is not None:
        print(f"created {claude_repo}")
    print("local secrets were not printed")


if __name__ == "__main__":
    main()
