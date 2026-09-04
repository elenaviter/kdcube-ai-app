#!/usr/bin/env python3
"""Create local standard descriptors and matching Compose credentials."""

from __future__ import annotations

import argparse
import getpass
import json
import secrets
import shutil
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROVIDER_PATHS = {
    "openai": ("openai", "api_key"),
    "anthropic": ("anthropic", "api_key"),
    "google": ("google", "api_key"),
    "openrouter": ("openrouter", "api_key"),
}


def _secret_descriptor(
    *, provider: str, provider_key: str | None, postgres: str, redis: str
) -> dict:
    document = {
        "services": {
            "openai": {"api_key": None},
            "anthropic": {"api_key": None, "claude_code_key": None},
            "google": {"api_key": None},
            "openrouter": {"api_key": None},
            "git": {"http_token": None, "http_user": "x-access-token"},
        },
        "infra": {
            "postgres": {"password": postgres},
            "redis": {"password": redis},
        },
    }
    if provider_key:
        service, field = PROVIDER_PATHS[provider]
        document["services"][service][field] = provider_key
    return document


def configure(
    root: Path,
    *,
    provider: str,
    provider_key: str | None,
) -> tuple[Path, Path, Path]:
    descriptors = root / "descriptors.local"
    compose_env = root / ".env"
    claude_repo = root / "output" / "claude-session-store.git"
    existing = [path for path in (descriptors, compose_env, claude_repo) if path.exists()]
    if existing:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"local configuration already exists: {rendered}")
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is required to initialize the Claude transcript store")

    postgres_password = secrets.token_urlsafe(24)
    redis_password = secrets.token_urlsafe(24)
    shutil.copytree(root / "descriptors.template", descriptors)
    secrets_path = descriptors / "secrets.yaml"
    secrets_path.write_text(
        json.dumps(
            _secret_descriptor(
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
                "AGENT_DEMO_POSTGRES_USER=kdcube_agents",
                "AGENT_DEMO_POSTGRES_DATABASE=kdcube_agents",
                "AGENT_DEMO_POSTGRES_PORT=55432",
                "AGENT_DEMO_REDIS_PORT=56379",
                "",
            )
        ),
        encoding="utf-8",
    )
    compose_env.chmod(0o600)
    claude_repo.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [git, "init", "--bare", str(claude_repo)],
        check=True,
        capture_output=True,
        text=True,
    )
    return descriptors, compose_env, claude_repo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=("none", *PROVIDER_PATHS),
        default="openai",
        help=(
            "Model credential to place in secrets.yaml; use none for an "
            "existing Claude CLI login."
        ),
    )
    args = parser.parse_args()
    provider_key = None
    if args.provider != "none":
        provider_key = getpass.getpass(f"{args.provider} API key: ").strip()
        if not provider_key:
            raise SystemExit("provider key cannot be empty")
    descriptors, compose_env, claude_repo = configure(
        HERE,
        provider=args.provider,
        provider_key=provider_key,
    )
    print(f"created {descriptors}")
    print(f"created {compose_env}")
    print(f"created {claude_repo}")
    print("local secrets were not printed")


if __name__ == "__main__":
    main()
