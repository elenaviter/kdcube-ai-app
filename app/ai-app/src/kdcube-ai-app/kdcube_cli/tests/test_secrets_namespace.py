from __future__ import annotations

import os
from pathlib import Path

import yaml

from kdcube_cli.secrets_namespace import migrate_platform_secret_namespace


def _write(path: Path, value: object) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    if os.name == "posix":
        path.chmod(0o600)


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_namespace_migration_moves_platform_and_misplaced_bundle_values(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config"
    config.mkdir()
    secret = "namespace-migration-canary"
    _write(
        config / "secrets.yaml",
        {
            "services": {"fixture": {"token": secret}},
            "users": {"owner": {"secrets": {"token": "user-canary"}}},
            "bundles": {
                "demo@1-0": {"secrets": {"provider": {"token": "bundle-canary"}}}
            },
        },
    )
    _write(
        config / "bundles.secrets.yaml",
        {"bundles": {"version": "1", "items": []}},
    )

    dry_run = migrate_platform_secret_namespace(config, dry_run=True)
    assert dry_run.changed is True
    assert dry_run.conflicts == ()
    assert dry_run.moved_platform_keys == 1
    assert dry_run.moved_bundle_keys == 1
    assert "namespace-migration-canary" not in str(dry_run.to_dict())
    assert "services" in _load(config / "secrets.yaml")

    applied = migrate_platform_secret_namespace(config, dry_run=False)
    assert applied.changed is True
    platform = _load(config / "secrets.yaml")
    assert platform == {
        "users": {"owner": {"secrets": {"token": "user-canary"}}},
        "platform": {"services": {"fixture": {"token": secret}}},
    }
    bundles = _load(config / "bundles.secrets.yaml")
    assert bundles["bundles"]["items"] == [
        {
            "id": "demo@1-0",
            "secrets": {"provider": {"token": "bundle-canary"}},
        }
    ]

    repeated = migrate_platform_secret_namespace(config, dry_run=False)
    assert repeated.changed is False
    assert repeated.conflicts == ()


def test_namespace_migration_reports_conflict_without_writing(tmp_path: Path) -> None:
    config = tmp_path / "config"
    config.mkdir()
    original = {
        "platform": {"services": {"fixture": {"token": "canonical"}}},
        "services": {"fixture": {"token": "legacy"}},
    }
    _write(config / "secrets.yaml", original)

    result = migrate_platform_secret_namespace(config, dry_run=False)

    assert result.to_dict()["ok"] is False
    assert result.changed is False
    assert result.conflicts == ("platform.services.fixture.token",)
    assert _load(config / "secrets.yaml") == original
