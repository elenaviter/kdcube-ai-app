from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from kdcube_cli.management.errors import ManagementCliError
from kdcube_cli.management.filesystem import apply_open_file_mode
from kdcube_cli.management.models import ManagementSecretTarget
from kdcube_cli.management.secret_export import ExportedSecret


@dataclass(frozen=True)
class SecretDescriptorExport:
    directory: Path
    platform_path: Path
    bundles_path: Path
    platform_count: int
    bundle_count: int


def _insert(root: dict[str, Any], dotted_key: str, value: str) -> None:
    parts = dotted_key.split(".")
    current = root
    for part in parts[:-1]:
        existing = current.get(part)
        if existing is None:
            nested: dict[str, Any] = {}
            current[part] = nested
            current = nested
            continue
        if not isinstance(existing, dict):
            raise ManagementCliError(
                "secret_export_descriptor_conflict",
                "The selected secret keys cannot be represented in one descriptor.",
            )
        current = existing
    leaf = parts[-1]
    if leaf in current:
        raise ManagementCliError(
            "secret_export_descriptor_conflict",
            "The secret export contains a duplicate descriptor key.",
        )
    current[leaf] = value


def _render(values: Sequence[ExportedSecret]) -> tuple[bytes, bytes, int, int]:
    platform: dict[str, Any] = {}
    bundles: dict[str, dict[str, Any]] = {}
    identities: set[tuple[str, str, str]] = set()
    platform_count = 0
    bundle_count = 0
    for exported in sorted(values, key=lambda item: item.target.identity):
        if exported.target.identity in identities:
            raise ManagementCliError(
                "secret_export_descriptor_conflict",
                "The secret export contains a duplicate descriptor key.",
            )
        identities.add(exported.target.identity)
        if exported.target.scope == "platform":
            _insert(platform, exported.target.key, exported.value)
            platform_count += 1
            continue
        bundle = bundles.setdefault(exported.target.bundle_id, {})
        _insert(bundle, exported.target.key, exported.value)
        bundle_count += 1

    platform_text = yaml.safe_dump(
        platform,
        allow_unicode=False,
        default_flow_style=False,
        sort_keys=False,
    )
    bundle_document = {
        "bundles": {
            "version": "1",
            "items": [
                {"id": bundle_id, "secrets": secrets}
                for bundle_id, secrets in sorted(bundles.items())
            ],
        }
    }
    bundles_text = yaml.safe_dump(
        bundle_document,
        allow_unicode=False,
        default_flow_style=False,
        sort_keys=False,
    )
    return (
        platform_text.encode("utf-8"),
        bundles_text.encode("utf-8"),
        platform_count,
        bundle_count,
    )


def _write_private(path: Path, content: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        apply_open_file_mode(descriptor, path, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _sync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_secret_descriptor_export(
    output_directory: Path,
    targets: Sequence[ManagementSecretTarget],
) -> Path:
    target = output_directory.expanduser().absolute()
    if not target.parent.is_dir():
        raise ManagementCliError(
            "secret_export_output_parent_missing",
            "The secret export output parent directory does not exist.",
        )
    if target.exists() or target.is_symlink():
        raise ManagementCliError(
            "secret_export_output_exists",
            "The secret export output directory already exists; choose a new directory.",
        )
    _render([ExportedSecret(target=item, value="") for item in targets])
    return target


def write_secret_descriptors(
    output_directory: Path,
    values: Sequence[ExportedSecret],
) -> SecretDescriptorExport:
    """Create one new directory containing the descriptor pair without clobbering."""

    target = validate_secret_descriptor_export(
        output_directory,
        [item.target for item in values],
    )
    parent = target.parent
    platform, bundles, platform_count, bundle_count = _render(values)
    temporary: Path | None = None
    target_created = False
    completed = False
    try:
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".tmp", dir=parent)
        )
        if os.name != "nt":
            os.chmod(temporary, 0o700)
        _write_private(temporary / "secrets.yaml", platform)
        _write_private(temporary / "bundles.secrets.yaml", bundles)
        _sync_directory(temporary)
        try:
            target.mkdir(mode=0o700)
            target_created = True
        except FileExistsError:
            raise ManagementCliError(
                "secret_export_output_exists",
                "The secret export output directory already exists; choose a new directory.",
            ) from None
        if os.name != "nt":
            os.chmod(target, 0o700)
        os.replace(temporary / "secrets.yaml", target / "secrets.yaml")
        os.replace(
            temporary / "bundles.secrets.yaml",
            target / "bundles.secrets.yaml",
        )
        _sync_directory(target)
        _sync_directory(parent)
        completed = True
    except ManagementCliError:
        raise
    except Exception as exc:
        raise ManagementCliError(
            "secret_export_output_write_failed",
            "The secret descriptor export could not be written.",
        ) from exc
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
        if target_created and not completed:
            shutil.rmtree(target, ignore_errors=True)
    return SecretDescriptorExport(
        directory=target,
        platform_path=target / "secrets.yaml",
        bundles_path=target / "bundles.secrets.yaml",
        platform_count=platform_count,
        bundle_count=bundle_count,
    )


__all__ = [
    "SecretDescriptorExport",
    "validate_secret_descriptor_export",
    "write_secret_descriptors",
]
