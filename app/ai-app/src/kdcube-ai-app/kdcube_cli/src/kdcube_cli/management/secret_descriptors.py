from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from kdcube_cli.management.errors import ManagementCliError
from kdcube_cli.management.filesystem import apply_open_file_mode
from kdcube_cli.management.models import ManagementSecretTarget
from kdcube_cli.management.secret_export import ExportedSecret

MAX_SECRET_DESCRIPTOR_VALUES = 4096


@dataclass(frozen=True)
class SecretDescriptorExport:
    directory: Path
    platform_path: Path
    bundles_path: Path
    platform_count: int
    bundle_count: int
    user_count: int

    @property
    def total_count(self) -> int:
        return self.platform_count + self.bundle_count + self.user_count


@dataclass(frozen=True)
class SecretDescriptorImport:
    directory: Path
    values: tuple[ExportedSecret, ...]
    platform_count: int
    bundle_count: int
    user_count: int

    @property
    def total_count(self) -> int:
        return self.platform_count + self.bundle_count + self.user_count


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


def _load_private_yaml(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ManagementCliError(
            "secret_import_descriptor_invalid",
            "The secret import requires regular descriptor files.",
        )
    if os.name == "posix" and path.stat().st_mode & 0o077:
        raise ManagementCliError(
            "secret_import_descriptor_permissions",
            "Secret import descriptors must be owner-only (mode 0600).",
        )
    try:
        if path.stat().st_size > 16 * 1024 * 1024:
            raise ManagementCliError(
                "secret_import_descriptor_too_large",
                "A secret import descriptor is too large.",
            )
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except ManagementCliError:
        raise
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ManagementCliError(
            "secret_import_descriptor_invalid",
            "A secret import descriptor could not be read.",
        ) from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ManagementCliError(
            "secret_import_descriptor_invalid",
            "A secret import descriptor must contain a mapping.",
        )
    return loaded


def _flatten_values(
    prefix: str,
    node: Any,
    output: dict[str, str],
) -> None:
    if node is None:
        return
    if isinstance(node, Mapping):
        for raw_key, value in node.items():
            key = str(raw_key or "").strip()
            if not key:
                raise ManagementCliError(
                    "secret_import_descriptor_invalid",
                    "A secret import descriptor contains an empty key.",
                )
            child = f"{prefix}.{key}" if prefix else key
            _flatten_values(child, value, output)
        return
    if isinstance(node, list):
        for index, value in enumerate(node):
            child = f"{prefix}.{index}" if prefix else str(index)
            _flatten_values(child, value, output)
        return
    value = node if isinstance(node, str) else str(node)
    if value == "":
        return
    stripped = value.strip()
    if (
        (stripped.startswith("<") and stripped.endswith(">"))
        or stripped.lower() in {"changeme", "change-me", "replace-me"}
    ):
        raise ManagementCliError(
            "secret_import_placeholder_value",
            "A secret import descriptor contains a placeholder value.",
        )
    if prefix in output:
        raise ManagementCliError(
            "secret_import_descriptor_conflict",
            "The secret import contains a duplicate provider key.",
        )
    output[prefix] = value


def load_secret_descriptors(
    input_directory: Path,
    *,
    include_platform: bool = True,
    allow_empty: bool = False,
) -> SecretDescriptorImport:
    """Read the ordinary descriptor pair into exact canonical provider values."""

    directory = input_directory.expanduser().absolute()
    if directory.is_symlink() or not directory.is_dir():
        raise ManagementCliError(
            "secret_import_directory_invalid",
            "The secret import path must be a descriptor directory.",
        )
    platform_data = (
        _load_private_yaml(directory / "secrets.yaml")
        if include_platform
        else {}
    )
    bundles_data = _load_private_yaml(directory / "bundles.secrets.yaml")
    platform_root = (
        platform_data.get("secrets")
        if isinstance(platform_data.get("secrets"), dict)
        else platform_data
    )
    invalid_roots = sorted(
        str(key) for key in platform_root if str(key) not in {"platform", "users"}
    )
    if invalid_roots:
        raise ManagementCliError(
            "secret_import_namespace_invalid",
            "secrets.yaml must contain only the platform and users roots.",
        )
    flattened: dict[str, str] = {}
    platform = platform_root.get("platform")
    if platform is not None:
        if not isinstance(platform, Mapping):
            raise ManagementCliError(
                "secret_import_descriptor_invalid",
                "secrets.yaml platform must be a mapping.",
            )
        _flatten_values("platform", platform, flattened)

    users = platform_root.get("users")
    if users is not None:
        if not isinstance(users, Mapping):
            raise ManagementCliError(
                "secret_import_descriptor_invalid",
                "secrets.yaml users must be a mapping.",
            )
        for raw_user_id, raw_user in users.items():
            user_id = str(raw_user_id or "").strip()
            if not user_id or not isinstance(raw_user, Mapping):
                raise ManagementCliError(
                    "secret_import_descriptor_invalid",
                    "Every user secret entry must have an exact id and mapping.",
                )
            if set(raw_user) - {"secrets", "bundles"}:
                raise ManagementCliError(
                    "secret_import_descriptor_invalid",
                    "A user secret entry may contain only secrets and bundles.",
                )
            direct = raw_user.get("secrets")
            if direct is not None:
                if not isinstance(direct, Mapping):
                    raise ManagementCliError(
                        "secret_import_descriptor_invalid",
                        "A user secrets block must be a mapping.",
                    )
                _flatten_values(f"users.{user_id}.secrets", direct, flattened)
            user_bundles = raw_user.get("bundles")
            if user_bundles is None:
                continue
            if not isinstance(user_bundles, Mapping):
                raise ManagementCliError(
                    "secret_import_descriptor_invalid",
                    "A user bundles block must be a mapping.",
                )
            for raw_bundle_id, raw_bundle in user_bundles.items():
                bundle_id = str(raw_bundle_id or "").strip()
                if (
                    not bundle_id
                    or not isinstance(raw_bundle, Mapping)
                    or set(raw_bundle) - {"secrets"}
                ):
                    raise ManagementCliError(
                        "secret_import_descriptor_invalid",
                        "Every user application entry must contain one secrets mapping.",
                    )
                secrets = raw_bundle.get("secrets")
                if not isinstance(secrets, Mapping):
                    raise ManagementCliError(
                        "secret_import_descriptor_invalid",
                        "A user application secrets block must be a mapping.",
                    )
                _flatten_values(
                    f"users.{user_id}.bundles.{bundle_id}.secrets",
                    secrets,
                    flattened,
                )

    if set(bundles_data) != {"bundles"} or not isinstance(
        bundles_data.get("bundles"), Mapping
    ):
        raise ManagementCliError(
            "secret_import_descriptor_invalid",
            "bundles.secrets.yaml must contain one bundles mapping.",
        )
    bundles_root = bundles_data["bundles"]
    if set(bundles_root) - {"version", "items"}:
        raise ManagementCliError(
            "secret_import_descriptor_invalid",
            "bundles.secrets.yaml bundles may contain only version and items.",
        )
    items = bundles_root.get("items")
    if not isinstance(items, list):
        raise ManagementCliError(
            "secret_import_descriptor_invalid",
            "bundles.secrets.yaml items must be a list.",
        )
    seen_bundle_ids: set[str] = set()
    for item in items:
        if not isinstance(item, Mapping) or set(item) - {"id", "secrets"}:
            raise ManagementCliError(
                "secret_import_descriptor_invalid",
                "Every bundle secret entry must contain only id and secrets.",
            )
        bundle_id = str(item.get("id") or "").strip()
        if not bundle_id or bundle_id in seen_bundle_ids:
            raise ManagementCliError(
                "secret_import_descriptor_invalid",
                "Every bundle secret entry requires one unique id.",
            )
        seen_bundle_ids.add(bundle_id)
        secrets = item.get("secrets")
        if secrets is not None:
            if not isinstance(secrets, Mapping):
                raise ManagementCliError(
                    "secret_import_descriptor_invalid",
                    "A bundle secrets block must be a mapping.",
                )
            _flatten_values(f"bundles.{bundle_id}.secrets", secrets, flattened)

    values: list[ExportedSecret] = []
    counts = {"platform": 0, "bundle": 0, "user": 0}
    for provider_key, value in sorted(flattened.items()):
        target = ManagementSecretTarget.from_provider_key(provider_key)
        values.append(ExportedSecret(target=target, value=value))
        counts[target.scope] += 1
    if len(values) > MAX_SECRET_DESCRIPTOR_VALUES:
        raise ManagementCliError(
            "secret_import_inventory_too_large",
            "The secret descriptor pair contains more than 4096 values.",
        )
    if not values and not allow_empty:
        raise ManagementCliError(
            "secret_import_empty",
            "The descriptor pair contains no secret values.",
        )
    return SecretDescriptorImport(
        directory=directory,
        values=tuple(values),
        platform_count=counts["platform"],
        bundle_count=counts["bundle"],
        user_count=counts["user"],
    )


def _render(
    values: Sequence[ExportedSecret],
) -> tuple[bytes, bytes, int, int, int]:
    # Keep the ordinary descriptor grammar canonical even when the selected
    # export currently contains only user-owned values.
    global_secrets: dict[str, Any] = {"platform": {}}
    bundles: dict[str, dict[str, Any]] = {}
    identities: set[tuple[str, str, str, str]] = set()
    platform_count = 0
    bundle_count = 0
    user_count = 0
    for exported in sorted(values, key=lambda item: item.target.identity):
        if exported.target.identity in identities:
            raise ManagementCliError(
                "secret_export_descriptor_conflict",
                "The secret export contains a duplicate descriptor key.",
            )
        identities.add(exported.target.identity)
        if exported.target.scope == "platform":
            _insert(global_secrets, exported.target.key, exported.value)
            platform_count += 1
            continue
        if exported.target.scope == "user":
            users = global_secrets.setdefault("users", {})
            user = users.setdefault(exported.target.user_id, {})
            if exported.target.bundle_id:
                user_bundles = user.setdefault("bundles", {})
                user_bundle = user_bundles.setdefault(
                    exported.target.bundle_id,
                    {},
                )
                secret_root = user_bundle.setdefault("secrets", {})
            else:
                secret_root = user.setdefault("secrets", {})
            _insert(secret_root, exported.target.key, exported.value)
            user_count += 1
            continue
        bundle = bundles.setdefault(exported.target.bundle_id, {})
        _insert(bundle, exported.target.key, exported.value)
        bundle_count += 1

    platform_text = yaml.safe_dump(
        global_secrets,
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
        user_count,
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


def validate_existing_secret_descriptor_directory(
    output_directory: Path,
    targets: Sequence[ManagementSecretTarget],
    *,
    replace: bool,
) -> Path:
    """Validate an existing descriptor directory before provider disclosure."""

    target = output_directory.expanduser().absolute()
    if target.is_symlink() or not target.is_dir():
        raise ManagementCliError(
            "secret_export_output_directory_invalid",
            "The secret export target must be an existing descriptor directory.",
        )
    for name in ("secrets.yaml", "bundles.secrets.yaml"):
        path = target / name
        if path.is_symlink():
            raise ManagementCliError(
                "secret_export_output_invalid",
                "A secret descriptor output path must not be a symbolic link.",
            )
        if path.exists() and (not path.is_file() or not replace):
            code = (
                "secret_export_output_invalid"
                if not path.is_file()
                else "secret_export_output_exists"
            )
            message = (
                "A secret descriptor output path is not a regular file."
                if not path.is_file()
                else "The descriptor directory already contains secret files; use the explicit replace option."
            )
            raise ManagementCliError(code, message)
    _render([ExportedSecret(target=item, value="") for item in targets])
    return target


def _write_pair_into_existing_directory(
    target: Path,
    *,
    platform: bytes,
    bundles: bytes,
) -> None:
    temporary = Path(
        tempfile.mkdtemp(prefix=".kdcube-secret-export.", suffix=".tmp", dir=target)
    )
    names = ("secrets.yaml", "bundles.secrets.yaml")
    backups: dict[str, Path] = {}
    replaced: list[str] = []
    try:
        if os.name != "nt":
            os.chmod(temporary, 0o700)
            os.chmod(target, 0o700)
        _write_private(temporary / "secrets.yaml", platform)
        _write_private(temporary / "bundles.secrets.yaml", bundles)
        for name in names:
            destination = target / name
            if destination.exists():
                backup = temporary / f"{name}.previous"
                _write_private(backup, destination.read_bytes())
                backups[name] = backup
        _sync_directory(temporary)
        for name in names:
            os.replace(temporary / name, target / name)
            replaced.append(name)
        _sync_directory(target)
    except Exception as exc:
        for name in reversed(replaced):
            try:
                (target / name).unlink(missing_ok=True)
            except OSError:
                pass
        for name, backup in backups.items():
            try:
                os.replace(backup, target / name)
            except OSError:
                pass
        try:
            _sync_directory(target)
        except OSError:
            pass
        raise ManagementCliError(
            "secret_export_output_write_failed",
            "The secret descriptor export could not be written.",
        ) from exc
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


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
    platform, bundles, platform_count, bundle_count, user_count = _render(values)
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
        user_count=user_count,
    )


def write_secret_descriptors_into_directory(
    output_directory: Path,
    values: Sequence[ExportedSecret],
    *,
    replace: bool = False,
) -> SecretDescriptorExport:
    """Write the literal pair beside ordinary descriptors, without another package."""

    target = validate_existing_secret_descriptor_directory(
        output_directory,
        [item.target for item in values],
        replace=replace,
    )
    platform, bundles, platform_count, bundle_count, user_count = _render(values)
    _write_pair_into_existing_directory(
        target,
        platform=platform,
        bundles=bundles,
    )
    return SecretDescriptorExport(
        directory=target,
        platform_path=target / "secrets.yaml",
        bundles_path=target / "bundles.secrets.yaml",
        platform_count=platform_count,
        bundle_count=bundle_count,
        user_count=user_count,
    )


__all__ = [
    "SecretDescriptorExport",
    "SecretDescriptorImport",
    "load_secret_descriptors",
    "validate_existing_secret_descriptor_directory",
    "validate_secret_descriptor_export",
    "write_secret_descriptors",
    "write_secret_descriptors_into_directory",
]
