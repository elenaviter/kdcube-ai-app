from __future__ import annotations

import copy
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class SecretNamespaceMigrationError(RuntimeError):
    pass


CANONICAL_GLOBAL_SECRET_ROOTS = frozenset({"platform", "users"})


@dataclass(frozen=True)
class SecretNamespaceMigrationResult:
    legacy_roots: tuple[str, ...]
    moved_platform_keys: int
    moved_bundle_keys: int
    equal_duplicates: int
    conflicts: tuple[str, ...]
    changed: bool
    dry_run: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "kdcube_cli.secret_namespace_migration.v1",
            "ok": not self.conflicts,
            "legacy_roots": list(self.legacy_roots),
            "moved_platform_keys": self.moved_platform_keys,
            "moved_bundle_keys": self.moved_bundle_keys,
            "equal_duplicates": self.equal_duplicates,
            "conflicts": list(self.conflicts),
            "changed": self.changed,
            "dry_run": self.dry_run,
            "canonical_roots": ["platform", "users"],
        }


def _load(path: Path, *, required: bool) -> dict[str, Any]:
    if path.is_symlink():
        raise SecretNamespaceMigrationError(
            f"{path.name} must not be a symbolic link."
        )
    if not path.exists():
        if required:
            raise SecretNamespaceMigrationError(f"{path.name} is missing.")
        return {}
    if not path.is_file():
        raise SecretNamespaceMigrationError(f"{path.name} is not a regular file.")
    if os.name == "posix" and path.stat().st_mode & 0o077:
        raise SecretNamespaceMigrationError(
            f"{path.name} must be owner-only (mode 0600)."
        )
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise SecretNamespaceMigrationError(
            f"{path.name} could not be read safely."
        ) from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise SecretNamespaceMigrationError(
            f"{path.name} must contain a mapping."
        )
    return loaded


def _descriptor_root(document: dict[str, Any]) -> dict[str, Any]:
    wrapped = document.get("secrets")
    if isinstance(wrapped, dict):
        return wrapped
    return document


def canonical_global_secret_root(document: dict[str, Any]) -> dict[str, Any]:
    """Return a descriptor's mutable canonical root or refuse legacy names."""

    root = _descriptor_root(document)
    legacy = sorted(
        str(key) for key in root if str(key) not in CANONICAL_GLOBAL_SECRET_ROOTS
    )
    if legacy:
        raise SecretNamespaceMigrationError(
            "secrets.yaml contains legacy top-level roots; run "
            "`kdcube secrets namespace migrate --dry-run` and then rerun "
            "with `--yes`."
        )
    return root


def _leaf_count(node: Any) -> int:
    if isinstance(node, dict):
        return sum(_leaf_count(value) for value in node.values())
    return 1


def _merge(
    destination: dict[str, Any],
    key: str,
    source: Any,
    *,
    identity: str,
    conflicts: list[str],
) -> tuple[int, int]:
    if key not in destination:
        destination[key] = copy.deepcopy(source)
        return _leaf_count(source), 0
    current = destination[key]
    if isinstance(current, dict) and isinstance(source, dict):
        moved = 0
        equal = 0
        for child, value in source.items():
            child_name = str(child)
            child_identity = f"{identity}.{child_name}"
            child_moved, child_equal = _merge(
                current,
                child_name,
                value,
                identity=child_identity,
                conflicts=conflicts,
            )
            moved += child_moved
            equal += child_equal
        return moved, equal
    if current == source:
        return 0, _leaf_count(source)
    conflicts.append(identity)
    return 0, 0


def _legacy_bundle_leaves(
    node: Any,
    *,
    parts: tuple[str, ...] = (),
) -> list[tuple[str, tuple[str, ...], Any]]:
    if isinstance(node, dict):
        leaves: list[tuple[str, tuple[str, ...], Any]] = []
        for key, value in node.items():
            leaves.extend(
                _legacy_bundle_leaves(
                    value,
                    parts=(*parts, str(key)),
                )
            )
        return leaves
    try:
        marker = parts.index("secrets")
    except ValueError as exc:
        raise SecretNamespaceMigrationError(
            "The legacy bundles root contains a key outside a bundle secrets path."
        ) from exc
    bundle_id = ".".join(parts[:marker]).strip()
    tail = parts[marker + 1 :]
    if not bundle_id or not tail:
        raise SecretNamespaceMigrationError(
            "The legacy bundles root contains an incomplete bundle secret path."
        )
    return [(bundle_id, tail, copy.deepcopy(node))]


def _bundle_items(document: dict[str, Any]) -> list[dict[str, Any]]:
    root = document.setdefault("bundles", {})
    if not isinstance(root, dict):
        raise SecretNamespaceMigrationError(
            "bundles.secrets.yaml bundles must be a mapping."
        )
    root.setdefault("version", "1")
    items = root.setdefault("items", [])
    if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
        raise SecretNamespaceMigrationError(
            "bundles.secrets.yaml items must be a list of mappings."
        )
    return items


def _bundle_secrets(
    document: dict[str, Any],
    *,
    bundle_id: str,
) -> dict[str, Any]:
    items = _bundle_items(document)
    item = next(
        (entry for entry in items if str(entry.get("id") or "") == bundle_id),
        None,
    )
    if item is None:
        item = {"id": bundle_id, "secrets": {}}
        items.append(item)
    secrets = item.setdefault("secrets", {})
    if not isinstance(secrets, dict):
        raise SecretNamespaceMigrationError(
            f"Bundle {bundle_id} has a non-mapping secrets block."
        )
    return secrets


def _insert_bundle_leaf(
    destination: dict[str, Any],
    tail: tuple[str, ...],
    value: Any,
    *,
    identity: str,
    conflicts: list[str],
) -> tuple[int, int]:
    cursor = destination
    for segment in tail[:-1]:
        existing = cursor.get(segment)
        if existing is None:
            existing = {}
            cursor[segment] = existing
        if not isinstance(existing, dict):
            conflicts.append(identity)
            return 0, 0
        cursor = existing
    return _merge(
        cursor,
        tail[-1],
        value,
        identity=identity,
        conflicts=conflicts,
    )


def _write_private_yaml(path: Path, payload: dict[str, Any]) -> None:
    rendered = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        if os.name == "posix":
            os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name == "posix":
            path.chmod(0o600)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def migrate_platform_secret_namespace(
    config_dir: Path,
    *,
    dry_run: bool,
) -> SecretNamespaceMigrationResult:
    """Move legacy global roots under platform and misplaced bundles out."""

    config = Path(config_dir).expanduser().resolve()
    platform_path = config / "secrets.yaml"
    bundles_path = config / "bundles.secrets.yaml"
    platform_document = _load(platform_path, required=True)
    bundle_document = _load(bundles_path, required=False)
    root = _descriptor_root(platform_document)
    platform = root.setdefault("platform", {})
    if not isinstance(platform, dict):
        raise SecretNamespaceMigrationError(
            "secrets.yaml platform must be a mapping."
        )
    users = root.get("users")
    if users is not None and not isinstance(users, dict):
        raise SecretNamespaceMigrationError("secrets.yaml users must be a mapping.")

    legacy_roots = tuple(
        sorted(
            str(key)
            for key in root
            if str(key) not in CANONICAL_GLOBAL_SECRET_ROOTS
        )
    )
    conflicts: list[str] = []
    moved_platform = 0
    moved_bundle = 0
    equal = 0

    legacy_bundles = root.get("bundles")
    if legacy_bundles is not None:
        for bundle_id, tail, value in _legacy_bundle_leaves(legacy_bundles):
            identity = f"bundles.{bundle_id}.secrets.{'.'.join(tail)}"
            moved, duplicates = _insert_bundle_leaf(
                _bundle_secrets(bundle_document, bundle_id=bundle_id),
                tail,
                value,
                identity=identity,
                conflicts=conflicts,
            )
            moved_bundle += moved
            equal += duplicates

    for legacy_root in legacy_roots:
        if legacy_root == "bundles":
            continue
        moved, duplicates = _merge(
            platform,
            legacy_root,
            root[legacy_root],
            identity=f"platform.{legacy_root}",
            conflicts=conflicts,
        )
        moved_platform += moved
        equal += duplicates

    conflicts = sorted(set(conflicts))
    changed = bool(legacy_roots) and not conflicts
    result = SecretNamespaceMigrationResult(
        legacy_roots=legacy_roots,
        moved_platform_keys=moved_platform,
        moved_bundle_keys=moved_bundle,
        equal_duplicates=equal,
        conflicts=tuple(conflicts),
        changed=changed,
        dry_run=dry_run,
    )
    if conflicts or dry_run or not changed:
        return result

    # Write the bundle destination first. A crash before the global write leaves
    # duplicate equal values, which a rerun recognizes, rather than losing data.
    if legacy_bundles is not None:
        _write_private_yaml(bundles_path, bundle_document)
    for legacy_root in legacy_roots:
        root.pop(legacy_root, None)
    _write_private_yaml(platform_path, platform_document)
    return result


__all__ = [
    "CANONICAL_GLOBAL_SECRET_ROOTS",
    "SecretNamespaceMigrationError",
    "SecretNamespaceMigrationResult",
    "canonical_global_secret_root",
    "migrate_platform_secret_namespace",
]
