from __future__ import annotations

import os
import tempfile
from pathlib import Path

from kdcube_cli.management.errors import ManagementCliError
from kdcube_cli.management.filesystem import apply_open_file_mode


def _sync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_private_secret_output(
    path: Path,
    *,
    replace: bool = False,
) -> Path:
    """Validate a disclosure target without creating or following it."""

    target = path.expanduser().absolute()
    parent = target.parent
    if not parent.is_dir():
        raise ManagementCliError(
            "secret_output_directory_missing",
            "The secret output directory does not exist.",
        )
    target_present = target.exists() or target.is_symlink()
    if target_present and not replace:
        raise ManagementCliError(
            "secret_output_exists",
            "The secret output file already exists; use --replace to replace it.",
        )
    if target_present and not target.is_symlink() and not target.is_file():
        raise ManagementCliError(
            "secret_output_not_regular_file",
            "The secret output path is not a regular file.",
        )
    return target


def write_private_secret(
    path: Path,
    value: str,
    *,
    replace: bool = False,
) -> Path:
    """Atomically write one disclosed secret without rendering its value."""

    target = validate_private_secret_output(path, replace=replace)
    parent = target.parent

    encoded = value.encode("utf-8")
    descriptor = -1
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=parent,
        )
        temporary = Path(temporary_name)
        apply_open_file_mode(descriptor, temporary, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())

        if replace:
            os.replace(temporary, target)
            temporary = None
        else:
            try:
                os.link(temporary, target)
            except FileExistsError as exc:
                raise ManagementCliError(
                    "secret_output_exists",
                    "The secret output file already exists; use --replace to replace it.",
                ) from exc
            except OSError as exc:
                raise ManagementCliError(
                    "secret_output_atomic_write_unavailable",
                    "The filesystem cannot create the secret output atomically.",
                ) from exc
            temporary.unlink()
            temporary = None
        os.chmod(target, 0o600)
        _sync_directory(parent)
        return target
    except ManagementCliError:
        raise
    except Exception as exc:
        raise ManagementCliError(
            "secret_output_write_failed",
            "The secret output file could not be written.",
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


__all__ = ["validate_private_secret_output", "write_private_secret"]
