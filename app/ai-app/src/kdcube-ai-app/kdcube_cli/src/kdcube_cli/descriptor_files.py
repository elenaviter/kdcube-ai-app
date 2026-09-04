# SPDX-License-Identifier: MIT
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path


SECRET_DESCRIPTOR_FILENAMES = frozenset({"secrets.yaml", "bundles.secrets.yaml"})
_OWNER_ONLY_MODE = 0o600


def is_secret_descriptor(path: Path) -> bool:
    return path.name in SECRET_DESCRIPTOR_FILENAMES


def enforce_secret_descriptor_permissions(path: Path) -> None:
    if os.name == "posix" and is_secret_descriptor(path) and path.exists():
        path.chmod(_OWNER_ONLY_MODE)


def _write_owner_only_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=str(path.parent))
    temp_path = Path(temp_name)
    try:
        if os.name == "posix":
            os.fchmod(fd, _OWNER_ONLY_MODE)
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(path)
        enforce_secret_descriptor_permissions(path)
    finally:
        if fd >= 0:
            os.close(fd)
        temp_path.unlink(missing_ok=True)


def write_descriptor_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    if is_secret_descriptor(path):
        _write_owner_only_bytes(path, text.encode(encoding))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding=encoding)


def copy_descriptor_file(source: Path, target: Path) -> None:
    if is_secret_descriptor(target):
        _write_owner_only_bytes(target, source.read_bytes())
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
