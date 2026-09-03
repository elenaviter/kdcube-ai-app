# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Root-key custody and envelope encryption for the host vault.

Custody stays behind ``RootKeyProvider``: the vault never reads the root key
from a KDCube descriptor, the runtime workdir, an environment variable, or a
test fixture. Two providers ship:

- ``FileRootKeyProvider``: root keys in a service-owned directory outside the
  runtime workdir (the platform installer creates it, mode 0700/0400). This
  is the reference host adapter; a hardware or OS keystore adapter can
  replace it behind the same interface.
- ``FakeInMemoryRootKeyProvider``: LABELED FAKE, for portable tests only.

Encryption is envelope-style: each committed record carries its value under a
fresh 256-bit data key (AES-256-GCM), and that data key is wrapped by the
root key of a named version. Rotating the root key rewraps data keys; values
are never re-encrypted and never appear in memory during rotation. Every
ciphertext binds its record identity (reference digest + generation) as
associated data, so a record copied to another reference fails to open.
"""

from __future__ import annotations

import base64
import json
import os
import re
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

from kdcube_ai_app.infra.secrets.host_vault.protocol import ErrorCode, VaultError

try:  # pragma: no cover - import guard
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except Exception as exc:  # noqa: BLE001
    AESGCM = None  # type: ignore[assignment]
    _CRYPTO_IMPORT_ERROR: BaseException | None = exc
else:
    _CRYPTO_IMPORT_ERROR = None

KEY_BYTES = 32
NONCE_BYTES = 12
_KEY_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def _require_crypto() -> None:
    if AESGCM is None:
        raise VaultError(
            ErrorCode.BACKEND_UNAVAILABLE,
            detail=f"cryptography unavailable: {type(_CRYPTO_IMPORT_ERROR).__name__}",
        )


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _unb64(text: str) -> bytes:
    try:
        return base64.b64decode(text, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise VaultError(ErrorCode.CORRUPT_RECORD, detail="base64") from exc


class RootKeyProvider(Protocol):
    """Where root keys live. ``current`` names the key new records wrap with;
    ``key`` returns an older version so committed records stay readable
    across rotations; ``rotate`` mints a new current key (custody-specific)."""

    def current_key_id(self) -> str: ...

    def key(self, key_id: str) -> bytes: ...

    def rotate(self) -> str: ...


class FakeInMemoryRootKeyProvider:
    """FAKE root-key provider for portable tests. Keys live in process memory
    and vanish with it. Never a custody choice for a deployment."""

    is_fake = True

    def __init__(self) -> None:
        self._keys: dict[str, bytes] = {}
        self._current = ""
        self.rotate()

    def current_key_id(self) -> str:
        return self._current

    def key(self, key_id: str) -> bytes:
        try:
            return self._keys[key_id]
        except KeyError as exc:
            raise VaultError(ErrorCode.CORRUPT_RECORD, detail="unknown root key version") from exc

    def rotate(self) -> str:
        key_id = f"fake-{len(self._keys) + 1:03d}"
        self._keys[key_id] = secrets.token_bytes(KEY_BYTES)
        self._current = key_id
        return key_id


class FileRootKeyProvider:
    """Root keys as files in a service-owned directory.

    Layout: ``<dir>/<key_id>.key`` (raw 32 bytes, mode 0400) and
    ``<dir>/CURRENT`` naming the active id. The directory must be outside the
    KDCube runtime workdir and owned by the vault service user; this class
    refuses group/other-readable key files so a misplaced key fails closed
    rather than silently serving."""

    is_fake = False

    def __init__(self, directory: Path) -> None:
        self._dir = Path(directory)

    def _key_path(self, key_id: str) -> Path:
        if not _KEY_ID_RE.match(key_id):
            raise VaultError(ErrorCode.CORRUPT_RECORD, detail="key id grammar")
        return self._dir / f"{key_id}.key"

    def current_key_id(self) -> str:
        marker = self._dir / "CURRENT"
        if not marker.is_file():
            raise VaultError(ErrorCode.BACKEND_UNAVAILABLE, detail="no current root key")
        key_id = marker.read_text(encoding="utf-8").strip()
        if not _KEY_ID_RE.match(key_id):
            raise VaultError(ErrorCode.BACKEND_UNAVAILABLE, detail="current marker grammar")
        return key_id

    def key(self, key_id: str) -> bytes:
        path = self._key_path(key_id)
        if not path.is_file():
            raise VaultError(ErrorCode.CORRUPT_RECORD, detail="unknown root key version")
        mode = path.stat().st_mode & 0o777
        if mode & 0o077:
            raise VaultError(ErrorCode.BACKEND_UNAVAILABLE, detail="root key file is group/other readable")
        data = path.read_bytes()
        if len(data) != KEY_BYTES:
            raise VaultError(ErrorCode.CORRUPT_RECORD, detail="root key length")
        return data

    def rotate(self) -> str:
        self._dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self._dir, 0o700)
        existing = sorted(p.stem for p in self._dir.glob("*.key"))
        key_id = f"root-{len(existing) + 1:04d}"
        path = self._key_path(key_id)
        tmp = path.with_suffix(".key.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        try:
            os.write(fd, secrets.token_bytes(KEY_BYTES))
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)
        marker = self._dir / "CURRENT"
        marker_tmp = self._dir / "CURRENT.tmp"
        marker_tmp.write_text(key_id, encoding="utf-8")
        os.replace(marker_tmp, marker)
        _fsync_dir(self._dir)
        return key_id


def _fsync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@dataclass(frozen=True)
class SealedValue:
    """A value at rest: ciphertext under a data key, the data key wrapped by a
    named root-key version. Serializable without secrets."""

    root_key_id: str
    wrapped_data_key: str  # base64: nonce || AESGCM(root, data_key, aad)
    ciphertext: str  # base64: nonce || AESGCM(data_key, value, aad)

    def to_dict(self) -> dict[str, str]:
        return {
            "root_key_id": self.root_key_id,
            "wrapped_data_key": self.wrapped_data_key,
            "ciphertext": self.ciphertext,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "SealedValue":
        try:
            return cls(
                root_key_id=str(data["root_key_id"]),
                wrapped_data_key=str(data["wrapped_data_key"]),
                ciphertext=str(data["ciphertext"]),
            )
        except (KeyError, TypeError) as exc:
            raise VaultError(ErrorCode.CORRUPT_RECORD, detail="sealed value shape") from exc


class Envelope:
    """Seal and open values with a root-key provider."""

    def __init__(self, keys: RootKeyProvider) -> None:
        _require_crypto()
        self._keys = keys

    @staticmethod
    def _aad(record_id: str) -> bytes:
        return json.dumps({"v": 1, "record": record_id}, sort_keys=True).encode("utf-8")

    def seal(self, value: bytes, *, record_id: str) -> SealedValue:
        data_key = secrets.token_bytes(KEY_BYTES)
        aad = self._aad(record_id)
        value_nonce = secrets.token_bytes(NONCE_BYTES)
        ciphertext = AESGCM(data_key).encrypt(value_nonce, value, aad)
        root_id = self._keys.current_key_id()
        wrap_nonce = secrets.token_bytes(NONCE_BYTES)
        wrapped = AESGCM(self._keys.key(root_id)).encrypt(wrap_nonce, data_key, aad)
        return SealedValue(
            root_key_id=root_id,
            wrapped_data_key=_b64(wrap_nonce + wrapped),
            ciphertext=_b64(value_nonce + ciphertext),
        )

    def _unwrap(self, sealed: SealedValue, *, record_id: str) -> bytes:
        aad = self._aad(record_id)
        blob = _unb64(sealed.wrapped_data_key)
        if len(blob) <= NONCE_BYTES:
            raise VaultError(ErrorCode.CORRUPT_RECORD, detail="wrapped key length")
        try:
            return AESGCM(self._keys.key(sealed.root_key_id)).decrypt(blob[:NONCE_BYTES], blob[NONCE_BYTES:], aad)
        except VaultError:
            raise
        except Exception as exc:  # noqa: BLE001 - any AEAD failure is corruption or wrong key
            raise VaultError(ErrorCode.CORRUPT_RECORD, detail="data key unwrap") from exc

    def open(self, sealed: SealedValue, *, record_id: str) -> bytes:
        data_key = self._unwrap(sealed, record_id=record_id)
        aad = self._aad(record_id)
        blob = _unb64(sealed.ciphertext)
        if len(blob) <= NONCE_BYTES:
            raise VaultError(ErrorCode.CORRUPT_RECORD, detail="ciphertext length")
        try:
            return AESGCM(data_key).decrypt(blob[:NONCE_BYTES], blob[NONCE_BYTES:], aad)
        except Exception as exc:  # noqa: BLE001
            raise VaultError(ErrorCode.CORRUPT_RECORD, detail="value open") from exc

    def rewrap(self, sealed: SealedValue, *, record_id: str) -> SealedValue:
        """Root-key rotation for one record: unwrap the data key with its
        recorded root version, wrap it again with the current one. The value
        ciphertext is untouched and never decrypted."""
        current = self._keys.current_key_id()
        if sealed.root_key_id == current:
            return sealed
        data_key = self._unwrap(sealed, record_id=record_id)
        aad = self._aad(record_id)
        wrap_nonce = secrets.token_bytes(NONCE_BYTES)
        wrapped = AESGCM(self._keys.key(current)).encrypt(wrap_nonce, data_key, aad)
        return SealedValue(
            root_key_id=current,
            wrapped_data_key=_b64(wrap_nonce + wrapped),
            ciphertext=sealed.ciphertext,
        )


__all__ = [
    "Envelope",
    "FakeInMemoryRootKeyProvider",
    "FileRootKeyProvider",
    "KEY_BYTES",
    "RootKeyProvider",
    "SealedValue",
]
