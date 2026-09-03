# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Durable encrypted secret store.

``DurableSecretStore`` is the service-owned storage interface; ``FileDurableSecretStore``
is the reference implementation: one file per record under a service-owned
directory, values sealed by ``keys.Envelope``.

Durability rules the implementation keeps:

- a mutation is acknowledged only after its record file is written to a
  candidate path, fsynced, atomically renamed over the committed path, and
  the directory entry is fsynced. A crash before the rename leaves the
  previous committed record untouched and a stray candidate that the next
  load ignores and removes;
- every record carries a generation; a replace or delete may name the
  generation it expects and fails with ``conflict`` when the committed one
  moved (concurrent replacement), so two brokers cannot silently overwrite;
- a delete commits a tombstone (generation advances, no value) rather than
  removing the file, so a deletion is as durable as a write and a later
  create continues the generation sequence;
- integrity: the record carries a digest over its metadata and sealed value;
  a mismatch, an unreadable file, an unknown key version, or an AEAD failure
  fails closed as ``corrupt_record``; nothing is guessed;
- bounds: value bytes and record size are capped at the protocol bound.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

from kdcube_ai_app.infra.secrets.host_vault.keys import Envelope, RootKeyProvider, SealedValue
from kdcube_ai_app.infra.secrets.host_vault.protocol import (
    MAX_VALUE_BYTES,
    ErrorCode,
    SecretReference,
    VaultError,
)

RECORD_FORMAT = "kdcube-host-vault-record/1"
MAX_RECORD_BYTES = MAX_VALUE_BYTES * 2 + 4096


@dataclass(frozen=True)
class StoredRecord:
    """What the store returns: never the plaintext by default."""

    reference_digest: str
    generation: int
    deleted: bool
    committed_at: float
    root_key_id: str


class DurableSecretStore(Protocol):
    def get(self, reference: SecretReference) -> tuple[StoredRecord, bytes] | None: ...

    def put(self, reference: SecretReference, value: bytes, *, expected_generation: int | None) -> StoredRecord: ...

    def delete(self, reference: SecretReference, *, expected_generation: int | None) -> StoredRecord: ...

    def rewrap_all(self) -> int: ...


def _fsync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class FileDurableSecretStore:
    """Reference durable store. ``root`` must be a service-owned directory
    outside the KDCube runtime workdir."""

    CANDIDATE_SUFFIX = ".candidate"

    def __init__(self, root: Path, keys: RootKeyProvider) -> None:
        self._root = Path(root)
        self._envelope = Envelope(keys)
        self._lock = threading.RLock()
        self._root.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self._root, 0o700)
        except OSError:
            pass
        self.recover()

    # ── layout ────────────────────────────────────────────────────────────

    def _path(self, reference: SecretReference) -> Path:
        # Two-level fan-out on the digest keeps directories bounded; the
        # digest, not the name, is the file name, so names never touch the fs.
        digest = reference.digest
        return self._root / digest[:2] / f"{digest}.json"

    @staticmethod
    def _record_id(reference: SecretReference, generation: int) -> str:
        return f"{reference.digest}#{generation}"

    def recover(self) -> int:
        """Crash recovery: drop candidates that never reached commit. Called
        on start; safe to call any time. Returns the number removed."""
        removed = 0
        for candidate in self._root.rglob(f"*{self.CANDIDATE_SUFFIX}"):
            try:
                candidate.unlink()
                removed += 1
            except OSError:
                pass
        return removed

    # ── record codec ──────────────────────────────────────────────────────

    @staticmethod
    def _digest(payload: dict) -> str:
        material = json.dumps({k: v for k, v in payload.items() if k != "integrity"}, sort_keys=True)
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _read_raw(self, reference: SecretReference) -> dict | None:
        path = self._path(reference)
        if not path.is_file():
            return None
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise VaultError(ErrorCode.BACKEND_UNAVAILABLE, detail=type(exc).__name__) from exc
        if len(raw) > MAX_RECORD_BYTES:
            raise VaultError(ErrorCode.CORRUPT_RECORD, detail="record too large")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise VaultError(ErrorCode.CORRUPT_RECORD, detail="record json") from exc
        if not isinstance(payload, dict) or payload.get("format") != RECORD_FORMAT:
            raise VaultError(ErrorCode.CORRUPT_RECORD, detail="record format")
        if payload.get("reference_digest") != reference.digest:
            raise VaultError(ErrorCode.CORRUPT_RECORD, detail="record identity")
        if self._digest(payload) != payload.get("integrity"):
            raise VaultError(ErrorCode.CORRUPT_RECORD, detail="record integrity")
        return payload

    def _write_committed(self, reference: SecretReference, payload: dict) -> None:
        payload = dict(payload)
        payload["integrity"] = self._digest(payload)
        data = json.dumps(payload, sort_keys=True).encode("utf-8")
        if len(data) > MAX_RECORD_BYTES:
            raise VaultError(ErrorCode.TOO_LARGE)
        path = self._path(reference)
        path.parent.mkdir(parents=True, exist_ok=True)
        candidate = path.with_name(path.name + self.CANDIDATE_SUFFIX)
        try:
            fd = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                os.write(fd, data)
                os.fsync(fd)
            finally:
                os.close(fd)
            self._commit_hook()
            os.replace(candidate, path)
            _fsync_dir(path.parent)
        except VaultError:
            raise
        except OSError as exc:
            raise VaultError(ErrorCode.BACKEND_UNAVAILABLE, detail=type(exc).__name__) from exc

    def _commit_hook(self) -> None:
        """Test seam: raising here simulates a crash between the candidate
        write and the atomic commit."""

    @staticmethod
    def _record(payload: dict) -> StoredRecord:
        return StoredRecord(
            reference_digest=str(payload["reference_digest"]),
            generation=int(payload["generation"]),
            deleted=bool(payload.get("deleted")),
            committed_at=float(payload.get("committed_at") or 0.0),
            root_key_id=str((payload.get("sealed") or {}).get("root_key_id") or ""),
        )

    # ── operations ────────────────────────────────────────────────────────

    def get(self, reference: SecretReference) -> tuple[StoredRecord, bytes] | None:
        with self._lock:
            payload = self._read_raw(reference)
            if payload is None or payload.get("deleted"):
                return None
            record = self._record(payload)
            sealed = SealedValue.from_dict(payload.get("sealed") or {})
            value = self._envelope.open(sealed, record_id=self._record_id(reference, record.generation))
            return record, value

    def _check_generation(self, payload: dict | None, expected: int | None) -> int:
        current = int(payload["generation"]) if payload else 0
        if expected is not None and expected != current:
            raise VaultError(ErrorCode.CONFLICT)
        return current

    def put(self, reference: SecretReference, value: bytes, *, expected_generation: int | None) -> StoredRecord:
        if len(value) > MAX_VALUE_BYTES:
            raise VaultError(ErrorCode.TOO_LARGE)
        with self._lock:
            payload = self._read_raw(reference)
            generation = self._check_generation(payload, expected_generation) + 1
            sealed = self._envelope.seal(value, record_id=self._record_id(reference, generation))
            new_payload = {
                "format": RECORD_FORMAT,
                "reference_digest": reference.digest,
                "generation": generation,
                "deleted": False,
                "committed_at": time.time(),
                "sealed": sealed.to_dict(),
            }
            self._write_committed(reference, new_payload)
            return self._record(new_payload)

    def delete(self, reference: SecretReference, *, expected_generation: int | None) -> StoredRecord:
        with self._lock:
            payload = self._read_raw(reference)
            if payload is None or payload.get("deleted"):
                if expected_generation is not None:
                    self._check_generation(payload, expected_generation)
                raise VaultError(ErrorCode.NOT_FOUND)
            generation = self._check_generation(payload, expected_generation) + 1
            tombstone = {
                "format": RECORD_FORMAT,
                "reference_digest": reference.digest,
                "generation": generation,
                "deleted": True,
                "committed_at": time.time(),
                "sealed": {},
            }
            self._write_committed(reference, tombstone)
            return self._record(tombstone)

    def _all_paths(self) -> Iterable[Path]:
        return sorted(self._root.rglob("*.json"))

    def rewrap_all(self) -> int:
        """Root-key rotation: rewrap every live record's data key under the
        current root key. Values are never decrypted. Returns rewrapped count.
        A record that fails to unwrap is left as is and counted nowhere; it
        surfaces as corrupt on its next read, which is the honest state."""
        count = 0
        with self._lock:
            for path in self._all_paths():
                try:
                    payload = json.loads(path.read_bytes().decode("utf-8"))
                except Exception:  # noqa: BLE001
                    continue
                if not isinstance(payload, dict) or payload.get("deleted"):
                    continue
                digest = str(payload.get("reference_digest") or "")
                generation = int(payload.get("generation") or 0)
                try:
                    sealed = SealedValue.from_dict(payload.get("sealed") or {})
                    rewrapped = self._envelope.rewrap(sealed, record_id=f"{digest}#{generation}")
                except VaultError:
                    continue
                if rewrapped is sealed:
                    continue
                payload["sealed"] = rewrapped.to_dict()
                payload = {k: v for k, v in payload.items() if k != "integrity"}
                pseudo = _DigestOnlyReference(digest)
                self._write_committed(pseudo, payload)  # type: ignore[arg-type]
                count += 1
        return count


class _DigestOnlyReference:
    """Path selector for records visited by digest during rotation (the store
    never learns names from disk, and rewrap needs none)."""

    def __init__(self, digest: str) -> None:
        self.digest = digest


__all__ = [
    "DurableSecretStore",
    "FileDurableSecretStore",
    "RECORD_FORMAT",
    "StoredRecord",
]
