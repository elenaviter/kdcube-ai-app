# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Deployment workload identity for the host vault.

Model (the 2026-09-03 17:46 contract):

1. a privileged host installer creates the service-owned appliance;
2. the deployment generates its private key INSIDE the protected boundary
   (``DeploymentKey.generate``); the key never leaves that boundary;
3. a one-use provisioning channel accepts a CSR (``EnrollmentTicket``);
4. the host CA issues a deployment certificate and records the namespace ACL
   in the trust registry;
5. the broker proves key possession through mTLS on every connection (the
   transport verifies the chain; the registry maps the presented
   certificate's fingerprint to a LIVE record);
6. rotation issues a new certificate that overlaps the old for a bounded
   interval;
7. revocation blocks the next connection;
8. certificate, deployment id, socket address, and process name alone never
   establish authority: only a chain-valid certificate whose fingerprint the
   registry lists as active does, and the ACL then decides the namespace.

Certificates and CA here are real X.509 (``cryptography``), so the same code
serves fake test material and a host CA. Nothing claims a production local
boundary: that needs the service-owned appliance.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from kdcube_ai_app.infra.secrets.host_vault.protocol import ErrorCode, SecretNamespace, VaultError

try:  # pragma: no cover - import guard
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID
except Exception as exc:  # noqa: BLE001
    x509 = None  # type: ignore[assignment]
    _CRYPTO_IMPORT_ERROR: BaseException | None = exc
else:
    _CRYPTO_IMPORT_ERROR = None

DEFAULT_CERT_DAYS = 90
ROTATION_OVERLAP_SECONDS = 24 * 3600


def _require_crypto() -> None:
    if x509 is None:
        raise VaultError(ErrorCode.BACKEND_UNAVAILABLE, detail=f"cryptography unavailable: {type(_CRYPTO_IMPORT_ERROR).__name__}")


def fingerprint_of(cert_pem: bytes) -> str:
    """SHA-256 of the DER certificate, hex. The registry key."""
    _require_crypto()
    cert = x509.load_pem_x509_certificate(cert_pem)
    return hashlib.sha256(cert.public_bytes(serialization.Encoding.DER)).hexdigest()


# ── the host CA ───────────────────────────────────────────────────────────


class HostIssuingCA:
    """The host service's local issuing CA. Its private key is service
    custody; in tests it is generated in memory and labeled so."""

    def __init__(self, *, key_pem: bytes, cert_pem: bytes) -> None:
        _require_crypto()
        self._key = serialization.load_pem_private_key(key_pem, password=None)
        self._cert = x509.load_pem_x509_certificate(cert_pem)

    @classmethod
    def generate(cls, *, common_name: str = "kdcube-hostd issuing CA", days: int = 3650) -> "HostIssuingCA":
        _require_crypto()
        key = ec.generate_private_key(ec.SECP256R1())
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])
        now = dt.datetime.now(dt.timezone.utc)
        cert = (
            x509.CertificateBuilder()
            .subject_name(name)
            .issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(minutes=5))
            .not_valid_after(now + dt.timedelta(days=days))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(x509.KeyUsage(
                digital_signature=True, key_cert_sign=True, crl_sign=True, content_commitment=False,
                key_encipherment=False, data_encipherment=False, key_agreement=False,
                encipher_only=False, decipher_only=False,
            ), critical=True)
            .sign(key, hashes.SHA256())
        )
        return cls(
            key_pem=key.private_bytes(
                serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
            ),
            cert_pem=cert.public_bytes(serialization.Encoding.PEM),
        )

    @property
    def cert_pem(self) -> bytes:
        return self._cert.public_bytes(serialization.Encoding.PEM)

    @property
    def key_pem(self) -> bytes:
        """Service custody only: the operator tool writes it at 0400."""
        return self._key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        )

    def issue(self, csr_pem: bytes, *, deployment_id: str, days: int = DEFAULT_CERT_DAYS) -> bytes:
        """Issue a client certificate for a CSR. The subject is REWRITTEN to the
        deployment id the host assigned; the CSR's own subject is not trusted."""
        csr = x509.load_pem_x509_csr(csr_pem)
        if not csr.is_signature_valid:
            raise VaultError(ErrorCode.INVALID_REQUEST, "CSR signature is invalid.")
        now = dt.datetime.now(dt.timezone.utc)
        subject = x509.Name([
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "kdcube-deployment"),
            x509.NameAttribute(NameOID.COMMON_NAME, deployment_id),
        ])
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(self._cert.subject)
            .public_key(csr.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(minutes=5))
            .not_valid_after(now + dt.timedelta(days=days))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.ExtendedKeyUsage([x509.ExtendedKeyUsageOID.CLIENT_AUTH]), critical=False)
            .sign(self._key, hashes.SHA256())
        )
        return cert.public_bytes(serialization.Encoding.PEM)

    def issue_server(self, *, hostnames: list[str], days: int = DEFAULT_CERT_DAYS) -> tuple[bytes, bytes]:
        """A server certificate for the vault endpoint (key + cert PEM)."""
        key = ec.generate_private_key(ec.SECP256R1())
        now = dt.datetime.now(dt.timezone.utc)
        subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, hostnames[0])])
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(self._cert.subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(minutes=5))
            .not_valid_after(now + dt.timedelta(days=days))
            .add_extension(x509.SubjectAlternativeName(_san_entries(hostnames)), critical=False)
            .add_extension(x509.ExtendedKeyUsage([x509.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .sign(self._key, hashes.SHA256())
        )
        return (
            key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()),
            cert.public_bytes(serialization.Encoding.PEM),
        )


def _san_entries(hostnames: list[str]) -> list:
    import ipaddress

    entries = []
    for name in hostnames:
        try:
            entries.append(x509.IPAddress(ipaddress.ip_address(name)))
        except ValueError:
            entries.append(x509.DNSName(name))
    return entries


# ── the deployment side ───────────────────────────────────────────────────


class DeploymentKey:
    """The deployment private key KD-1. Generated inside the protected
    boundary; exported only as PEM to the identity mount the appliance owns,
    never returned to a CLI, Connection Hub, or a user session."""

    def __init__(self, key_pem: bytes) -> None:
        _require_crypto()
        self._key = serialization.load_pem_private_key(key_pem, password=None)

    @classmethod
    def generate(cls) -> "DeploymentKey":
        _require_crypto()
        key = ec.generate_private_key(ec.SECP256R1())
        return cls(key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        ))

    @property
    def pem(self) -> bytes:
        return self._key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        )

    def csr(self, *, requested_name: str = "kdcube-deployment") -> bytes:
        return (
            x509.CertificateSigningRequestBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, requested_name)]))
            .sign(self._key, hashes.SHA256())
            .public_bytes(serialization.Encoding.PEM)
        )

    def write_identity_files(self, directory: Path, *, cert_pem: bytes, ca_pem: bytes) -> None:
        """The three appliance identity files; the key at 0400."""
        directory.mkdir(parents=True, exist_ok=True)
        key_path = directory / "host-vault-client.key"
        fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o400)
        try:
            os.write(fd, self.pem)
            os.fsync(fd)
        finally:
            os.close(fd)
        (directory / "host-vault-client.crt").write_bytes(cert_pem)
        (directory / "host-vault-ca.crt").write_bytes(ca_pem)


# ── enrollment and trust registry ─────────────────────────────────────────


@dataclass
class TrustRecord:
    deployment_id: str
    fingerprint: str
    not_after: float
    namespaces: tuple[str, ...]  # "<tenant>/<project>/<application>" or "<tenant>/<project>/*"
    status: str = "active"  # active | revoked
    issued_at: float = field(default_factory=time.time)
    revoked_at: float | None = None
    supersedes: str = ""  # fingerprint of the certificate this one rotates out

    def to_dict(self) -> dict[str, Any]:
        return {
            "deployment_id": self.deployment_id,
            "fingerprint": self.fingerprint,
            "not_after": self.not_after,
            "namespaces": list(self.namespaces),
            "status": self.status,
            "issued_at": self.issued_at,
            "revoked_at": self.revoked_at,
            "supersedes": self.supersedes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrustRecord":
        return cls(
            deployment_id=str(data["deployment_id"]),
            fingerprint=str(data["fingerprint"]),
            not_after=float(data["not_after"]),
            namespaces=tuple(str(item) for item in data.get("namespaces") or ()),
            status=str(data.get("status") or "active"),
            issued_at=float(data.get("issued_at") or 0.0),
            revoked_at=float(data["revoked_at"]) if data.get("revoked_at") is not None else None,
            supersedes=str(data.get("supersedes") or ""),
        )

    def allows(self, namespace: SecretNamespace) -> bool:
        wanted = namespace.path
        for rule in self.namespaces:
            if rule == wanted:
                return True
            if rule.endswith("/*") and wanted.startswith(rule[:-1]):
                return True
        return False


@dataclass(frozen=True)
class EnrollmentTicket:
    """The one-use provisioning channel token, owned by the host runtime
    controller. It is minted per appliance creation, consumed on first use,
    and never printed by the CLI or written into the ordinary workdir."""

    ticket_id: str
    deployment_id: str
    namespaces: tuple[str, ...]
    expires_at: float


class TrustRegistry:
    """The host vault's live trust registry: which certificate fingerprints
    identify which deployment, with which namespace ACL, in which status.
    Persisted as one JSON document, atomically replaced on every change."""

    def __init__(self, path: Path | None = None, *, ca: HostIssuingCA | None = None) -> None:
        """``ca`` is needed to issue (enroll, rotate); a server that only
        identifies presented certificates opens the registry without it."""
        self._path = Path(path) if path else None
        self._ca = ca
        self._lock = threading.RLock()
        self._records: dict[str, TrustRecord] = {}
        self._tickets: dict[str, EnrollmentTicket] = {}
        self._stamp: tuple[int, int] | None = None
        self._load()

    # persistence
    def _file_stamp(self) -> tuple[int, int] | None:
        if self._path is None:
            return None
        try:
            st = self._path.stat()
        except FileNotFoundError:
            return None
        return (st.st_mtime_ns, st.st_size)

    def _load(self) -> None:
        if self._path is None or not self._path.is_file():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - a corrupt registry trusts nobody
            raise VaultError(ErrorCode.BACKEND_UNAVAILABLE, detail=f"trust registry unreadable: {type(exc).__name__}") from exc
        records = {}
        for row in data.get("records") or []:
            record = TrustRecord.from_dict(row)
            records[record.fingerprint] = record
        self._records = records
        self._stamp = self._file_stamp()

    def _reload_if_changed(self) -> None:
        """Operator edits (revoke, enroll) through another process land on the
        next identification: the ACL is live, never a boot-time snapshot."""
        if self._path is None:
            return
        with self._lock:
            if self._file_stamp() != self._stamp:
                self._load()

    def _issuer(self) -> HostIssuingCA:
        if self._ca is None:
            raise VaultError(ErrorCode.BACKEND_UNAVAILABLE, detail="registry opened without the issuing CA")
        return self._ca

    def _save(self) -> None:
        if self._path is None:
            return
        payload = {"format": "kdcube-host-vault-trust/1", "records": [r.to_dict() for r in self._records.values()]}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".json.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, json.dumps(payload, sort_keys=True).encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, self._path)
        self._stamp = self._file_stamp()

    @property
    def ca_cert_pem(self) -> bytes:
        return self._issuer().cert_pem

    # enrollment
    def mint_ticket(self, *, deployment_id: str, namespaces: list[str], ttl_seconds: int = 600) -> EnrollmentTicket:
        ticket = EnrollmentTicket(
            ticket_id=hashlib.sha256(os.urandom(32)).hexdigest(),
            deployment_id=deployment_id,
            namespaces=tuple(namespaces),
            expires_at=time.time() + ttl_seconds,
        )
        with self._lock:
            self._tickets[ticket.ticket_id] = ticket
        return ticket

    def enroll(self, *, ticket_id: str, csr_pem: bytes, days: int = DEFAULT_CERT_DAYS) -> tuple[bytes, TrustRecord]:
        """Consume a one-use ticket, issue the certificate, record the ACL."""
        self._reload_if_changed()
        with self._lock:
            ticket = self._tickets.pop(ticket_id, None)
            if ticket is None:
                raise VaultError(ErrorCode.UNAUTHENTICATED, "enrollment ticket is unknown or already used.")
            if ticket.expires_at < time.time():
                raise VaultError(ErrorCode.UNAUTHENTICATED, "enrollment ticket expired.")
            cert_pem = self._issuer().issue(csr_pem, deployment_id=ticket.deployment_id, days=days)
            record = TrustRecord(
                deployment_id=ticket.deployment_id,
                fingerprint=fingerprint_of(cert_pem),
                not_after=time.time() + days * 86400,
                namespaces=ticket.namespaces,
            )
            self._records[record.fingerprint] = record
            self._save()
            return cert_pem, record

    def rotate(self, *, current_fingerprint: str, csr_pem: bytes, days: int = DEFAULT_CERT_DAYS,
               overlap_seconds: int = ROTATION_OVERLAP_SECONDS) -> tuple[bytes, TrustRecord]:
        """A new key, enrolled by an active identity for the same deployment.
        The old certificate stays valid for a bounded overlap, then is
        lapses at that instant."""
        self._reload_if_changed()
        with self._lock:
            current = self._records.get(current_fingerprint)
            if current is None or current.status != "active" or current.not_after < time.time():
                raise VaultError(ErrorCode.UNAUTHENTICATED, "rotation requires an active identity.")
            cert_pem = self._issuer().issue(csr_pem, deployment_id=current.deployment_id, days=days)
            record = TrustRecord(
                deployment_id=current.deployment_id,
                fingerprint=fingerprint_of(cert_pem),
                not_after=time.time() + days * 86400,
                namespaces=current.namespaces,
                supersedes=current.fingerprint,
            )
            current.not_after = min(current.not_after, time.time() + overlap_seconds)
            self._records[record.fingerprint] = record
            self._save()
            return cert_pem, record

    def revoke(self, fingerprint: str) -> None:
        self._reload_if_changed()
        with self._lock:
            record = self._records.get(fingerprint)
            if record is None:
                raise VaultError(ErrorCode.NOT_FOUND, "no such identity.")
            record.status = "revoked"
            record.revoked_at = time.time()
            self._save()

    # authentication
    def identify(self, cert_pem: bytes, *, now: float | None = None) -> TrustRecord:
        """The live record for a PRESENTED certificate. The transport has
        already verified the chain and proven key possession (mTLS); this is
        the registry half: known fingerprint, active, not expired here."""
        moment = now if now is not None else time.time()
        fingerprint = fingerprint_of(cert_pem)
        self._reload_if_changed()
        with self._lock:
            record = self._records.get(fingerprint)
        if record is None:
            raise VaultError(ErrorCode.UNAUTHENTICATED, detail="unknown fingerprint")
        if record.status != "active":
            raise VaultError(ErrorCode.UNAUTHENTICATED, detail="revoked")
        if record.not_after < moment:
            raise VaultError(ErrorCode.UNAUTHENTICATED, detail="expired")
        return record

    def records(self) -> list[TrustRecord]:
        with self._lock:
            return list(self._records.values())


__all__ = [
    "DEFAULT_CERT_DAYS",
    "DeploymentKey",
    "EnrollmentTicket",
    "HostIssuingCA",
    "ROTATION_OVERLAP_SECONDS",
    "TrustRecord",
    "TrustRegistry",
    "fingerprint_of",
]
