# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Durable host vault for KDCube provider secrets.

The local Compose runtime can select this flow explicitly while retaining the
existing secrets-service contract:

    trusted Connection Hub app in KDCube
      -> KDCube ISecretsManager (secrets-service)
      -> deployment-local kdcube-secrets broker        broker.py
      -> mTLS with enrolled deployment workload key    identity.py, transport.py
      -> durable encrypted host vault                  service.py, storage.py, keys.py
      -> raw value returns to trusted KDCube process

Modules:

- ``protocol``: the versioned wire contract (operations, request envelope,
  fixed error codes, secret references, replay controls, sanitization).
- ``keys``: root-key custody adapter and envelope encryption (data keys
  wrapped by a versioned root key). The only in-memory provider is labeled
  fake, for tests.
- ``storage``: the durable encrypted store: atomic commit, generations,
  crash recovery, bounded records, fail-closed on corruption.
- ``audit``: append-only, secret-safe audit records.
- ``identity``: deployment enrollment (CSR -> certificate), live trust
  registry with namespace ACLs, rotation overlap, revocation, expiry.
- ``service``: the vault service: authenticate the presented workload
  certificate, authorize the exact namespace, apply replay controls, run the
  store operation, audit.
- ``transport``: mTLS server and client over stdlib ``ssl``/``http``.
- ``broker``: the stateless ``kdcube-secrets`` adapter that translates the
  existing internal secrets-service operations into vault requests.

Selecting the backend does not migrate existing secret values. Migration and
plaintext cleanup remain explicit operator actions after readback verification.
"""

from kdcube_ai_app.infra.secrets.host_vault.protocol import (
    PROTOCOL_VERSION,
    ErrorCode,
    Operation,
    SecretNamespace,
    SecretReference,
    VaultError,
    VaultRequest,
    VaultResponse,
)

__all__ = [
    "PROTOCOL_VERSION",
    "ErrorCode",
    "Operation",
    "SecretNamespace",
    "SecretReference",
    "VaultError",
    "VaultRequest",
    "VaultResponse",
]
