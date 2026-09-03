# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Mutual-TLS transport for the host vault: stdlib ``ssl`` + HTTP/1.1 JSON.

Server: requires a client certificate chained to the host CA
(``CERT_REQUIRED``), hands the verified peer certificate to the service.
Client: presents the deployment certificate and key, verifies the vault's
server certificate against the same CA, sends one JSON request per POST.

What the transport does NOT do: it never trusts a header, a deployment id in
the body, the socket address, or a bearer as identity. A caller bearer sent
in ``Authorization`` is ignored; the only identity input is the TLS peer
certificate, and the service still consults the live trust registry for it.
"""

from __future__ import annotations

import json
import ssl
import threading
from dataclasses import dataclass
import socket
from http.client import HTTPSConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from kdcube_ai_app.infra.secrets.host_vault.protocol import ErrorCode, VaultError, VaultRequest, VaultResponse

VAULT_PATH = "/v1/vault"
MAX_BODY_BYTES = 256 * 1024


@dataclass(frozen=True)
class ServerTLS:
    """Files the host service owns."""

    cert_file: Path
    key_file: Path
    ca_file: Path

    def context(self) -> ssl.SSLContext:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        ctx.load_cert_chain(str(self.cert_file), str(self.key_file))
        ctx.load_verify_locations(str(self.ca_file))
        ctx.verify_mode = ssl.CERT_REQUIRED
        return ctx


@dataclass(frozen=True)
class ClientTLS:
    """The appliance identity mount as the broker sees it."""

    cert_file: Path
    key_file: Path
    ca_file: Path

    def context(self, *, server_hostname_check: bool = True) -> ssl.SSLContext:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        ctx.load_verify_locations(str(self.ca_file))
        ctx.load_cert_chain(str(self.cert_file), str(self.key_file))
        ctx.check_hostname = server_hostname_check
        ctx.verify_mode = ssl.CERT_REQUIRED
        return ctx


Handler = Callable[[Any, bytes | None], VaultResponse]


def _peer_cert_pem(connection: ssl.SSLSocket) -> bytes | None:
    der = connection.getpeercert(binary_form=True)
    if not der:
        return None
    return ssl.DER_cert_to_PEM_cert(der).encode("ascii")


class _VaultHTTPHandler(BaseHTTPRequestHandler):
    server_version = "kdcube-host-vault/1"
    handler: Handler  # set per server class

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        # No request logging: bodies carry values, and the audit sink is the record.
        return

    def _reply(self, status: int, response: VaultResponse) -> None:
        data = json.dumps(response.to_wire()).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:  # noqa: N802 - stdlib name
        if self.path != VAULT_PATH:
            self._reply(404, VaultResponse.failure(VaultError(ErrorCode.INVALID_REQUEST, "unknown path")))
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY_BYTES:
            self._reply(413, VaultResponse.failure(VaultError(ErrorCode.TOO_LARGE)))
            return
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:  # noqa: BLE001
            self._reply(400, VaultResponse.failure(VaultError(ErrorCode.INVALID_REQUEST)))
            return
        peer = _peer_cert_pem(self.connection) if isinstance(self.connection, ssl.SSLSocket) else None
        response = type(self).handler(body, peer)
        self._reply(200 if response.ok else 403 if response.code in (ErrorCode.UNAUTHENTICATED, ErrorCode.FORBIDDEN) else 400, response)


class HostVaultServer:
    """Serve one ``HostVaultService`` over mTLS. ``serve_in_thread`` for tests
    and the thin deployment entrypoint alike."""

    def __init__(self, *, tls: ServerTLS, handler: Handler, host: str = "127.0.0.1", port: int = 0) -> None:
        handler_cls = type("BoundVaultHandler", (_VaultHTTPHandler,), {"handler": staticmethod(handler)})
        self._server = ThreadingHTTPServer((host, port), handler_cls)
        self._server.socket = tls.context().wrap_socket(self._server.socket, server_side=True)
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def serve_in_thread(self) -> None:
        self._thread = threading.Thread(target=self._server.serve_forever, name="host-vault", daemon=True)
        self._thread.start()

    def serve_forever(self) -> None:
        self._server.serve_forever()

    def shutdown(self) -> None:
        self._server.shutdown()
        self._server.server_close()


class HostVaultClient:
    """The broker's client: one request, one response, over a fresh mTLS
    connection. Bearer headers are never sent; identity is the certificate."""

    def __init__(self, *, host: str, port: int, tls: ClientTLS, server_hostname: str | None = None,
                 timeout: float = 10.0, hostname_check: bool = True) -> None:
        self._host = host
        self._port = port
        self._tls = tls
        self._server_hostname = server_hostname or host
        self._timeout = timeout
        self._hostname_check = hostname_check

    def _connect(self) -> HTTPSConnection:
        """An HTTPS connection whose TLS server-name check uses the vault's
        configured name even when the socket address is an IP or an alias."""
        context = self._tls.context(server_hostname_check=self._hostname_check)
        server_hostname = self._server_hostname

        class _NamedHTTPSConnection(HTTPSConnection):
            def connect(self) -> None:  # noqa: D401 - stdlib override
                raw = socket.create_connection((self.host, self.port), self.timeout)
                self.sock = context.wrap_socket(raw, server_hostname=server_hostname)

        return _NamedHTTPSConnection(self._host, self._port, timeout=self._timeout, context=context)

    def call(self, request: VaultRequest) -> VaultResponse:
        connection = self._connect()
        try:
            data = json.dumps(request.to_wire()).encode("utf-8")
            connection.request("POST", VAULT_PATH, body=data, headers={"Content-Type": "application/json"})
            raw = connection.getresponse().read()
        except Exception as exc:  # noqa: BLE001 - TLS/socket text never reaches callers
            raise VaultError(ErrorCode.BACKEND_UNAVAILABLE, detail=type(exc).__name__) from exc
        finally:
            connection.close()
        try:
            return VaultResponse.from_wire(json.loads(raw.decode("utf-8")))
        except VaultError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise VaultError(ErrorCode.INTERNAL, detail=type(exc).__name__) from exc


__all__ = [
    "ClientTLS",
    "HostVaultClient",
    "HostVaultServer",
    "MAX_BODY_BYTES",
    "ServerTLS",
    "VAULT_PATH",
]
