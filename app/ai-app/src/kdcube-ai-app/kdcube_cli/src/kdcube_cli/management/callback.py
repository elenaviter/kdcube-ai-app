from __future__ import annotations

import html
import secrets
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from kdcube_cli.management.errors import ManagementCliError
from kdcube_cli.management.validation import validate_web_url


@dataclass(frozen=True)
class AuthorizationCallback:
    code: str
    issuer: str | None = None


class _CallbackState:
    def __init__(
        self,
        *,
        path: str,
        expected_state: str,
        expected_issuer: str,
        issuer_required: bool,
    ) -> None:
        self.path = path
        self.expected_state = expected_state
        self.expected_issuer = expected_issuer
        self.issuer_required = issuer_required
        self.event = threading.Event()
        self.lock = threading.Lock()
        self.callback: AuthorizationCallback | None = None
        self.error: ManagementCliError | None = None

    def complete(self, query: str) -> tuple[int, str]:
        try:
            values = parse_qs(
                query,
                keep_blank_values=True,
                strict_parsing=True,
                max_num_fields=16,
            )
        except ValueError:
            return 400, "The authorization response is invalid."
        state_values = values.get("state", [])
        if len(state_values) != 1 or not secrets.compare_digest(
            state_values[0], self.expected_state
        ):
            return 400, "The authorization response state is invalid."
        with self.lock:
            if self.event.is_set():
                return 409, "This authorization response was already received."
            provider_errors = values.get("error", [])
            code_values = values.get("code", [])
            issuer_values = values.get("iss", [])
            if provider_errors:
                if len(provider_errors) != 1:
                    return 400, "The authorization response is invalid."
                self.error = ManagementCliError(
                    "oauth_authorization_denied",
                    "The authorization server did not approve this request.",
                )
                self.event.set()
                return 400, "Authorization was not approved."
            if len(code_values) != 1 or not code_values[0]:
                return 400, "The authorization response has no code."
            if len(issuer_values) > 1:
                return 400, "The authorization response issuer is invalid."
            issuer = issuer_values[0].rstrip("/") if issuer_values else ""
            if issuer and not secrets.compare_digest(issuer, self.expected_issuer):
                return 400, "The authorization response issuer is invalid."
            if self.issuer_required and not issuer:
                return 400, "The authorization response has no issuer."
            self.callback = AuthorizationCallback(
                code=code_values[0],
                issuer=issuer or None,
            )
            self.event.set()
            return 200, (
                "Authorization response received. You may close this tab. Return to "
                "the terminal to confirm setup."
            )

    def close(self) -> None:
        with self.lock:
            if self.event.is_set():
                return
            self.error = ManagementCliError(
                "oauth_callback_closed",
                "The OAuth callback was closed before authorization completed.",
            )
            self.event.set()


def _handler_for(state: _CallbackState) -> type[BaseHTTPRequestHandler]:
    class CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if len(self.path) > 8192:
                self._respond(414, "The authorization response is too large.")
                return
            parsed = urlsplit(self.path)
            if parsed.path != state.path:
                self._respond(404, "This callback address is not active.")
                return
            status, message = state.complete(parsed.query)
            self._respond(status, message)

        def _respond(self, status: int, message: str) -> None:
            body = (
                '<!doctype html><html><head><meta charset="utf-8">'
                "<title>KDCube</title></head><body><p>"
                f"{html.escape(message)}"
                "</p></body></html>"
            ).encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Pragma", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *args: object) -> None:
            return

    return CallbackHandler


class LoopbackCallbackServer:
    def __init__(
        self,
        *,
        expected_state: str,
        expected_issuer: str,
        issuer_required: bool,
        port: int = 0,
    ) -> None:
        state = str(expected_state or "").strip()
        if not state:
            raise ManagementCliError(
                "oauth_callback_configuration_invalid",
                "The OAuth callback configuration is invalid.",
            )
        issuer = validate_web_url(
            expected_issuer,
            code="oauth_callback_configuration_invalid",
            allow_query=False,
        ).rstrip("/")
        try:
            selected_port = int(port)
        except (TypeError, ValueError):
            raise ManagementCliError(
                "oauth_callback_configuration_invalid",
                "The OAuth callback port is invalid.",
            ) from None
        if selected_port < 0 or selected_port > 65535:
            raise ManagementCliError(
                "oauth_callback_configuration_invalid",
                "The OAuth callback port is invalid.",
            )
        # KDCube's native-client DCR policy allows an ephemeral loopback port
        # while keeping the registered callback path exact.
        path = "/callback"
        self._state = _CallbackState(
            path=path,
            expected_state=state,
            expected_issuer=issuer,
            issuer_required=issuer_required,
        )
        try:
            self._server = ThreadingHTTPServer(
                ("127.0.0.1", selected_port),
                _handler_for(self._state),
            )
        except OSError as exc:
            raise ManagementCliError(
                "oauth_callback_unavailable",
                "KDCube CLI could not open a local authorization callback.",
            ) from exc
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="kdcube-management-callback",
            daemon=True,
        )
        self._closed = False
        self._thread.start()

    @property
    def redirect_uri(self) -> str:
        port = int(self._server.server_address[1])
        return f"http://127.0.0.1:{port}{self._state.path}"

    def wait(self, *, timeout_seconds: float = 300.0) -> AuthorizationCallback:
        timeout = max(1.0, min(float(timeout_seconds), 1800.0))
        if not self._state.event.wait(timeout):
            raise ManagementCliError(
                "oauth_callback_timeout",
                "Timed out waiting for browser authorization.",
            )
        if self._state.error is not None:
            raise self._state.error
        if self._state.callback is None:
            raise ManagementCliError(
                "oauth_callback_invalid",
                "The browser returned an invalid authorization response.",
            )
        return self._state.callback

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._state.close()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2.0)

    # typing.Self is unavailable on the package's Python 3.9 floor.
    def __enter__(self) -> LoopbackCallbackServer:  # noqa: PYI034
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
