from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Protocol

from kdcube_cli.management.credentials import normalize_bearer
from kdcube_cli.management.errors import ManagementCliError
from kdcube_cli.management.models import ManagementRequest

MAX_MANAGEMENT_RESPONSE_BYTES = 2 * 1024 * 1024


class ManagementTransport(Protocol):
    async def execute(
        self,
        request: ManagementRequest,
        bearer: str,
    ) -> tuple[int, Mapping[str, Any]]: ...


class HttpxManagementTransport:
    def __init__(self, *, transport: Any = None, timeout_seconds: float = 60.0) -> None:
        self._transport = transport
        self._timeout_seconds = max(1.0, min(float(timeout_seconds), 300.0))

    async def execute(
        self,
        request: ManagementRequest,
        bearer: str,
    ) -> tuple[int, Mapping[str, Any]]:
        credential = normalize_bearer(bearer)
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {credential}",
            "Idempotency-Key": request.invocation_id,
        }
        json_body: Mapping[str, Any] | None = None
        if request.method == "POST":
            headers["Content-Type"] = "application/json"
            json_body = request.body
        try:
            import httpx2

            async with (
                httpx2.AsyncClient(
                    timeout=httpx2.Timeout(self._timeout_seconds),
                    follow_redirects=False,
                    transport=self._transport,
                    trust_env=False,
                ) as client,
                client.stream(
                    request.method,
                    request.url,
                    headers=headers,
                    json=json_body,
                ) as response,
            ):
                if response.status_code < 200 or response.status_code >= 600:
                    raise ManagementCliError(
                        "management_http_status_invalid",
                        "KDCube returned an invalid management status.",
                    )
                if 300 <= response.status_code < 400:
                    raise ManagementCliError(
                        "management_redirect_rejected",
                        "KDCube attempted to redirect a management request.",
                    )
                length = response.headers.get("content-length")
                if length:
                    try:
                        if int(length) > MAX_MANAGEMENT_RESPONSE_BYTES:
                            raise ManagementCliError(
                                "management_response_too_large",
                                "The KDCube management response is too large.",
                            )
                    except ValueError:
                        pass
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > MAX_MANAGEMENT_RESPONSE_BYTES:
                        raise ManagementCliError(
                            "management_response_too_large",
                            "The KDCube management response is too large.",
                        )
                status = int(response.status_code)
        except ManagementCliError:
            raise
        # Custom HTTP transports may raise backend-specific exceptions. Their
        # text can include request headers, so only a fixed failure is exposed.
        except Exception:  # noqa: BLE001
            raise ManagementCliError(
                "management_request_failed",
                "The selected KDCube management service could not be reached.",
            ) from None
        try:
            payload = json.loads(bytes(body))
        except (UnicodeError, ValueError):
            raise ManagementCliError(
                "management_response_invalid",
                "KDCube returned an invalid management response.",
            ) from None
        if not isinstance(payload, Mapping):
            raise ManagementCliError(
                "management_response_invalid",
                "KDCube returned an invalid management response.",
            )
        return status, dict(payload)
