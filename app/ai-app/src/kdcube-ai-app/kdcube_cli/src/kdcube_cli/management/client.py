from __future__ import annotations

from kdcube_cli.management.models import (
    ManagementDenial,
    ManagementRequest,
    ManagementResult,
)
from kdcube_cli.management.transport import ManagementTransport


class ManagementClient:
    def __init__(self, *, transport: ManagementTransport) -> None:
        self._transport = transport

    async def execute(
        self,
        request: ManagementRequest,
        *,
        bearer: str,
    ) -> ManagementResult | ManagementDenial:
        status, payload = await self._transport.execute(request, bearer)
        if 200 <= status < 300:
            return ManagementResult.from_mapping(payload, request=request)
        return ManagementDenial.from_mapping(
            payload,
            status=status,
            request=request,
        )
