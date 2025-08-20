# fastapi_gateway_adapter.py
"""
FastAPI adapter for the simplified gateway
"""
from contextlib import asynccontextmanager

from fastapi import Request, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

from kdcube_ai_app.infra.gateway.gateway import (
    RequestGateway
)
from kdcube_ai_app.infra.gateway.backpressure import BackpressureError
from kdcube_ai_app.infra.gateway.rate_limiter import RateLimitError
from kdcube_ai_app.infra.gateway.definitions import GatewayError
from kdcube_ai_app.auth.sessions import UserType, UserSession, RequestContext
from kdcube_ai_app.auth.AuthManager import RequirementBase, AuthenticationError, AuthorizationError, RequireUser, \
    RequireRoles
from kdcube_ai_app.infra.namespaces import CONFIG


class CircuitBreakerStatusResponse(BaseModel):
    name: str
    state: str
    failure_count: int
    success_count: int
    total_requests: int
    total_failures: int
    consecutive_failures: int
    current_window_failures: int
    last_failure_time: Optional[float]
    last_success_time: Optional[float]
    opened_at: Optional[float]

class CircuitBreakerSummaryResponse(BaseModel):
    total_circuits: int
    open_circuits: int
    half_open_circuits: int
    closed_circuits: int

class CircuitBreakersResponse(BaseModel):
    summary: CircuitBreakerSummaryResponse
    circuits: Dict[str, CircuitBreakerStatusResponse]

STATE_FLAG = "_gw_processed"
STATE_SESSION = "user_session"
STATE_USER_TYPE = "user_type"

class FastAPIGatewayAdapter:
    """FastAPI adapter for the request gateway"""

    def __init__(self, gateway: RequestGateway):
        self.gateway = gateway

    def _extract_context(self, request: Request) -> RequestContext:
        """Extract request context from FastAPI request"""
        return RequestContext(
            client_ip=request.client.host if request.client else "unknown",
            user_agent=request.headers.get("user-agent", ""),
            authorization_header=request.headers.get("authorization"),
            id_token=request.headers.get(CONFIG.ID_TOKEN_HEADER_NAME),
        )

    async def process_request(self,
                              request: Request,
                              requirements: List[RequirementBase] = None,
                              bypass_throttling: bool = False) -> UserSession:
        """Process request and return session"""
        context = self._extract_context(request)
        endpoint = request.url.path

        try:
            session = await self.gateway.process_request(
                context,
                requirements,
                endpoint,
                bypass_throttling
            )
            return session


        except AuthenticationError as e:
            raise HTTPException(status_code=401, detail=e.message)
        except AuthorizationError as e:
            raise HTTPException(status_code=403, detail=e.message)
        except RateLimitError as e:
            sess = e.session
            content={"detail": e.message, "user_type": sess.user_type.value}
            raise HTTPException(
                status_code=429,
                detail=content,
                headers={"Retry-After": str(e.retry_after)}
            )
        except BackpressureError as e:
            sess = e.session
            content={"detail": e.message, "user_type": sess.user_type.value}
            raise HTTPException(
                status_code=503,
                detail=content,
                headers={"Retry-After": str(e.retry_after)}
            )
        except GatewayError as e:
            raise HTTPException(status_code=e.code, detail=e.message)

    def require_admin(self, *requirements: RequirementBase):
        """Create FastAPI dependency that enforces requirements with throttling bypass"""
        async def dependency(request: Request) -> UserSession:
            return await self.process_request(request, list(requirements), bypass_throttling=True)
        return dependency

    def require(self, *requirements: RequirementBase):
        """Create FastAPI dependency that enforces requirements"""
        async def dependency(request: Request) -> UserSession:
            return await self.process_request(request, list(requirements))
        return dependency

    def get_session(self):
        """Create FastAPI dependency that just gets the session (no requirements)"""
        async def dependency(request: Request) -> UserSession:
            return await self.process_request(request, [])
        return dependency


    # --- your dependency ---
    def get_user_session_dependency(self):
        async def dependency(request: Request) -> UserSession:
            # If middleware already did it, reuse
            existing: Optional[UserSession] = getattr(request.state, STATE_SESSION, None)
            if existing is not None:
                return existing

            # Otherwise process once here and mark
            session = await self.process_request(request, [])
            setattr(request.state, STATE_SESSION, session)
            setattr(request.state, STATE_USER_TYPE, session.user_type.value)
            setattr(request.state, STATE_FLAG, True)
            return session
        return dependency

    def auth_without_pressure(self):
        """
        Authenticate + authorize for admin endpoints, bypass throttling for privileged users
        """
        async def dependency(request: Request) -> UserSession:

            existing: Optional[UserSession] = getattr(request.state, STATE_SESSION, None)
            if existing is not None:
                return existing

            # Use admin bypass for monitoring/admin endpoints
            session = await self.process_request(
                request,
                [RequireUser(), RequireRoles("kdcube:role:super-admin")],
                bypass_throttling=True
            )
            setattr(request.state, STATE_SESSION, session)
            setattr(request.state, STATE_USER_TYPE, session.user_type.value)
            setattr(request.state, STATE_FLAG, True)

            return session
        return dependency

    async def middleware(self, request: Request, call_next):
        """FastAPI middleware that adds session to request state"""
        # Skip processing for excluded paths
        excluded_paths = ["/health", "/monitoring", "/docs", "/openapi.json", "/favicon.ico"]

        if any(request.url.path.startswith(path) for path in excluded_paths):
            return await call_next(request)

        try:
            # Process request through gateway (no requirements at middleware level)
            session = await self.process_request(request, [])

            # Add session to request state
            request.state.user_session = session
            request.state.user_type = session.user_type.value

            # Process request
            response = await call_next(request)

            # Add useful headers
            response.headers["X-User-Type"] = session.user_type.value
            response.headers["X-Session-ID"] = session.session_id

            return response

        except HTTPException as e:
            # Return proper error response
            headers = getattr(e, 'headers', {})
            return JSONResponse(
                status_code=e.status_code,
                content={"detail": e.detail},
                headers=headers
            )

    async def get_system_status(self) -> dict:
        """Get system status"""
        return await self.gateway.get_system_status()


# Convenience functions for dependency injection
def create_session_dependency(gateway_adapter: FastAPIGatewayAdapter):
    """Create a dependency that just gets the session"""
    return gateway_adapter.get_session()


def create_auth_dependency(gateway_adapter: FastAPIGatewayAdapter, *requirements: RequirementBase):
    """Create a dependency with specific auth requirements"""
    return gateway_adapter.require(*requirements)


# Example usage helpers
def get_user_session_from_state(request: Request) -> Optional[UserSession]:
    """Get user session from request state (if middleware was used)"""
    return getattr(request.state, 'user_session', None)


def get_user_type_from_state(request: Request) -> Optional[UserType]:
    """Get user type from request state (if middleware was used)"""
    session = get_user_session_from_state(request)
    return session.user_type if session else None

import uuid
from typing import Optional, Callable
from fastapi import Request, Depends

from kdcube_ai_app.infra.accounting import AccountingSystem, SystemResource, with_accounting
from kdcube_ai_app.auth.sessions import UserSession

class AccountingContextBinder:
    """
    Centralizes:
      - init_storage once
      - per-request context (FastAPI dependency)
      - per-socket connect snapshot
      - per-socket event rebind + optional accounting scope
    """

    def __init__(
            self,
            gateway_adapter: FastAPIGatewayAdapter,
            storage_backend,
            get_tenant_fn: Callable[[], str],
            accounting_enabled: bool = True,
            default_component: str = "chat-rest",
    ):
        self.gateway_adapter = gateway_adapter
        self.storage_backend = storage_backend
        self.get_tenant = get_tenant_fn
        self.enabled = accounting_enabled
        self.default_component = default_component

        # Initialize storage once (safe to call multiple times)
        AccountingSystem.init_storage(self.storage_backend, enabled=self.enabled)

    # -------- FastAPI dependency (HTTP) --------
    def http_dependency(self, component: Optional[str] = None):
        """
        Use in FastAPI endpoints:
            session: UserSession = Depends(binder.http_dependency("chat-rest"))
        """
        base_dep = self.gateway_adapter.get_session()
        comp = component or self.default_component

        async def dep(request: Request, session: UserSession = Depends(base_dep)) -> UserSession:
            AccountingSystem.set_context(
                user_id=getattr(session, "user_id", None),
                session_id=getattr(session, "session_id", None),
                project_id=request.path_params.get("project"),
                tenant_id=self.get_tenant(),
                request_id=request.headers.get("X-Request-ID", str(uuid.uuid4())),
                component=comp,
            )
            # optional convenience for code that expects request.state.user
            request.state.user = {
                "user_id": getattr(session, "user_id", None),
                "roles": getattr(session, "roles", []),
            }
            return session

        return dep

    # -------- Socket.IO: save minimal snapshot at connect --------
    async def socket_on_connect_save(self, sio, sid: str, session: UserSession, *, component: Optional[str] = None):
        """
        Call in your connect handler after you have UserSession.
        Stores enough info to re-apply context on each event.
        """
        acc = {
            "user_id": getattr(session, "user_id", None),
            "session_id": getattr(session, "session_id", None),
            "tenant_id": getattr(session, "tenant_id", None) or self.get_tenant(),
            "project_id": getattr(session, "project_id", None),  # or inject from handshake if you have one
            "component": component or self.default_component,
        }
        sock_state = await sio.get_session(sid) or {}
        sock_state["accounting_ctx"] = acc
        # keep prior items (e.g., 'user_session', 'authenticated')
        await sio.save_session(sid, sock_state)

    # -------- Socket.IO: reapply context per event (one-liner) --------
    async def socket_apply_event_context(self, sio, sid: str, *, component: Optional[str] = None, extra: Dict[str, Any] = None):
        """
        Call this at the top of each socket event handler (one line).
        """
        sess = await sio.get_session(sid) or {}
        acc = dict(sess.get("accounting_ctx") or {})
        if component:
            acc["component"] = component
        AccountingSystem.set_context(
            **acc,
            request_id=str(uuid.uuid4()),  # new request id per event
            **(extra or {})
        )

    # -------- Socket.IO: optional scoped helper around your work --------
    @asynccontextmanager
    async def socket_event_scope(
            self,
            sio,
            sid: str,
            *,
            component: Optional[str] = None,
            seeds: Optional[List[SystemResource]] = None,
            metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Use as:
            async with binder.socket_event_scope(self.sio, sid, component="chat.socket.workflow",
                                                seeds=[conv_seed], metadata={"task_id": task_id}):
                ... do the work ...
        """
        await self.socket_apply_event_context(sio, sid, component=component)
        async with with_accounting(
                component or (await sio.get_session(sid) or {}).get("accounting_ctx", {}).get("component", self.default_component),
                seed_system_resources=seeds or [],
                metadata=metadata or {}
        ):
            yield

    # Convenience to make a conversation seed
    @staticmethod
    def conversation_seed(session: UserSession, sid: str) -> SystemResource:
        return SystemResource(
            resource_type="conversation",
            resource_id=session.session_id or "unknown",
            rn=f"conv:{session.session_id or 'unknown'}",
            metadata={"socket_id": sid}
        )