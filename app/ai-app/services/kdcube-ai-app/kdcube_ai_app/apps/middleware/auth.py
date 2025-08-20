# /middleware/auth.py
"""
Framework-specific adapters for the auth system
"""

# ================================
# FastAPI Adapter
# ================================
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.requests import Request
from starlette.status import HTTP_400_BAD_REQUEST

from kdcube_ai_app.auth.AuthManager import AuthManager, RequirementBase, AuthenticationError, AuthorizationError, \
    HTTP_401_UNAUTHORIZED, PRIVILEGED_ROLES
from kdcube_ai_app.auth.sessions import RequestContext, UserType
from kdcube_ai_app.infra.namespaces import CONFIG


class UserSessionError(HTTPException):
    def __init__(self) -> None:
        super().__init__(status_code=HTTP_400_BAD_REQUEST, detail="No user session id provided")


class UserSessionID:
    def __init__(self, header_name: str = "User-Session-ID", auto_error: bool = False):
        self.header_name = header_name
        self.auto_error = auto_error

    def __call__(self, request: Request):
        user_session_id = request.headers.get(self.header_name)
        if self.auto_error and (user_session_id is None or user_session_id.strip() == ""):
            raise UserSessionError()
        return user_session_id


class FastAPIAuthAdapter:
    """Adapter for FastAPI framework"""

    def __init__(self,
                 auth_manager: AuthManager,
                 session_manager: 'SessionManager',
                 service_role_name: str):
        self.auth_manager = auth_manager
        self.security = HTTPBearer(auto_error=False)
        self.service_role_name = service_role_name
        self.session_manager = session_manager

    def _extract_context(self, request: Request) -> RequestContext:
        """Extract request context from FastAPI request"""
        return RequestContext(
            client_ip=request.client.host if request.client else "unknown",
            user_agent=request.headers.get("user-agent", ""),
            authorization_header=request.headers.get("authorization"),
            id_token=request.headers.get(CONFIG.ID_TOKEN_HEADER_NAME),
        )

    def require(self, *requirements: RequirementBase, require_all: bool = True):
        async def dependency(
                credentials: HTTPAuthorizationCredentials = Depends(self.security),
                user_session_id: str = Depends(UserSessionID(auto_error=False)),
                request: Request = None,
        ):
            if not credentials:
                raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="No authentication credentials provided")

            id_token = request.headers.get(CONFIG.ID_TOKEN_HEADER_NAME)

            try:
                user = await self.auth_manager.authenticate_and_authorize_with_both(
                    credentials.credentials,
                    id_token,
                    *requirements,
                    require_all=require_all
                )
                # service “on behalf of” branch stays the same...
                if self.service_role_name in (user.roles or []):
                    if user_session_id is None or user_session_id.strip() == "":
                        raise UserSessionError()
                    session = await self.session_manager.get_session_by_id(user_session_id)
                    if session:
                        return session.to_user()
                    raise UserSessionError()
                return user
            except AuthenticationError as e:
                raise HTTPException(status_code=e.code, detail=e.message)
            except AuthorizationError as e:
                raise HTTPException(status_code=e.code, detail=e.message)
        return dependency

    def require_session(self, *requirements: RequirementBase, require_all: bool = True):
        async def dependency(
                credentials: HTTPAuthorizationCredentials = Depends(self.security),
                user_session_id: str = Depends(UserSessionID(auto_error=False)),
                request: Request = None,
        ):
            if not credentials:
                raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="No authentication credentials provided")

            id_token = request.headers.get(CONFIG.ID_TOKEN_HEADER_NAME)

            try:
                user = await self.auth_manager.authenticate_and_authorize_with_both(
                    credentials.credentials,
                    id_token,
                    *requirements,
                    require_all=require_all
                )
                user_type = UserType.PRIVILEGED if PRIVILEGED_ROLES & set(user.roles or []) else UserType.REGISTERED
                user_data = {
                    "user_id": getattr(user, 'sub', None) or user.username,
                    "username": user.username,
                    "email": user.email,
                    "roles": user.roles or [],
                    "permissions": user.permissions or []
                }

                if self.service_role_name in (user.roles or []):
                    if user_session_id is None or user_session_id.strip() == "":
                        raise UserSessionError()
                    session = await self.session_manager.get_session_by_id(user_session_id)
                    if session:
                        return session
                    raise UserSessionError()
                else:
                    request_context = self._extract_context(request)
                    session = await self.session_manager.get_or_create_session(request_context, user_type, user_data)
                return session
            except AuthenticationError as e:
                raise HTTPException(status_code=e.code, detail=e.message)
            except AuthorizationError as e:
                raise HTTPException(status_code=e.code, detail=e.message)
        return dependency
