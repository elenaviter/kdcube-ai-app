# kdcube_ai_app/apps/knowledge_base/api/socketio/kb.py
"""
Modular Socket.IO handler for KB progress events
- Creates a Socket.IO server with Redis manager
- Listens to orchestrator pubsub channel and relays events to clients
- Clean start/stop lifecycle
"""
import os
import asyncio
import logging

import socketio

from typing import Optional

from kdcube_ai_app.apps.middleware.accounting import MiddlewareAuthWithAccounting
from kdcube_ai_app.auth.sessions import UserSession
from kdcube_ai_app.auth.AuthManager import RequireRoles, RequirePermissions, AuthenticationError
from kdcube_ai_app.infra.orchestration.app.communicator import ServiceCommunicator

logger = logging.getLogger("KB.SocketIO")

class SocketIOKBHandler:
    def __init__(
            self,
            allowed_origins,
            redis_url: str,
            orchestrator_identity: str,
                    instance_id: Optional[str] = None,
        *,
        auth_with_acct: MiddlewareAuthWithAccounting,  # << single dependency
        component_name: str = "kb-socket",
    ):
        self.allowed_origins = allowed_origins
        self.redis_url = redis_url
        self.orchestrator_identity = orchestrator_identity
        self.instance_id = instance_id
        self.auth_with_acct = auth_with_acct
        self.component_name = component_name

        # Socket.IO
        self.sio = self._create_socketio_server()
        self._setup_event_handlers()

        # PubSub
        self._redis = None
        self._listener_task: asyncio.Task | None = None

        self.comm = ServiceCommunicator(redis_url, orchestrator_identity)
        self.relay_channel = "kb.process_resource_out"

    # ----- Socket.IO -----

    def _create_socketio_server(self):
        mgr = socketio.AsyncRedisManager(self.redis_url)
        return socketio.AsyncServer(
            cors_allowed_origins=self.allowed_origins,
            async_mode="asgi",
            client_manager=mgr,
            logger=False,
            engineio_logger=False,
        )

    def _setup_event_handlers(self):
        if not self.sio:
            return

        @self.sio.on("connect")
        async def handle_connect(sid, environ, auth):
            return await self._handle_connect(sid, environ, auth)

        @self.sio.on("disconnect")
        async def handle_disconnect(sid):
            logger.info(f"KB socket disconnected: {sid}")

        # service can attach a default on-behalf session after connecting
        @self.sio.on("attach_session")
        async def attach_session(sid, data):
            try:
                socket_state = await self.sio.get_session(sid)
                if not socket_state or not socket_state.get("authenticated"):
                    return False

                on_behalf_id = (data or {}).get("user_session_id")
                if not on_behalf_id:
                    await self.sio.emit("socket_error", {"error": "user_session_id required"}, to=sid)
                    return False

                session = await self.auth_with_acct.get_session_by_id(on_behalf_id)
                if not session:
                    await self.sio.emit("socket_error", {"error": "Unknown user_session_id"}, to=sid)
                    return False

                socket_state["default_on_behalf_session_id"] = on_behalf_id
                await self.sio.save_session(sid, socket_state)

                # join room for this session
                await self.sio.enter_room(sid, session.session_id)

                await self.sio.emit("session_attached", {
                    "user_session_id": session.session_id,
                    "user_type": session.user_type.value,
                    "username": session.username
                }, to=sid)
                return True
            except Exception as e:
                logger.exception("attach_session failed")
                await self.sio.emit("socket_error", {"error": str(e)}, to=sid)
                return False

        # example work event — resolve on-behalf session per event
        @self.sio.on("kb_process_resource")
        async def kb_process_resource(sid, payload):
            session, project, tenant = await self._resolve_event_context(sid, payload)
            # bind accounting for this event
            self.auth_with_acct.apply_event_accounting(
                session=session,
                component=f"{self.component_name}.event",
                tenant_id=tenant,
                project_id=project,
                extra={"socket_event": "kb_process_resource"}
            )
            # ... enqueue work; include both target_sid and session_id for relay ...
            # await self.comm.publish(...)

        @self.sio.on("kb_search")
        async def kb_search(sid, payload):
            session, project, tenant = await self._resolve_event_context(sid, payload)
            self.auth_with_acct.apply_event_accounting(
                session=session,
                component=f"{self.component_name}.event",
                tenant_id=tenant,
                project_id=project,
                extra={"socket_event": "kb_search"}
            )
            # ... perform search or dispatch job ...

    async def _handle_connect(self, sid, environ, auth):
        origin = environ.get("HTTP_ORIGIN")
        logger.info(f"KB socket connect from {origin} (sid={sid})")

        # internal bypass stays
        if auth and auth.get("token") == os.getenv("INTERNAL_SERVICE_TOKEN"):
            await self.sio.save_session(sid, {
                "authenticated": True,
                "internal": True,
                "project": (auth or {}).get("project"),
                "tenant": (auth or {}).get("tenant"),
                "user_session": None,
                "service": True,
            })
            return True

        if not auth or not auth.get("bearer_token"):
            logger.warning("KB socket connect rejected: missing bearer token")
            return False

        try:
            # allow connect without user_session_id (service mode)
            session = await self.auth_with_acct.process_socket_connect(
                auth, environ,
                RequireRoles("kdcube:role:super-admin", require_all=False),
                RequirePermissions("kdcube:*:knowledge_base:*;read", require_all=True),
                require_all=False,
                component=self.component_name,
                require_existing_session=False,
                verify_token_session_match=False,
            )
        except AuthenticationError as e:
            logger.warning(f"KB socket auth error: {e}")
            return False
        except Exception as e:
            logger.error(f"KB socket connect error: {e}")
            return False

        # store minimal state
        is_service = bool(session is None)  # if session is None we assume service-only connect
        sock_state = {
            "authenticated": True,
            "internal": False,
            "project": (auth or {}).get("project"),
            "tenant": (auth or {}).get("tenant"),
            "user_session": (session.__dict__ if session else None),
            "service": is_service,
        }
        await self.sio.save_session(sid, sock_state)

        # UI case: join session room immediately
        if session:
            await self.sio.enter_room(sid, session.session_id)

        await self.sio.emit("session_info", {
            "connected_as_service": is_service,
            "session_id": getattr(session, "session_id", None),
            "user_type": getattr(session, "user_type", None).value if session else None,
            "username": getattr(session, "username", None),
            "project": sock_state.get("project"),
            "tenant": sock_state.get("tenant"),
        }, to=sid)

        logger.info(f"KB socket connected sid={sid}, service={is_service}, session={getattr(session,'session_id',None)}")
        return True

    async def _resolve_event_context(self, sid: str, payload: dict):
        """
        Decide which session this event should spend against:
        1) payload['on_behalf_session_id'] (highest priority)
        2) socket_state['default_on_behalf_session_id'] (set by 'attach_session')
        3) socket_state['user_session'] (UI user)
        Return (session or None, project, tenant)
        """
        sock = await self.sio.get_session(sid) or {}
        project = (payload or {}).get("project") or sock.get("project")
        tenant = (payload or {}).get("tenant") or sock.get("tenant")

        target_id = (payload or {}).get("on_behalf_session_id") or sock.get("default_on_behalf_session_id")
        session = None
        if target_id:
            session = await self.auth_with_acct.get_session_by_id(target_id)
            if session:
                # ensure we’re in that room for downstream emits
                await self.sio.enter_room(sid, session.session_id)
        elif sock.get("user_session"):
            # UI user path
            from kdcube_ai_app.auth.sessions import UserSession as US
            session = US(**sock["user_session"])

        return session, project, tenant

    async def _on_pubsub_message(self, message: dict):
        """
        Worker can target by sid and/or by session room.
        """
        try:
            target_sid = message.get("target_sid")
            session_id = message.get("session_id")                 # allow room targeting
            event = message["event"]
            data = message["data"]
            resource_id = data["resource_id"]
            channel = f"resource_processing_progress:{resource_id}"
            unified = {"event": event, **data}
            logger.info(f"KB socket pubsub: {channel}, to sid={target_sid}, session={session_id}")

            if target_sid:
                await self.sio.emit(channel, unified, room=target_sid)
            if session_id:
                await self.sio.emit(channel, unified, room=session_id)
        except Exception as e:
            logger.error(f"Relay error: {e}")

    def get_asgi_app(self):
        return socketio.ASGIApp(self.sio) if self.sio else None


    async def _save_socket_session(self, sid, session: Optional[UserSession], internal: bool = False):
        payload = {
            "authenticated": True,
            "internal": internal,
            "user_session": (session.__dict__ if session else None),
        }
        await self.sio.save_session(sid, payload)

    # ----- Lifecycle -----
    async def start(self):
        if getattr(self, "_listener_started", False):
            return
        # Subscribe to orchestrator channel (identity prefix is added inside)
        await self.comm.subscribe(self.relay_channel)
        # Start background listener
        await self.comm.start_listener(self._on_pubsub_message)
        self._listener_started = True
        logger.info("Socket.IO KB handler subscribed & listening.")

    async def stop(self):
        await self.comm.stop_listener()
        self._listener_started = False
        logger.info("Socket.IO KB handler stopped.")
