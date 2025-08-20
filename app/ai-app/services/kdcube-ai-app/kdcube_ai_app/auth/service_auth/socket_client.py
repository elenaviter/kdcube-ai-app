# persistent_kb_socket_client.py
import asyncio
import os
import time
import uuid
from typing import Optional, Dict, Any

import socketio

from kdcube_ai_app.auth.service_auth.base import IdpConfig, TokenBundle
from kdcube_ai_app.auth.service_auth.factory import create_service_idp

DEFAULT_REFRESH_MARGIN_SEC = 120  # reconnect before token expires

class PersistentKBServiceSocketClient:
    def __init__(
        self,
        *,
        kb_socket_url: str,
        idp_cfg: IdpConfig,
        project: Optional[str] = None,
        tenant: Optional[str] = None,
        refresh_margin_sec: int = DEFAULT_REFRESH_MARGIN_SEC,
        reconnection: bool = True,
        reconnection_attempts: int = 0,   # 0 = forever
        reconnection_delay: float = 2.0,
    ):
        self.kb_socket_url = kb_socket_url.rstrip("/")
        self.idp_cfg = idp_cfg
        self.project = project
        self.tenant = tenant
        self.refresh_margin_sec = refresh_margin_sec

        # Token & IdP
        self._idp = create_service_idp(self.idp_cfg)
        self._tokens: Optional[TokenBundle] = None
        self.socketio_path = "/socket.io"

        # Socket
        self.sio = socketio.AsyncClient(
            reconnection=reconnection,
            reconnection_attempts=reconnection_attempts,
            reconnection_delay=reconnection_delay,
        )

        # Pending requests: request_id -> Future
        self._pending: Dict[str, asyncio.Future] = {}

        # Background tasks
        self._tasks: list[asyncio.Task] = []
        self._connected_once = asyncio.Event()

        # Wire events
        self._register_handlers()

    # ---------- Public API ----------

    async def start(self):
        """Authenticate, connect, and start background token-refresh."""
        await self._ensure_tokens()
        await self._connect()
        # Start background token refresh/reconnect loop
        self._tasks.append(asyncio.create_task(self._refresh_loop(), name="kb-refresh-loop"))
        await self._connected_once.wait()

    async def stop(self):
        """Stop background tasks and disconnect."""
        for t in self._tasks:
            t.cancel()
        self._tasks.clear()
        if self.sio.connected:
            await self.sio.disconnect()
        try:
            self._idp.close()
        except Exception:
            pass

    async def submit_kb_search(
        self,
        *,
        query: str,
        on_behalf_session_id: str,
        top_k: int = 5,
        resource_id: Optional[str] = None,
        timeout_sec: float = 15.0,
        request_id: Optional[str] = None,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Send a kb_search command and wait for the correlated kb_search_result by request_id.
        """
        if not self.sio.connected:
            # Attempt reconnect with fresh tokens
            await self._reconnect_with_fresh_tokens()

        req_id = request_id or uuid.uuid4().hex
        fut = asyncio.get_running_loop().create_future()
        self._pending[req_id] = fut

        payload = {
            "request_id": req_id,
            "query": query,
            "top_k": top_k,
            "resource_id": resource_id,
            "on_behalf_session_id": on_behalf_session_id,
            "project": self.project,
            "tenant": self.tenant,
        }
        if extra_payload:
            payload.update(extra_payload)

        await self.sio.emit("kb_search", payload)

        try:
            result = await asyncio.wait_for(fut, timeout=timeout_sec)
            return result
        finally:
            self._pending.pop(req_id, None)

    # ---------- Internals ----------

    def _register_handlers(self):
        @self.sio.event
        async def connect():
            self._connected_once.set()
            print("[kb-client] connected")

        @self.sio.event
        async def disconnect():
            print("[kb-client] disconnected")

        @self.sio.on("session_info")
        async def session_info(data):
            print("[kb-client] session_info:", data)

        @self.sio.on("socket_error")
        async def socket_error(data):
            print("[kb-client] socket_error:", data)

        @self.sio.on("kb_search_result")
        async def kb_search_result(data):
            # Correlate by request_id, if present
            req_id = (data or {}).get("request_id")
            if req_id and req_id in self._pending:
                fut = self._pending.get(req_id)
                if fut and not fut.done():
                    fut.set_result(data)
            else:
                # No correlation id -> just print
                print("[kb-client] kb_search_result (uncorrelated):", data)

    async def _ensure_tokens(self):
        if self._tokens is None:
            self._tokens = await asyncio.to_thread(self._idp.authenticate)
        else:
            # Ensure fields are populated for timing; also handle already-expired
            if getattr(self._tokens, "access_expires_at", None) is None:
                self._tokens.ensure_exp_fields()
            if self._is_access_expiring_soon():
                self._tokens = await asyncio.to_thread(self._idp.refresh, self._tokens)

    def _is_access_expiring_soon(self) -> bool:
        if not self._tokens or getattr(self._tokens, "access_expires_at", None) is None:
            return True
        return (self._tokens.access_expires_at - time.time()) < self.refresh_margin_sec

    async def _connect(self):
        await self.sio.connect(
            self.kb_socket_url,
            transports=["websocket"],
            socketio_path=self.socketio_path,
            auth={
                "bearer_token": self._tokens.access_token,
                "id_token": self._tokens.id_token,
                "project": self.project,
                "tenant": self.tenant,
            },
            wait=True,
            namespaces=["/"],
        )

    async def _reconnect_with_fresh_tokens(self):
        await self._ensure_tokens()
        if self.sio.connected:
            try:
                await self.sio.disconnect()
            except Exception:
                pass
        await self._connect()

    async def _refresh_loop(self):
        # Periodically check token expiry and reconnect with fresh tokens
        try:
            while True:
                await asyncio.sleep(30)
                if self._is_access_expiring_soon():
                    print("[kb-client] refreshing tokens & reconnecting before expiry")
                    await self._reconnect_with_fresh_tokens()
        except asyncio.CancelledError:
            return


# ---------------------------
# Runnable example entrypoint
# ---------------------------
async def _demo():

    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())

    # Build IdP config from env (Cognito example; swap if you add other providers)
    from botocore.config import Config as BotoConfig
    idp_cfg = IdpConfig(
        "cognito",
        region=os.getenv("COGNITO_REGION"),
        user_pool_id=os.getenv("COGNITO_USER_POOL_ID"),
        client_id=os.getenv("COGNITO_SERVICE_CLIENT_ID"),
        client_secret=os.getenv("COGNITO_SERVICE_CLIENT_SECRET") or None,
        username=os.getenv("OIDC_SERVICE_ADMIN_USERNAME"),
        password=os.getenv("OIDC_SERVICE_ADMIN_PASSWORD"),
        use_admin_api=True,
        boto_cfg=BotoConfig(retries={"max_attempts": 3, "mode": "standard"}),
    )

    client = PersistentKBServiceSocketClient(
        kb_socket_url=os.getenv("KB_SOCKET_URL", "http://localhost:8000/socket.io"),
        idp_cfg=idp_cfg,
        project=os.getenv("DEFAULT_PROJECT_NAME"),
        tenant=os.getenv("TENANT_ID"),
    )
    await client.start()

    # 1) Direct user test: paste browser tokens if you want to test this path
    USER_ACCESS_TOKEN = os.getenv("TEST_USER_ACCESS_TOKEN", "eyJraWQiOiI3eVdRQkZFVFo5VUlkNE5tZm5jVzJTUExcLzE5MHF3c0dcL3U5UzF1WmV1cTg9IiwiYWxnIjoiUlMyNTYifQ.eyJzdWIiOiI5MjQ1MzRiNC1kMGYxLTcwZTUtNDVkYy1mOTI1NTg0ZjNkNWMiLCJjb2duaXRvOmdyb3VwcyI6WyJrZGN1YmU6cm9sZTpzdXBlci1hZG1pbiJdLCJpc3MiOiJodHRwczpcL1wvY29nbml0by1pZHAuZXUtd2VzdC0xLmFtYXpvbmF3cy5jb21cL2V1LXdlc3QtMV9SVWx3WlNxNkgiLCJ2ZXJzaW9uIjoyLCJjbGllbnRfaWQiOiJtNTBpcGEyM3JsNjRzZGVoYTZicThxdTI0Iiwib3JpZ2luX2p0aSI6ImQwOTE1YWNlLTk0MzQtNDU2Yi1hZjIzLWIzNjQ2YjRjYzBlMiIsImV2ZW50X2lkIjoiNmUwMzU3MzYtZjkzNC00ZGMwLThlNTUtNTBiMzY0MmY2YjcwIiwidG9rZW5fdXNlIjoiYWNjZXNzIiwic2NvcGUiOiJwaG9uZSBvcGVuaWQgcHJvZmlsZSBlbWFpbCIsImF1dGhfdGltZSI6MTc1NTczMDI2MCwiZXhwIjoxNzU1NzMzODYwLCJpYXQiOjE3NTU3MzAyNjAsImp0aSI6IjZkNDAzNGFhLTMxNDctNDk0MS05ZWQ3LWQxNWQ1NGI0MGE3NSIsInVzZXJuYW1lIjoiZWxlbmEudiJ9.oTXQTimXtvTdvp37gYHCUVTeHAqoTQeWGecOSP1g6savBhMYi-_JoT_zQni5rI9hQHnVG1tomwzUUF0rnnHQTpuwLaLRnTJNBUArbeLE7ByJZL7of4nPJc4HB9A7QExbDaRcQfcwnkPdeLV5V9OnGLDodh-Yr6pNfe9prHTcxHs8rk3l2Nc5TTUr43BJGrzIbbN7MgbVMCsLLF0qk2XTT0RhYRIazO5Ysc-DuwdLZix1g0oFnX8dYLaouyGOBFpaXqFAN8ZVeIcVLbpHpC0vLXvKsPTgPtosexbWvch_1EqmXcXhiIU6MGUjoFjb8fCedZcZNtp9Zt59a5-PSnhIhQ")
    USER_ID_TOKEN = os.getenv("TEST_USER_ID_TOKEN", "eyJraWQiOiJtYWJ0SHNFK0FDY1N1RDN3MzlCR21aSHc5dDA4M0UwdEwzd0pQYU9aRUtRPSIsImFsZyI6IlJTMjU2In0.eyJhdF9oYXNoIjoieWJrUDNrZ0gxMFpVcEZVRTN4Um83dyIsInN1YiI6IjkyNDUzNGI0LWQwZjEtNzBlNS00NWRjLWY5MjU1ODRmM2Q1YyIsImNvZ25pdG86Z3JvdXBzIjpbImtkY3ViZTpyb2xlOnN1cGVyLWFkbWluIl0sImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJpc3MiOiJodHRwczpcL1wvY29nbml0by1pZHAuZXUtd2VzdC0xLmFtYXpvbmF3cy5jb21cL2V1LXdlc3QtMV9SVWx3WlNxNkgiLCJjb2duaXRvOnVzZXJuYW1lIjoiZWxlbmEudiIsIm9yaWdpbl9qdGkiOiJkMDkxNWFjZS05NDM0LTQ1NmItYWYyMy1iMzY0NmI0Y2MwZTIiLCJhdWQiOiJtNTBpcGEyM3JsNjRzZGVoYTZicThxdTI0IiwiZXZlbnRfaWQiOiI2ZTAzNTczNi1mOTM0LTRkYzAtOGU1NS01MGIzNjQyZjZiNzAiLCJ0b2tlbl91c2UiOiJpZCIsImF1dGhfdGltZSI6MTc1NTczMDI2MCwiZXhwIjoxNzU1NzMzODYwLCJpYXQiOjE3NTU3MzAyNjAsImp0aSI6IjljMTg1ZmRiLWNmZmMtNDIyYi1iYWRlLTZlOGU4M2JmMmRmYiIsImVtYWlsIjoibGVuYUBuZXN0bG9naWMuY29tIn0.NdpPErbQpLeRdcKVFEG32ryJYdRmlzF7yhRVcmesGLQ8qnUZh_g0FqIW0Ky-lvy7aH8vDmFnq2rN2Hf-VTAUXxSrBuGtXWhM1v_Cb84qN71vFpB5pjFohjIivB6hOUWSSJ1ouylIf5NUJx6cfJQOSFADgwKogWt6IHpeEsorcuWDrKmVR90jEGus3SOvd69wAuRCbPj82e1ZfKp_4V6yG-Y_SIpERwEOKKz_k3_xAuLia9WmZmzbryfhYt4rLmqPToh81RWnPDwve9KxHTNHaadRMUBqSWwPNPqcT3p5XTBqrdgQhnWrdYAmVoLvjCFawcaTcSlx5tmOCtkCZ9Lwug")

    # 2) On-behalf test: pass a valid session id (from your chat gateway)
    default_on_behalf_session_id = "1575eaf7-ca97-4f7e-a6e3-fca107400a90"
    SESSION_ID = os.getenv("TEST_ON_BEHALF_SESSION_ID", default_on_behalf_session_id)

    project = os.environ.get("DEFAULT_PROJECT_NAME")
    tenant = os.environ.get("TENANT_ID")

    # Example: submit 2 searches on behalf of different users
    on_behalf_1 = os.getenv("TEST_ON_BEHALF_SESSION_ID_1", default_on_behalf_session_id)
    on_behalf_2 = os.getenv("TEST_ON_BEHALF_SESSION_ID_2")

    if on_behalf_1:
        r1 = await client.submit_kb_search(
            query="zero trust architecture controls",
            on_behalf_session_id=on_behalf_1,
            top_k=5,
            timeout_sec=20,
        )
        print("Result for user1:", r1)

    # if on_behalf_2:
    #     r2 = await client.submit_kb_search(
    #         query="identity governance best practices",
    #         on_behalf_session_id=on_behalf_2,
    #         top_k=5,
    #         timeout_sec=20,
    #     )
    #     print("Result for user2:", r2)

    # Keep running forever (daemon)
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(_demo())
