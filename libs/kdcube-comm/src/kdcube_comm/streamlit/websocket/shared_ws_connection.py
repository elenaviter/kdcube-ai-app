# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# libs/kdcube-comm/streamlit/websocket/shared_ws_connection.py

"""
Shared (process-wide) WebSocket connector for Streamlit.

- One socket per Streamlit process (cached via @st.cache_resource).
- On incoming messages, broadcasts a small state update to ALL sessions:
      {"last_message": <payload>, "new_event": True}
  Each session folds it into its own inbox on rerun.

Public API:
    SharedConnection.get(address: str) -> SharedConnection
        .send_message(message: dict) -> None
        .send_message_async(message: dict) -> Awaitable[None]
"""

from __future__ import annotations
import threading, json, functools
import websocket
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from streamlit.runtime import get_instance

from .notifier import notify_all

IDENTITY = "SHARED WS"


@st.cache_resource
def _cache():
    return {"conn": None}


def _on_error(ws, error):  # pragma: no cover
    print(error)


def _on_message(ws, message, conn: "SharedConnection"):
    try:
        data = json.loads(message)
    except Exception:
        data = {"raw": message}
    # Hint all sessions to append the message on their next rerun
    state = {"last_message": data, "new_event": True}
    conn.notify_all_fn(state)


def _on_close(ws, code, msg, conn: "SharedConnection"):  # pragma: no cover
    print(f"{IDENTITY}.on_close: {code} {msg}")
    conn.ws = None
    conn.reconnect_needed = True


def _run_ws(conn: "SharedConnection"):
    ws = websocket.WebSocketApp(
        conn.address,
        on_message=functools.partial(_on_message, conn=conn),
        on_error=conn.on_error_cb,
        on_close=functools.partial(_on_close, conn=conn),
    )
    conn.ws = ws
    ws.run_forever(ping_interval=5)
    conn.reconnect_needed = True


class SharedConnection:
    """Process-wide shared WebSocket connection."""

    def __init__(self, address: str, on_error_cb=None):
        self.address = address
        self.ws = None
        self.reconnect_needed = False
        self.ws_thread = None
        self.recon_thread = None
        self.on_error_cb = on_error_cb or _on_error
        self.notify_all_fn = None
        self.stop_event = None

    @staticmethod
    def get(address: str) -> "SharedConnection":
        ctx = get_script_run_ctx()
        c = _cache()
        if not c["conn"]:
            c["conn"] = SharedConnection(address)
            c["conn"].connect()
        # stash current session ctx so notifier can target sessions later
        st.session_state["ctx"] = ctx
        return c["conn"]

    def connect(self) -> None:
        if not self.ws_thread:
            runtime = get_instance()
            self.stop_event = runtime._get_async_objs().must_stop
            self.ws_thread = threading.Thread(target=_run_ws, args=(self,), daemon=True)
            add_script_run_ctx(self.ws_thread)
            self.ws_thread.start()
            self.notify_all_fn = functools.partial(notify_all, thread=self.ws_thread)

        if not self.recon_thread:
            def _reconnect():
                import time
                while not self.stop_event.is_set():
                    if self.reconnect_needed:
                        try:
                            _run_ws(self)
                            self.reconnect_needed = False
                        except Exception:
                            time.sleep(5)
                    time.sleep(2)

            self.recon_thread = threading.Thread(target=_reconnect, daemon=True)
            add_script_run_ctx(self.recon_thread)
            self.recon_thread.start()

    def send_message(self, message: dict) -> None:
        if self.ws and self.ws.sock and self.ws.sock.connected:
            self.ws.send(json.dumps(message))
        else:
            print(f"{IDENTITY}.send_message: no connection")

    async def send_message_async(self, message: dict) -> None:
        import asyncio
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.send_message, message)