# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# libs/kdcube-comm/streamlit/websocket/dedicated_ws_connection.py

"""
Dedicated (per-session) WebSocket connector for Streamlit.

- One socket per Streamlit session/tab.
- Runs a background thread with websocket-client (sync) and an auto-reconnect thread.
- On incoming messages, triggers a rerun ONLY for the current session via a per-session notifier,
  setting:
      st.session_state["last_message"] = <payload>
      st.session_state["new_event"] = True
  Your UI can then append it into an inbox on the next rerun.

Public API:
    Connection(address: str, on_message_cb: Callable[[dict], None], on_error_cb: Optional[Callable]=None)
        .connect() -> Connection
        .send_message(message: dict) -> None
        .send_message_async(message: dict) -> Awaitable[None]

Requires:
    streamlit>=1.30
    websocket-client>=1.8
"""

from __future__ import annotations
import json, functools, time, asyncio, threading
import websocket
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

from .notifier import notify


def _on_error(ws, error):  # pragma: no cover
    print(error)


def _on_close(ws, code, msg):  # pragma: no cover
    print(f"WS closed: {code} {msg}")
    st.session_state['websocket'] = None
    st.session_state['reconnect_needed'] = True


def _on_message(ws, message, conn: "Connection"):
    try:
        data = json.loads(message)
    except Exception:
        data = {"raw": message}
    # Ask THIS session to rerun and fold the message into its state
    if getattr(conn, "notify_fn", None):
        conn.notify_fn({"last_message": data, "new_event": True})
    elif conn.on_message_cb:
        # fallback (visible on next manual rerun)
        conn.on_message_cb(data)


def _run_ws(conn: "Connection"):
    ws = websocket.WebSocketApp(
        conn.address,
        on_message=functools.partial(_on_message, conn=conn),
        on_error=conn.on_error_cb,
        on_close=_on_close,
    )
    st.session_state['websocket'] = ws
    ws.run_forever(ping_interval=5)


def _reconnect(conn: "Connection"):
    while True:
        if 'session_stop_event' in st.session_state and st.session_state.session_stop_event.is_set():
            break
        if st.session_state.get('reconnect_needed', False):
            try:
                _run_ws(conn)
                st.session_state['reconnect_needed'] = False
            except Exception:
                time.sleep(5)
        time.sleep(2)


class Connection:
    """Per-session WebSocket connection."""

    def __init__(self, address: str, on_message_cb, on_error_cb=None):
        self.address = address
        self.on_message_cb = on_message_cb
        self.on_error_cb = on_error_cb or _on_error
        self.notify_fn = None  # set in connect()

    def connect(self) -> "Connection":
        if 'session_stop_event' not in st.session_state:
            st.session_state.session_stop_event = threading.Event()

        # Watcher to detect end-of-session and close WS cleanly
        if 'watch' not in st.session_state:
            def _watch():
                from streamlit.runtime import get_instance
                runtime = get_instance()
                ctx = get_script_run_ctx()
                while runtime.is_active_session(ctx.session_id):
                    time.sleep(20)
                st.session_state.session_stop_event.set()
                ws = st.session_state.get("websocket")
                try:
                    if ws and ws.sock and ws.sock.connected:
                        ws.close()
                except Exception:
                    pass
            st.session_state.watch = threading.Thread(target=_watch, daemon=True)
            add_script_run_ctx(st.session_state.watch)
            st.session_state.watch.start()

        # Socket thread
        if 'websocket_thread' not in st.session_state:
            st.session_state.websocket_thread = threading.Thread(target=_run_ws, args=(self,), daemon=True)
            add_script_run_ctx(st.session_state.websocket_thread)
            st.session_state.websocket_thread.start()

            # Bind per-session notifier using current session ctx + thread
            ctx = get_script_run_ctx()
            st.session_state["ctx"] = ctx
            self.notify_fn = functools.partial(
                notify,
                session_id=ctx.session_id,
                thread=st.session_state.websocket_thread,
            )

        # Reconnect thread
        if 'reconnect_thread' not in st.session_state:
            st.session_state.reconnect_thread = threading.Thread(target=_reconnect, args=(self,), daemon=True)
            add_script_run_ctx(st.session_state.reconnect_thread)
            st.session_state.reconnect_thread.start()
        return self

    def send_message(self, message: dict) -> None:
        ws = st.session_state.get('websocket')
        if ws and ws.sock and ws.sock.connected:
            ws.send(json.dumps(message))
        else:
            print("NO CONNECTION")

    async def send_message_async(self, message: dict) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.send_message, message)