# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# libs/kdcube-comm/streamlit/streamlit_ws_duplex/client/app.py

import os, json, streamlit as st
from dotenv import load_dotenv
load_dotenv()

ADDR = os.environ.get("WS_INTEGRATION_ADDRESS", "ws://localhost:8011/ws")

st.set_page_config(page_title="WS Duplex (Dedicated vs Shared)", layout="wide")
st.title("🔁 WebSocket Duplex Demo")

# state
if "inbox" not in st.session_state: st.session_state["inbox"] = []
if "new_event" not in st.session_state: st.session_state["new_event"] = False

mode = st.radio("Mode", ["Dedicated (per session)", "Shared (singleton)"], horizontal=True)

# connect
if mode.startswith("Shared"):
    from kdcube_comm.streamlit.websocket.shared_ws_connection import SharedConnection
    conn = SharedConnection.get(ADDR)
else:
    from kdcube_comm.streamlit.websocket.dedicated_ws_connection import Connection as Dedicated
    def _on_msg(data):
        # fallback if notifier didn’t fire
        st.session_state["last_message"] = data
        st.session_state["new_event"] = True
    conn = Dedicated(ADDR, _on_msg).connect()

# fold in newly delivered message (set by notifier) on each rerun
if st.session_state.get("new_event") and "last_message" in st.session_state:
    st.session_state["inbox"].append(st.session_state["last_message"])
    st.session_state["new_event"] = False

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Send")
    msg = st.text_area("Echo (JSON)", value='{"op": "ping", "note": "hello from client"}', height=120)
    if st.button("Send echo"):
        try: payload = json.loads(msg)
        except Exception: payload = {"raw": msg}
        conn.send_message(payload)

    st.divider()
    st.caption("Per-connection ticker (only this socket)")
    c1, c2 = st.columns(2)
    if c1.button("Pause ticks"):  conn.send_message({"op":"ticks","enable":False})
    if c2.button("Resume ticks"): conn.send_message({"op":"ticks","enable":True})

    st.divider()
    st.caption("Broadcast controls (affect ALL sockets)")
    b1, b2 = st.columns(2)
    if b1.button("Pause broadcast ticks"):  conn.send_message({"op":"broadcast_ticks","enable":False})
    if b2.button("Resume broadcast ticks"): conn.send_message({"op":"broadcast_ticks","enable":True})

    interval = st.number_input("Broadcast interval (sec)", min_value=0.1, value=5.0, step=0.1)
    if st.button("Set broadcast interval"):
        conn.send_message({"op":"broadcast_ticks","enable":True,"interval_sec":float(interval)})

    btext = st.text_input("Broadcast message", "Hello, everyone!")
    if st.button("Send broadcast message"):
        conn.send_message({"op":"broadcast","text": btext})

    st.divider()
    if st.button("Clear inbox"):
        st.session_state["inbox"].clear()

with col2:
    st.subheader(f"Incoming  •  {len(st.session_state['inbox'])} messages  •  server: `{ADDR}`")
    # Auto-expand: render newest first, no expanders
    for m in reversed(st.session_state["inbox"][-200:]):
        st.json(m)
