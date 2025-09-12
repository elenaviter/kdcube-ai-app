# Streamlit Communication Utilities. Websocket

Lightweight WebSocket helpers for Streamlit that make **duplex** (send & receive) easy.

## What’s inside

* **Dedicated connection** — one socket **per Streamlit session/tab**. Best when each user/tab must have its own feed.
* **Shared connection** — one socket **per Streamlit process** (singleton) that **fans out** messages to all sessions.

Both run `websocket-client` in background threads and **trigger safe reruns** when data arrives.

## Install

```bash
pip install kdcube-comm[streamlit]
# (optional demo server & app): pip install kdcube-comm[streamlit,examples]
```

## Message handling pattern (both modes)

On each rerun, append the last delivered message:

```python
if st.session_state.get("new_event") and "last_message" in st.session_state:
    st.session_state.setdefault("inbox", []).append(st.session_state["last_message"])
    st.session_state["new_event"] = False
```

## Use: Dedicated (per session)

```python
import json, streamlit as st
from kdcube_comm.streamlit.websocket.dedicated_ws_connection import Connection

ADDR = "ws://localhost:8011/ws"
def on_msg(d): 
    st.session_state.update(last_message=d, new_event=True)

conn = Connection(ADDR, on_message_cb=on_msg).connect()   # start socket for THIS session
# fold new data:
if st.session_state.get("new_event"): 
    st.session_state.setdefault("inbox", []).append(st.session_state["last_message"])
    st.session_state["new_event"] = False
# send:
conn.send_message({"op":"ping"})
# render:
for m in reversed(st.session_state.get("inbox", [])): 
    st.json(m)
```

## Use: Shared (singleton)

```python
import json, streamlit as st
from kdcube_comm.streamlit.websocket.shared_ws_connection import SharedConnection

ADDR = "ws://localhost:8011/ws"
conn = SharedConnection.get(ADDR)  # one process-wide socket, messages fanned out to all sessions
if st.session_state.get("new_event"): 
    st.session_state.setdefault("inbox", []).append(st.session_state["last_message"])
    st.session_state["new_event"] = False
conn.send_message({"op":"ping"})
for m in reversed(st.session_state.get("inbox", [])): st.json(m)
```

## Minimal API

* **Dedicated**: `Connection(address, on_message_cb).connect(); send_message(dict); await send_message_async(dict)`
* **Shared**: `SharedConnection.get(address); send_message(dict); await send_message_async(dict)`

> Tip: Use **Dedicated** when each tab/user needs isolated streams. Use **Shared** when you want one upstream WS and cheap fan-out to many Streamlit sessions.

Both connectors:
- Use `websocket-client` under the hood (threaded),
- Provide `send_message(...)` (sync) and `send_message_async(...)` (awaitable wrapper),
- Integrate with Streamlit’s threading context to trigger safe reruns on new messages.

> Example of usage: [FastAPI WS server + Streamlit demo app: streamlit_ws_duplex](../../../../examples/streamlit/streamlit_ws_duplex)

---