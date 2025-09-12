# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# libs/kdcube-comm/streamlit/streamlit_ws_duplex/server/main.py

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio, json, time, uuid
from typing import Dict

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# ====== broadcast registry ======
connections: Dict[str, dict] = {}         # sid -> {"ws": WebSocket, "lock": asyncio.Lock}
broadcast_enabled = asyncio.Event()
broadcast_enabled.set()
broadcast_interval = 5.0

async def broadcast_loop():
    i = 0
    while True:
        await broadcast_enabled.wait()
        await asyncio.sleep(broadcast_interval)
        if not broadcast_enabled.is_set():
            continue

        dead = []
        payload = {"event": "broadcast_tick", "i": i, "ts": time.time()}
        text = json.dumps(payload)

        for sid, rec in list(connections.items()):
            ws = rec["ws"]
            lock: asyncio.Lock = rec["lock"]
            try:
                async with lock:
                    await ws.send_text(text)
            except Exception:
                dead.append(sid)
        for sid in dead:
            connections.pop(sid, None)
        i += 1

@app.on_event("startup")
async def _start():
    app.state.broadcast_task = asyncio.create_task(broadcast_loop())

@app.on_event("shutdown")
async def _stop():
    t = getattr(app.state, "broadcast_task", None)
    if t:
        t.cancel()

@app.get("/")
def root():
    return {"ok": True, "ws": "/ws"}

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    sid = uuid.uuid4().hex[:8]

    # register for broadcast mode
    connections[sid] = {"ws": ws, "lock": asyncio.Lock()}
    await ws.send_text(json.dumps({"event": "welcome", "sid": sid, "ts": time.time()}))
    print(f"[WS] connected sid={sid}")

    # per-connection ticker (dedicated mode)
    tick_enabled = asyncio.Event(); tick_enabled.set()
    tick_i = 0

    async def ticker():
        nonlocal tick_i
        try:
            while True:
                await tick_enabled.wait()
                await asyncio.sleep(3)
                if not tick_enabled.is_set():
                    continue
                try:
                    async with connections[sid]["lock"]:
                        await ws.send_text(json.dumps({"event": "tick", "i": tick_i, "ts": time.time()}))
                except Exception:
                    break
                tick_i += 1
        except Exception:
            pass

    tick_task = asyncio.create_task(ticker())

    try:
        while True:
            msg = await ws.receive_text()
            # --- LOG EVERY INCOMING MESSAGE ---
            print(f"[WS {sid}] RX: {msg}")

            try:
                data = json.loads(msg)
            except Exception:
                data = {"raw": msg}

            # ---- control: per-connection ticks ----
            if isinstance(data, dict) and data.get("op") == "ticks":
                enable = data.get("enable")
                if isinstance(enable, str):
                    enable = enable.lower() in ("1","true","yes","on")
                if enable:
                    tick_enabled.set()
                    ack = {"event": "ticks", "status": "resumed"}
                    print(f"[WS {sid}] per-conn ticks RESUMED")
                else:
                    tick_enabled.clear()
                    ack = {"event": "ticks", "status": "paused"}
                    print(f"[WS {sid}] per-conn ticks PAUSED")
                async with connections[sid]["lock"]:
                    await ws.send_text(json.dumps(ack))
                continue

            # ---- control: broadcast ticker ----
            if isinstance(data, dict) and data.get("op") == "broadcast_ticks":
                enable = data.get("enable")
                if isinstance(enable, str):
                    enable = enable.lower() in ("1","true","yes","on")
                interval = data.get("interval_sec")
                global broadcast_interval
                if isinstance(interval, (int, float)) and interval > 0.05:
                    broadcast_interval = float(interval)
                    print(f"[WS {sid}] broadcast interval set -> {broadcast_interval}s")
                if enable:
                    broadcast_enabled.set()
                    ack = {"event": "broadcast_ticks", "status": "resumed", "interval_sec": broadcast_interval}
                    print(f"[WS {sid}] broadcast ticks RESUMED")
                else:
                    broadcast_enabled.clear()
                    ack = {"event": "broadcast_ticks", "status": "paused"}
                    print(f"[WS {sid}] broadcast ticks PAUSED")
                async with connections[sid]["lock"]:
                    await ws.send_text(json.dumps(ack))
                continue

            # ---- control: broadcast message to everyone ----
            if isinstance(data, dict) and data.get("op") == "broadcast":
                text = str(data.get("text") or "")
                envelope = json.dumps({"event": "broadcast_message", "from": sid, "text": text, "ts": time.time()})
                dead = []
                for other_sid, rec in list(connections.items()):
                    try:
                        async with rec["lock"]:
                            await rec["ws"].send_text(envelope)
                    except Exception:
                        dead.append(other_sid)
                for d in dead:
                    connections.pop(d, None)
                # ack to sender
                async with connections.get(sid, {"lock": asyncio.Lock()})["lock"]:
                    await ws.send_text(json.dumps({"event": "broadcast_ack", "ok": True}))
                print(f"[WS {sid}] broadcast sent")
                continue

            # ---- default echo (to this client only) ----
            async with connections[sid]["lock"]:
                await ws.send_text(json.dumps({
                    "event": "echo",
                    "sid": sid,
                    "received": data,
                    "ts": time.time()
                }))
    except WebSocketDisconnect:
        print(f"[WS] disconnected sid={sid}")
    finally:
        tick_task.cancel()
        connections.pop(sid, None)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    import os, uvicorn
    os.environ.setdefault("PYTHONASYNCIODEBUG", "1")
    uvicorn.run("main:app",
                host="0.0.0.0",
                port=int(os.environ.get("WS_PORT", 8000)),
                reload=True,
                log_level="debug")
