## Streamlit Websocket Comm Demo 🚀

```bash
# 1.a) install the lib + examples
pip install -e .[streamlit,examples]

# or just using sources
export PYTHONPATH="$PWD/src:$PYTHONPATH"

# 2) run the echo server
uvicorn examples.streamlit.streamlit_ws_duplex.server.main:app --reload --port 8011 --log-level debug

# 3) run the Streamlit demo
streamlit run examples/streamlit/streamlit_ws_duplex/client/app.py --server.port 8501
```

Choose **Dedicated** or **Shared** in the UI, then:

* Pause/resume **per-connection** ticks (only the current socket),
* Pause/resume **broadcast** ticks (all sockets),
* Change broadcast interval,
* Send a broadcast message,
* Send arbitrary JSON and see it echoed back.