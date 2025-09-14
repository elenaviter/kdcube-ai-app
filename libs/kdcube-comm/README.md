# kdcube-comm

Communication utilities.

## Contents
### Utilities
| Tech       | namespace  | Utility                                                           | Description                                                   | How to install                          | Example                                                        | Notes |
|------------|------------|-------------------------------------------------------------------|---------------------------------------------------------------|-----------------------------------------|----------------------------------------------------------------|-------|
| Streamlit  | streamlit  | [Streamlit duplex websocket](src/kdcube_comm/streamlit/websocket) | Broadcast/direct websocket channel support for Streamlit apps   | `pip install -e .[streamlit,examples]`  | [streamlit_ws_duplex](examples/streamlit/streamlit_ws_duplex)  |       |

### Examples
- [examples](examples)

### Install
Install the library with feature and with examples:
```bash
pip install -e .[<namespace>,examples]
```

I.e.
```bash
pip install -e .[streamlit,examples]
```