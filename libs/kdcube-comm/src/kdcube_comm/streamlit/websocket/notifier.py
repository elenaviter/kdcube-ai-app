# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# libs/kdcube-comm/streamlit/websocket/notifier.py

"""
Internal helpers to trigger safe reruns across Streamlit sessions.

WARNING: This uses private Streamlit runtime APIs:
- Runtime.instance()._session_mgr.list_sessions()
- session._handle_rerun_script_request()

They work well for internal tools/demos, but are not officially supported.
"""

from __future__ import annotations
from typing import Dict, Text
import copy, threading

from streamlit.runtime import Runtime
from streamlit.runtime.app_session import AppSession
from streamlit.runtime.scriptrunner import add_script_run_ctx


def list_active_sessions() -> list[AppSession]:
    return [s.session for s in Runtime.instance()._session_mgr.list_sessions()]


def _attach_ctx(thread: threading.Thread, ctx) -> None:
    try:
        add_script_run_ctx(thread, ctx)
    except TypeError:
        add_script_run_ctx(thread)


def notify_all(state: Dict, thread: threading.Thread) -> None:
    """Apply 'state' to all sessions and request rerun."""
    for session in list_active_sessions():
        ss = session.session_state
        if "ctx" not in ss:
            continue
        ctx = ss["ctx"]
        _attach_ctx(thread, ctx)

        for k, v in state.items():
            ss[k] = copy.deepcopy(v)

        session._handle_rerun_script_request()


def notify(state: Dict, session_id: Text, thread: threading.Thread) -> None:
    """Apply 'state' to a specific session and request rerun."""
    session = next((s for s in list_active_sessions() if s.id == session_id), None)
    if not session:
        return
    ss = session.session_state
    if "ctx" not in ss:
        return

    ctx = ss["ctx"]
    _attach_ctx(thread, ctx)

    for k, v in state.items():
        ss[k] = copy.deepcopy(v)

    session._handle_rerun_script_request()