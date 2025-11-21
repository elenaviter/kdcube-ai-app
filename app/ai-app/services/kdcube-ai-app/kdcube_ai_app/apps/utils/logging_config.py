# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# logging_config.py
import logging
import os


def _to_level(name: str, default: int) -> int:
    try:
        return getattr(logging, (name or "").upper())
    except Exception:
        return default

def configure_logging():
    # --- Root config ---
    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = os.getenv("LOG_FORMAT",
                           "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    level = _to_level(log_level_name, logging.INFO)

    # Make root the single source of truth
    logging.basicConfig(level=level, format=log_format, force=True)

    # Optional: route warnings.warn(...) into logging
    logging.captureWarnings(True)

    # --- Normalize noisy / framework loggers ---
    # You can tune these via env if needed.
    desired_levels = {
        # uvicorn (you run with log_config=None, but set levels anyway)
        "uvicorn": os.getenv("UVICORN_LEVEL", log_level_name),
        "uvicorn.error": os.getenv("UVICORN_ERROR_LEVEL", log_level_name),
        "uvicorn.access": os.getenv("UVICORN_ACCESS_LEVEL", "WARNING"),

        # python-socketio / python-engineio
        "socketio": os.getenv("SOCKETIO_LEVEL", "INFO"),
        "socketio.server": os.getenv("SOCKETIO_LEVEL", "INFO"),
        "socketio.client": os.getenv("SOCKETIO_LEVEL", "INFO"),
        "engineio": os.getenv("ENGINEIO_LEVEL", "WARNING"),
        "engineio.server": os.getenv("ENGINEIO_LEVEL", "WARNING"),
        "engineio.client": os.getenv("ENGINEIO_LEVEL", "WARNING"),


        # asyncio, etc.
        "asyncio": os.getenv("ASYNCIO_LEVEL", "WARNING"),
        "watchfiles": os.getenv("WATCHFILES_LEVEL", "WARNING"),
        "aiohttp.access": os.getenv("AIOHTTP_ACCESS_LEVEL", "WARNING"),

        # 🧊 quiet AWS creds noise
        "aiobotocore.credentials": os.getenv("AIOBOTOCORE_CREDENTIALS_LEVEL", "WARNING"),
        # (optional, sometimes botocore itself also logs similarly)
        "botocore.credentials": os.getenv("BOTOCORE_CREDENTIALS_LEVEL", "WARNING"),
    }

    for name, lvl_name in desired_levels.items():
        lg = logging.getLogger(name)
        # Remove any handlers these libs may have attached (causes duplicates)
        for h in list(lg.handlers):
            lg.removeHandler(h)
        lg.propagate = True                 # bubble up to root only
        lg.setLevel(_to_level(lvl_name, level))

    # Mute noisy socket.io chatter
    logging.getLogger("socketio").setLevel(logging.WARNING)
    logging.getLogger("socketio.server").setLevel(logging.WARNING)
    # (optional) also quiet engineio ping/pong
    logging.getLogger("engineio").setLevel(logging.WARNING)
    logging.getLogger("engineio.server").setLevel(logging.WARNING)