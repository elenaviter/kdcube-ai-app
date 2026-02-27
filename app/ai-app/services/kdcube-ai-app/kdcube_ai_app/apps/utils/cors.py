# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

# apps/utils/cors.py

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from kdcube_ai_app.apps.chat.sdk.config import get_settings

def configure_cors(app: FastAPI):
    settings = get_settings()
    cors_config = settings.CORS_CONFIG_OBJ
    allow_origins = None
    if cors_config:
        allow_origins = cors_config.allow_origins
        allow_credentials = cors_config.allow_credentials
        allow_headers = cors_config.allow_headers
        allow_methods = cors_config.allow_methods

        app.add_middleware(
            CORSMiddleware,
            allow_origins=allow_origins,
            allow_credentials=bool(allow_credentials),
            allow_methods=allow_methods,
            allow_headers=allow_headers,
        )
    return allow_origins
