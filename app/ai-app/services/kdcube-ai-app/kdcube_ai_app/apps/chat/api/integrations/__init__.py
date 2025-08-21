# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

"""
Modular integrations api management system using FastAPI routers.
File: api/integrations/__init__.py
"""
from typing import Callable

from fastapi import FastAPI

from .integrations import router as integrations_router


def mount_integrations_routers(app: FastAPI):
    """
    Mount all monitoring routers to the FastAPI app.

    Args:
        app: Your existing FastAPI application
        auth_manager: AuthManager instance
    """

    # Mount content rebuild router
    integrations_router.state = app.state
    app.include_router(
        monitoring_router,
        prefix="",
        tags=["CB integrations"],
    )
    return app


# Export for convenience
__all__ = ["mount_integrations_routers"]
