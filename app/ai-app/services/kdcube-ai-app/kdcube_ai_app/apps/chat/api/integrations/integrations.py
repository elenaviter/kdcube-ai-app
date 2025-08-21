# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

from typing import Optional
import logging

from fastapi import Query
from fastapi import APIRouter

from kdcube_ai_app.infra.plugin.agentic_loader import (
    AgenticBundleSpec, get_workflow_instance, clear_agentic_caches
)
from kdcube_ai_app.apps.chat.inventory import ConfigRequest, create_workflow_config

"""
Integrations API

File: api/integrations/integrations.py
"""


logger = logging.getLogger("KBMonitoring.API")

# Create router
router = APIRouter()

@router.get("/debug/agentic")
async def debug_agentic(
        load: bool = Query(False, description="If true, attempt to load & instantiate the workflow"),
        clear_cache: bool = Query(False, description="If true, clear loader caches before any operation"),
        bundle_path: Optional[str] = Query(None, description="Override of AGENTIC_BUNDLE_PATH"),
        bundle_module: Optional[str] = Query(None, description="Override of AGENTIC_BUNDLE_MODULE"),
        singleton: Optional[bool] = Query(None, description="Override of AGENTIC_SINGLETON"),
):
    """
    Returns info about agentic bundle configuration and (optionally) a real load probe.
    """
    import os

    if clear_cache:
        clear_agentic_caches()

    env_path = os.getenv("AGENTIC_BUNDLE_PATH")
    env_module = os.getenv("AGENTIC_BUNDLE_MODULE")
    env_singleton = os.getenv("AGENTIC_SINGLETON") in {"1", "true", "True"}

    path = bundle_path or env_path
    module = bundle_module or env_module
    use_singleton = env_singleton if singleton is None else bool(singleton)

    info = {
        "configured": {
            "AGENTIC_BUNDLE_PATH": env_path,
            "AGENTIC_BUNDLE_MODULE": env_module,
            "AGENTIC_SINGLETON": env_singleton,
        },
        "effective": {
            "path": path,
            "module": module,
            "singleton": use_singleton,
            "load_requested": load,
            "cache_cleared": clear_cache,
        },
    }

    if not load or not path:
        return info | {"note": "Set load=true and ensure a path is configured to test loading."}

    # Build a minimal config (or pull from query/body if you prefer)
    cfg_req = ConfigRequest(selected_model="gpt-4o")
    wf_config = create_workflow_config(cfg_req)

    try:
        spec = AgenticBundleSpec(path=path, module=module, singleton=use_singleton)
        wf, init_fn, mod = get_workflow_instance(
            spec,
            wf_config,
            step_emitter=None,
            delta_emitter=None,
        )
        return info | {
            "load_result": {
                "module": getattr(mod, "__name__", str(mod)),
                "workflow_type": type(wf).__name__,
                "has_initial_state": bool(init_fn),
            }
        }
    except Exception as e:
        return info | {"load_error": str(e)}