"""
Distributed Locks maintenance API

File: apps/orchestration/orchestration_app.py
"""
from fastapi import APIRouter, HTTPException, Depends
import logging
import os
from datetime import datetime

from dotenv import load_dotenv, find_dotenv
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from kdcube_ai_app.auth.AuthManager import RequirementBase, RequireUser
from kdcube_ai_app.infra.orchestration.orchestration import IOrchestrator, \
    OrchestratorFactory

load_dotenv(find_dotenv())

ORCHESTRATOR_TYPE = os.environ.get("ORCHESTRATOR_TYPE", "dramatiq")
DEFAULT_ORCHESTRATOR_IDENTITY = f"kdcube_orchestrator_{ORCHESTRATOR_TYPE}"
ORCHESTRATOR_IDENTITY = os.environ.get("ORCHESTRATOR_IDENTITY", DEFAULT_ORCHESTRATOR_IDENTITY)

# .env
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "")
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = os.environ.get("REDIS_PORT", "6379")
REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0"

logger = logging.getLogger("Orchestrator.API")

# Create router
router = APIRouter()


def check_requirements(*requirements: RequirementBase):
    async def wrapper(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False))):
        if router.auth_manager is None:
            raise HTTPException(
                status_code=500,
                detail="AuthManager not configured"
            )
        return await router.auth_manager.check_requirements(credentials, *requirements)

    return wrapper


# Create orchestrator instance
orchestrator: IOrchestrator = OrchestratorFactory.create_orchestrator(
    orchestrator_type=ORCHESTRATOR_TYPE,
    redis_url=REDIS_URL,
    orchestrator_identity=ORCHESTRATOR_IDENTITY
)

logger.info(f"Initialized {ORCHESTRATOR_TYPE} orchestrator with identity: {ORCHESTRATOR_IDENTITY}")


@router.get("/task/{task_id}/status")
async def get_task_status(task_id: str,
                          user=Depends(check_requirements(RequireUser()))):
    """Get task status - works with any orchestrator!"""
    try:
        result = orchestrator.get_task_status(task_id)
        return {
            "orchestrator_type": ORCHESTRATOR_TYPE,
            "task_id": task_id,
            "status": result.status,
            "result": result.result,
            "error": result.error,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get task status for {task_id}: {e}; user: {user.id}")
        return {
            "orchestrator_type": ORCHESTRATOR_TYPE,
            "task_id": task_id,
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get("/queue/stats")
async def get_queue_stats(user=Depends(check_requirements(RequireUser()))):
    """Get queue statistics - orchestrator agnostic!"""
    try:
        stats = orchestrator.get_queue_stats()
        stats["timestamp"] = datetime.utcnow().isoformat()
        return stats
    except Exception as e:
        logger.error(f"Failed to get queue stats: {e}; user: {user.id}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info")
async def get_orchestrator_info():
    """Get orchestrator information"""
    return {
        "orchestrator_type": ORCHESTRATOR_TYPE,
        "orchestrator_identity": ORCHESTRATOR_IDENTITY,
        "redis_url": REDIS_URL.replace(REDIS_PASSWORD, "***") if REDIS_PASSWORD else REDIS_URL,
        "timestamp": datetime.utcnow().isoformat()
    }
