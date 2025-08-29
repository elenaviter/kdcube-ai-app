# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# kb_actors.py
"""
Knowledge Base processing actors for Dramatiq.
Contains the actual task definitions separated from server setup.
"""
import time
import logging
import os
import dramatiq
from dotenv import load_dotenv, find_dotenv

from kdcube_ai_app.infra.orchestration.app.communicator import ServiceCommunicator

# Load environment
load_dotenv(find_dotenv())

# Logging
logger = logging.getLogger("KB.Actors")

# Configuration
# Queue prefix to match web server expectations
KDCUBE_ORCHESTRATOR_QUEUES_PREFIX = "kdcube_orch_"

ORCHESTRATOR_TYPE = os.environ.get("ORCHESTRATOR_TYPE", "dramatiq")
DEFAULT_ORCHESTRATOR_IDENTITY = f"kdcube_orchestrator_{ORCHESTRATOR_TYPE}"
ORCHESTRATOR_IDENTITY = os.environ.get("ORCHESTRATOR_IDENTITY", DEFAULT_ORCHESTRATOR_IDENTITY)

# ==============================================================================
#                              CALLBACK ACTORS
# ==============================================================================

# Note: These are commented out because we handle success/failure directly in the main actor
# to avoid middleware configuration complexity

# @dramatiq.actor(queue_name="health_check")
# def task_success_callback(task_result):
#     """Called when a task succeeds"""
#     logger.info(f"Task succeeded: {task_result}")

# @dramatiq.actor(queue_name="health_check")
# def task_failure_callback(task_id, exception_data):
#     """Called when a task fails after all retries"""
#     logger.error(f"Task {task_id} failed permanently: {exception_data}")

# Global service communicator
# sc = ServiceCommunicator(orchestrator_identity=ORCHESTRATOR_IDENTITY)

# ==============================================================================
#                          MAIN KB PROCESSING ACTOR
# ==============================================================================

@dramatiq.actor(
    actor_name="process_kb_resource",
    max_retries=3,
    min_backoff=1000,        # 1 second minimum backoff
    max_backoff=300000,      # 5 minute maximum backoff
    time_limit=3600000,      # 1 hour timeout
    queue_name=f"{KDCUBE_ORCHESTRATOR_QUEUES_PREFIX}low_priority",
    store_results=True,
    # callbacks=[task_success_callback, task_failure_callback]
)
async def process_kb_resource(  kdcube_path: str,
                                resource_id: str,
                                version: str,
                                target_sid: str,
                                processing_mode: str,
                                stages_config: dict,
                                ctx: dict = None):
    """
    Main KB processing actor.

    Args:

        storage_path: Storage path for KB
        resource_id: Resource ID to process
        version: Resource version
        target_sid: Socket.IO session ID for progress updates
        processing_mode: Processing mode for the KB
        stages_config: Configuration for each stage
        ctx: execution ctx
    """
    from kdcube_ai_app.apps.knowledge_base.processor import process_kb_resource as pkb

    await pkb(kdcube_path, resource_id, version, target_sid, processing_mode, stages_config, ctx)

# ==============================================================================
#                              HEALTH CHECK ACTOR
# ==============================================================================

@dramatiq.actor(queue_name="health_check")
def health_check():
    """Simple health check task"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "worker_pid": os.getpid(),
        "orchestrator_identity": ORCHESTRATOR_IDENTITY
    }
