# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# knowledge_base/processor.py
"""
Knowledge Base processing actors for Dramatiq.
Contains the actual task definitions separated from server setup.
"""
import time
import logging
import os
from dotenv import load_dotenv, find_dotenv

from kdcube_ai_app.infra.accounting.envelope import AccountingEnvelope, bind_accounting
from kdcube_ai_app.infra.accounting import with_accounting
from kdcube_ai_app.infra.orchestration.app.communicator import ServiceCommunicator
from kdcube_ai_app.storage.storage import create_storage_backend

# Load environment
load_dotenv(find_dotenv())

# Logging
logger = logging.getLogger("KB.OrchestratorProcessor")
ORCHESTRATOR_TYPE = os.environ.get("ORCHESTRATOR_TYPE", "dramatiq")
DEFAULT_ORCHESTRATOR_IDENTITY = f"kdcube_orchestrator_{ORCHESTRATOR_TYPE}"
ORCHESTRATOR_IDENTITY = os.environ.get("ORCHESTRATOR_IDENTITY", DEFAULT_ORCHESTRATOR_IDENTITY)

# Global service communicator
sc = ServiceCommunicator(orchestrator_identity=ORCHESTRATOR_IDENTITY)

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
    task_start = time.time()
    worker_pid = os.getpid()

    # Import here - avoids any import issues and works perfectly with Dramatiq

    from kdcube_ai_app.apps.knowledge_base.core import KnowledgeBase
    envelope = AccountingEnvelope.from_dict(ctx)
    tenant = envelope.tenant_id
    project = envelope.project_id

    logger.info(f"[WORKER-{worker_pid}] START processing resource '{resource_id}' version '{version}' for project '{project}'")

    STORAGE_KWARGS = {}
    kdcube_storage_backend = create_storage_backend(kdcube_path, **STORAGE_KWARGS)

    async with bind_accounting(envelope, storage_backend=kdcube_storage_backend, enabled=True):
        async with with_accounting("kb.orchestrator.process_kb_resource"):
            try:
                embedding_model = stages_config.get("embedding", {}).get("model_record")
                # Create KB instance
                kb_workdir = f"{kdcube_path}/kb/tenants/{tenant}/projects/{project}/knowledge_base"
                kb = KnowledgeBase(tenant, project, kb_workdir, embedding_model=embedding_model, processing_mode=processing_mode)

                async def emit_progress(event_name: str, progress: float, message_text: str):
                    """Helper to emit progress events"""
                    payload = {
                        "resource_id": resource_id,
                        "version": version,
                        "progress": progress,
                        "message": message_text,
                        "worker_pid": worker_pid,
                        "elapsed_time": time.time() - task_start
                    }
                    await sc.pub(event_name, target_sid, payload)

                # Processing stages with detailed progress reporting
                await emit_progress("processing_started", 0.0, f"Starting KB processing on worker {worker_pid}")

                # Stage 1: Extraction
                stage_start = time.time()
                await emit_progress("processing_extraction", 0.1, "Extracting content from document")

                await kb.extract_only(resource_id, version)

                stage_duration = time.time() - stage_start
                await emit_progress("processing_extraction_complete", 0.25, f"Extraction complete ({stage_duration:.2f}s)")
                logger.info(f"  ✓ Extraction done ({stage_duration:.1f}s)")

                # Stage 2: Segmentation
                stage_start = time.time()
                await emit_progress("processing_segmentation", 0.25, "Creating content segments")

                await kb.process_resource(resource_id, version, stages=["segmentation"])

                stage_duration = time.time() - stage_start
                await emit_progress("processing_segmentation_complete", 0.5, f"Segmentation complete ({stage_duration:.2f}s)")
                logger.info(f"  ✓ Segmentation done ({stage_duration:.1f}s)")

                # Calculate progress steps based on optional stages
                metadata_stage_config = stages_config.get("metadata")
                search_indexing_stage_config = stages_config.get("search_indexing")

                # Dynamic progress calculation
                current_progress = 0.5
                remaining_stages = []
                if metadata_stage_config:
                    remaining_stages.append("metadata")
                remaining_stages.append("embedding")  # Always include embedding
                if search_indexing_stage_config:
                    remaining_stages.append("search_indexing")

                progress_step = (0.95 - current_progress) / len(remaining_stages)

                # Stage 3: Metadata (optional)
                if metadata_stage_config:
                    try:
                        stage_start = time.time()
                        await emit_progress("processing_metadata", current_progress, "Analyzing metadata")

                        await kb.process_resource(resource_id,
                                                  version,
                                                  stages=["metadata"],
                                                  stages_config=stages_config,
                                                  force_reprocess=True)

                        stage_duration = time.time() - stage_start
                        current_progress += progress_step
                        await emit_progress("processing_metadata_complete", current_progress, f"Metadata analysis complete ({stage_duration:.2f}s)")
                        logger.info(f"  ✓ Metadata done ({stage_duration:.1f}s)")

                    except Exception as e:
                        logger.warning(f"Metadata stage non-critical failure: {e}")

                # Stage 4: Embedding
                logger.info("Processing embeddings...")
                try:
                    stage_start = time.time()
                    await emit_progress("processing_embedding", current_progress, "Calculating embeddings")

                    await kb.process_resource(resource_id, version,
                                              stages=["embedding"],
                                              stages_config=stages_config,
                                              force_reprocess=True)

                    stage_duration = time.time() - stage_start
                    current_progress += progress_step
                    await emit_progress("processing_embedding_complete", current_progress, f"Embedding calculation complete ({stage_duration:.2f}s)")
                    logger.info(f"  ✓ Embeddings done ({stage_duration:.1f}s)")

                except Exception as e:
                    logger.warning(f"Embeddings stage non-critical failure: {e}")

                # Stage 5: Search Indexing (optional)
                if search_indexing_stage_config:
                    try:
                        stage_start = time.time()
                        await emit_progress("processing_search_indexing", current_progress, "Indexing segments for search")

                        await kb.process_resource(resource_id, version,
                                                  stages=["search_indexing"],
                                                  stages_config=stages_config,
                                                  force_reprocess=True)

                        stage_duration = time.time() - stage_start
                        current_progress += progress_step
                        await emit_progress("processing_search_indexing_complete", current_progress, f"Search indexing complete ({stage_duration:.2f}s)")
                        logger.info(f"  ✓ Search indexing done ({stage_duration:.1f}s)")

                    except Exception as e:
                        logger.warning(f"Search indexing stage failure: {e}")
                        # Don't fail the entire job if search indexing fails
                        await emit_progress("processing_search_indexing_failed", current_progress, f"Search indexing failed: {str(e)}")

                # Final completion
                total_duration = time.time() - task_start

                # Summary message with completed stages
                completed_stages = ["extraction", "segmentation"]
                if metadata_stage_config:
                    completed_stages.append("metadata")
                completed_stages.append("embedding")
                if search_indexing_stage_config:
                    completed_stages.append("search_indexing")

                stages_summary = ", ".join(completed_stages)
                completion_message = f"KB processing completed ({stages_summary}) in {total_duration:.2f}s"

                await emit_progress("processing_completed", 1.0, completion_message)

                logger.info(f"[WORKER-{worker_pid}] KB processing succeeded for resource '{resource_id}' in {total_duration:.2f}s")
                logger.info(f"  Completed stages: {stages_summary}")

                # Log successful completion (replaces success callback)
                logger.info(f"TASK SUCCESS: Resource {resource_id} processed successfully by worker {worker_pid}")

                # Return result with stage information
                return {
                    "resource_id": resource_id,
                    "version": version,
                    "duration": total_duration,
                    "worker_pid": worker_pid,
                    "status": "completed",
                    "completed_stages": completed_stages,
                    "search_indexed": search_indexing_stage_config is not None
                }

            except Exception as e:
                total_duration = time.time() - task_start
                logger.error(f"[WORKER-{worker_pid}] KB processing failed for '{resource_id}' after {total_duration:.2f}s: {e}")

                # Emit failure event
                error_payload = {
                    "resource_id": resource_id,
                    "version": version,
                    "error": str(e),
                    "worker_pid": worker_pid,
                    "duration": total_duration
                }
                await sc.pub("processing_failed", target_sid, error_payload)

                # Re-raise for orchestator(dramatiq) to handle retries
                raise
