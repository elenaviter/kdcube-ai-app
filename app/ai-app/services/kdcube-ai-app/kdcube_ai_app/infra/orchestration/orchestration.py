# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# orchestrator_interface.py
import json
import time
import uuid
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from dataclasses import dataclass

import redis

logger = logging.getLogger("Orchestrator")

@dataclass
class TaskResult:
    """Standard task result across all orchestrators"""
    task_id: str
    status: str  # 'submitted', 'pending', 'running', 'completed', 'failed'
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class IOrchestrator(ABC):
    """
    Thin orchestrator interface - just submit_task with task name and kwargs.
    Implementations can use Celery, Dramatiq, or any other backend.
    """

    @abstractmethod
    def submit_task(self, task_name: str, queue: str = None, **kwargs) -> TaskResult:
        """
        Submit a task with given name and arguments.

        Args:
            task_name: Name of the task/actor that workers expect
            queue: Optional queue name (orchestrator-specific)
            **kwargs: Task arguments
        """
        pass

    @abstractmethod
    def get_task_status(self, task_id: str) -> TaskResult:
        """Get the current status of a task"""
        pass

    @abstractmethod
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        pass

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Check orchestrator health"""
        pass

# ==============================================================================
#                           CELERY IMPLEMENTATION
# ==============================================================================

class CeleryOrchestrator(IOrchestrator):
    """Celery-based orchestrator implementation"""

    def __init__(self, broker_url: str, backend_url: str):
        self.broker_url = broker_url
        self.backend_url = backend_url
        self._celery_app = None

    @property
    def celery_app(self):
        """Lazy initialization of Celery app"""
        if self._celery_app is None:
            from celery import Celery
            self._celery_app = Celery(
                "kb_processing",
                broker=self.broker_url,
                backend=self.backend_url,
            )
            self._celery_app.conf.update(
                task_serializer='json',
                accept_content=['json'],
                result_serializer='json',
            )
        return self._celery_app

    def submit_task(self, task_name: str, queue: str = None, **kwargs) -> TaskResult:
        """
        Submit task to Celery.

        Args:
            task_name: Exact task name that Celery workers expect
            queue: Optional queue/routing key for Celery
            **kwargs: Task arguments
        """
        try:
            # Convert kwargs to args list for Celery
            args = []
            for key in sorted(kwargs.keys()):  # Sorted for consistent order
                args.append(kwargs[key])

            # Celery send_task options
            options = {}
            if queue:
                options['queue'] = queue

            task = self.celery_app.send_task(task_name, args=args, **options)
            logger.info(f"Celery task '{task_name}' submitted to queue '{queue or 'default'}': {task.id}")
            return TaskResult(task_id=task.id, status='submitted')
        except Exception as e:
            logger.error(f"Failed to submit Celery task '{task_name}': {e}")
            return TaskResult(task_id="", status='failed', error=str(e))

    def get_task_status(self, task_id: str) -> TaskResult:
        """Get Celery task status"""
        try:
            task = self.celery_app.AsyncResult(task_id)
            status_map = {
                'PENDING': 'pending',
                'STARTED': 'running',
                'SUCCESS': 'completed',
                'FAILURE': 'failed',
                'RETRY': 'pending',
                'REVOKED': 'failed'
            }
            celery_status = task.status
            status = status_map.get(celery_status, 'unknown')

            result_data = None
            error_msg = None

            if celery_status == 'SUCCESS':
                result_data = task.result
            elif celery_status == 'FAILURE':
                error_msg = str(task.result)

            return TaskResult(
                task_id=task_id,
                status=status,
                result=result_data,
                error=error_msg
            )
        except Exception as e:
            logger.error(f"Failed to get Celery task status: {e}")
            return TaskResult(task_id=task_id, status='failed', error=str(e))

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get Celery queue statistics"""
        try:
            inspect = self.celery_app.control.inspect()
            stats = inspect.stats() or {}
            active_tasks = inspect.active() or {}

            return {
                "orchestrator_type": "celery",
                "workers": len(stats),
                "total_active_tasks": sum(len(tasks) for tasks in active_tasks.values()),
                "worker_stats": stats
            }
        except Exception as e:
            logger.error(f"Failed to get Celery stats: {e}")
            return {"orchestrator_type": "celery", "error": str(e)}

    def health_check(self) -> Dict[str, Any]:
        """Check Celery health"""
        try:
            inspect = self.celery_app.control.inspect()
            stats = inspect.stats()
            is_healthy = stats is not None and len(stats) > 0

            return {
                "orchestrator_type": "celery",
                "healthy": is_healthy,
                "active_workers": len(stats) if stats else 0,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Celery health check failed: {e}")
            return {
                "orchestrator_type": "celery",
                "healthy": False,
                "error": str(e),
                "timestamp": time.time()
            }

# ==============================================================================
#                         DRAMATIQ IMPLEMENTATION
# ==============================================================================

class DramatiqOrchestrator(IOrchestrator):
    """
    Dramatiq-based orchestrator implementation.
    Works via Redis without importing Dramatiq - perfect for remote workers!
    Generic infrastructure - no hardcoded task names or queues.
    """

    def __init__(self, redis_url: str, orchestrator_identity: str = "kdcube_orchestrator_dramatiq", default_queue: str = "default"):
        self.redis_url = redis_url
        self.orchestrator_identity = orchestrator_identity
        self.default_queue = default_queue
        self.redis_client = redis.Redis.from_url(redis_url)
        import uuid, json
        from dramatiq import Message
        from dramatiq.brokers.redis import RedisBroker
        import dramatiq

        self.broker = RedisBroker(url=redis_url)
        dramatiq.set_broker(self.broker)

    def _create_dramatiq_message(self, actor_name: str, args: list, kwargs: dict = None) -> dict:
        """Create a Dramatiq-compatible message"""
        message_id = str(uuid.uuid4())
        return {
            "id": message_id,
            "actor_name": actor_name,
            "args": args,
            "kwargs": kwargs or {},
            "options": {},
            "message_id": message_id,
            "message_timestamp": int(time.time() * 1000)  # Dramatiq uses milliseconds
        }

    def _enqueue_dramatiq_message(self,
                                  task_name: str,
                                  queue_name: str,
                                  data: dict) -> str:
        """Enqueue a message to Dramatiq via Redis"""
        try:
            # Dramatiq queue format: dramatiq:default:{queue_name}
            self.broker.declare_queue(queue_name)
            redis_message_id = str(uuid.uuid4())
            from dramatiq import Message
            # # args=(project, storage_path, resource_id, version, target_sid),
            msg = Message(
                queue_name=queue_name,
                actor_name=task_name,
                args=(),
                kwargs=data,
                options={"redis_message_id": redis_message_id}
            )
            # 3) Push it into Redis the *Dramatiq* way
            self.broker.enqueue(msg)
            return msg.message_id
            # queue_key = f"dramatiq:{queue_name}"
            # message_json = json.dumps(message)
            #
            #
            # # Push to Redis list (Dramatiq uses LPUSH for enqueueing)
            # self.redis_client.lpush(queue_key, message_json)
            # logger.info(f"Dramatiq message enqueued to {queue_name}: {message['id']}")
            # return message["id"]

        except Exception as e:
            logger.error(f"Failed to enqueue Dramatiq message: {e}")
            raise

    def submit_task(self, task_name: str, queue: str = None, *args, **kwargs) -> TaskResult:
        """
        Submit any actor with any positional or keyword arguments,
        using Dramatiq’s own broker & serialization so keys and envelopes match.
        """
        queue_name = queue or self.default_queue

        try:
            # 1) Create the same RedisBroker + middleware your workers use
            from dramatiq.brokers.redis import RedisBroker
            import dramatiq

            broker = RedisBroker(url=self.redis_url)
            dramatiq.set_broker(broker)

            # 2) Declare the queue so workers will BRPOP from it
            broker.declare_queue(queue_name)

            # 3) Build a properly formatted Message with all args/kwargs
            import uuid
            from dramatiq import Message

            msg = Message(
                queue_name=queue_name,
                actor_name=task_name,
                args=list(args),
                kwargs=kwargs,
                options={"redis_message_id": str(uuid.uuid4())}
            )

            # 4) Enqueue via the broker (correct key naming & dispatch script)
            broker.enqueue(msg)
            logger.info(f"Dramatiq task '{task_name}' enqueued to '{queue_name}' → {msg.message_id}")

            return TaskResult(task_id=msg.message_id, status="submitted")

        except Exception as e:
            logger.error(f"Failed to enqueue Dramatiq task '{task_name}': {e}")
            return TaskResult(task_id="", status="failed", error=str(e))

    def submit_task_(self, task_name: str, queue: str = None, **kwargs) -> TaskResult:
        """
        Submit task to Dramatiq via Redis.

        Args:
            task_name: Exact actor name that Dramatiq workers expect
            queue: Queue name (optional, uses default_queue if not provided)
            **kwargs: Task arguments
        """
        try:
            # Convert kwargs to args list for Dramatiq (sorted for consistency)
            args = []
            for key in sorted(kwargs.keys()):
                args.append(kwargs[key])

            # Use task_name as-is (client provides correct actor name)
            # message = self._create_dramatiq_message(task_name, args)

            # Use provided queue or default
            queue_name = queue or self.default_queue

            #task_id = self._enqueue_dramatiq_message(queue_name, message)
            task_id = self._enqueue_dramatiq_message(task_name, queue_name, args)
            logger.info(f"Dramatiq task '{task_name}' submitted to queue '{queue_name}': {task_id}")

            return TaskResult(task_id=task_id, status='submitted')

        except Exception as e:
            logger.error(f"Failed to submit Dramatiq task '{task_name}': {e}")
            return TaskResult(task_id="", status='failed', error=str(e))

    def get_task_status(self, task_id: str) -> TaskResult:
        """Get Dramatiq task status via Redis results backend"""
        try:
            # Dramatiq results are stored in Redis with key pattern
            result_key = f"dramatiq:result.{task_id}"
            result_data = self.redis_client.get(result_key)

            if result_data is None:
                return TaskResult(task_id=task_id, status='pending')

            # Parse result
            result_json = json.loads(result_data)

            # Check if it's a success or failure result
            if "error" in result_json:
                return TaskResult(
                    task_id=task_id,
                    status='failed',
                    error=result_json["error"]
                )
            else:
                return TaskResult(
                    task_id=task_id,
                    status='completed',
                    result=result_json
                )

        except Exception as e:
            logger.error(f"Failed to get Dramatiq task status: {e}")
            return TaskResult(task_id=task_id, status='failed', error=str(e))

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get Dramatiq queue statistics via Redis - discovers queues dynamically"""
        try:
            # Discover all Dramatiq queues by scanning Redis keys
            queue_pattern = "dramatiq:*"
            queue_keys = self.redis_client.keys(queue_pattern)

            stats = {}
            for queue_key in queue_keys:
                # Extract queue name from key (dramatiq:default.queue_name -> queue_name)
                queue_name = queue_key.decode('utf-8').replace('dramatiq:', '')
                length = self.redis_client.llen(queue_key)
                stats[queue_name] = length

            return {
                "orchestrator_type": "dramatiq",
                "queues": stats,
                "total_pending": sum(stats.values()) if stats else 0
            }

        except Exception as e:
            logger.error(f"Failed to get Dramatiq stats: {e}")
            return {"orchestrator_type": "dramatiq", "error": str(e)}

    def health_check(self) -> Dict[str, Any]:
        """Check Dramatiq health by testing Redis connectivity"""
        try:
            # Just check if we can connect to Redis and ping it
            self.redis_client.ping()

            return {
                "orchestrator_type": "dramatiq",
                "healthy": True,
                "redis_connected": True,
                "timestamp": time.time()
            }

        except Exception as e:
            logger.error(f"Dramatiq health check failed: {e}")
            return {
                "orchestrator_type": "dramatiq",
                "healthy": False,
                "error": str(e),
                "timestamp": time.time()
            }

# ==============================================================================
#                           ORCHESTRATOR FACTORY
# ==============================================================================

class OrchestratorFactory:
    """Factory to create the appropriate orchestrator based on configuration"""

    @staticmethod
    def create_orchestrator(
            orchestrator_type: str,
            redis_url: str,
            orchestrator_identity: str = None,
            default_queue: str = "default"
    ) -> IOrchestrator:
        """
        Create orchestrator instance based on type.

        Args:
            orchestrator_type: "celery" or "dramatiq"
            redis_url: Redis connection URL
            orchestrator_identity: Identity for the orchestrator
            default_queue: Default queue name for task submission
        """

        if orchestrator_type.lower() == "celery":
            return CeleryOrchestrator(
                broker_url=redis_url,
                backend_url=redis_url
            )

        elif orchestrator_type.lower() == "dramatiq":
            identity = orchestrator_identity or "kdcube_orchestrator_dramatiq"
            return DramatiqOrchestrator(
                redis_url=redis_url,
                orchestrator_identity=identity,
                default_queue=default_queue
            )

        else:
            raise ValueError(f"Unknown orchestrator type: {orchestrator_type}")

# ==============================================================================
#                              USAGE EXAMPLE
# ==============================================================================

def example_usage():
    """Example of how to use the generic orchestrator interface"""
    import os

    # Configuration
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    orchestrator_type = os.environ.get("ORCHESTRATOR_TYPE", "dramatiq")

    # Create orchestrator - completely generic!
    orchestrator = OrchestratorFactory.create_orchestrator(
        orchestrator_type=orchestrator_type,
        redis_url=redis_url
    )

    # CLIENT SIDE: Define your actual task names that workers expect
    if orchestrator_type == "dramatiq":
        # Use actual Dramatiq actor names
        KB_PROCESS_TASK = "process_kb_resource"
        KB_BATCH_TASK = "process_kb_resource_batch"
        KB_PROCESS_QUEUE = "kb_processing_high_priority"
        BATCH_QUEUE = "kb_processing_batch"
    else:  # celery
        # Use actual Celery task names
        KB_PROCESS_TASK = "kb.process_resource_in"
        KB_BATCH_TASK = "kb.process_resource_batch"
        KB_PROCESS_QUEUE = "high_priority"
        BATCH_QUEUE = "batch"

    # Submit a KB processing task - infrastructure is generic!
    result = orchestrator.submit_task(
        KB_PROCESS_TASK,  # Client provides correct task name
        queue=KB_PROCESS_QUEUE,  # Client provides correct queue
        project="test-project",
        storage_path="/path/to/storage",
        resource_id="resource-123",
        version="1.0",
        target_sid="socket-abc"
    )

    print(f"Task submitted: {result.task_id}, Status: {result.status}")

    # Submit a batch task
    batch_result = orchestrator.submit_task(
        KB_BATCH_TASK,  # Client provides correct task name
        queue=BATCH_QUEUE,  # Client provides correct queue
        project="test-project",
        storage_path="/path/to/storage",
        resource_ids=["resource-1", "resource-2", "resource-3"],
        target_sid="socket-def"
    )

    print(f"Batch task submitted: {batch_result.task_id}")

    # Generic infrastructure methods work the same
    status = orchestrator.get_task_status(result.task_id)
    stats = orchestrator.get_queue_stats()
    health = orchestrator.health_check()

    print(f"Task status: {status.status}")
    print(f"Stats: {stats}")
    print(f"Health: {health}")

if __name__ == "__main__":
    example_usage()