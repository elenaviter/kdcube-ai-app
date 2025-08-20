# infra/accounting/__init__.py
"""
Self-contained accounting system with async-safe context isolation using contextvars
"""

import contextvars
import uuid
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from kdcube_ai_app.infra.accounting.usage import ServiceUsage


# ================================
# ASYNC-SAFE CONTEXT USING CONTEXTVARS
# ================================

class AccountingContext:
    """Accounting context data"""

    def __init__(self):
        self.user_id: Optional[str] = None
        self.session_id: Optional[str] = None
        self.project_id: Optional[str] = None
        self.tenant_id: Optional[str] = None
        self.request_id: Optional[str] = None
        self.component: Optional[str] = None  # Current component context
        self.extra: Dict[str, Any] = {}
        # Enrichment set by with_accounting(...):
        #   - seed_system_resources: List[SystemResource]
        #   - metadata: Dict[str, Any]
        #   - any other keys you decide
        self.event_enrichment: Dict[str, Any] = {}

    def update(self, **kwargs):
        """Update context fields"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.extra[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "project_id": self.project_id,
            "tenant_id": self.tenant_id,
            "request_id": self.request_id,
            "component": self.component,
            **self.extra
        }

# Context variables for async-safe storage
_context_var: contextvars.ContextVar[Optional[AccountingContext]] = contextvars.ContextVar(
    'accounting_context', default=None
)
_storage_var: contextvars.ContextVar[Optional['IAccountingStorage']] = contextvars.ContextVar(
    'accounting_storage', default=None
)
_default_storage = None

def _get_context() -> AccountingContext:
    """Get async-safe accounting context"""
    context = _context_var.get()
    if context is None:
        context = AccountingContext()
        _context_var.set(context)
    return context

def _get_enrichment() -> Dict[str, Any]:
    return _get_context().event_enrichment or {}

def _get_storage():
    """Get async-safe accounting storage"""
    return _storage_var.get()

def _set_storage(storage):
    """Set async-safe accounting storage"""
    _storage_var.set(storage)

def _set_context(context: AccountingContext):
    """Set async-safe accounting context"""
    _context_var.set(context)

# ================================
# PUBLIC API FOR CONTEXT MANAGEMENT
# ================================

def set_context(**kwargs):
    """Set accounting context fields"""
    context = _get_context()
    context.update(**kwargs)

def get_context() -> Dict[str, Any]:
    """Get current accounting context as dict"""
    return _get_context().to_dict()

def set_component(component: str):
    """Set current component context"""
    _get_context().component = component

def clear_context():
    """Clear accounting context"""
    _context_var.set(None)

def get_enrichment() -> Dict[str, Any]: return dict(_get_context().event_enrichment or {})

# ================================
# STORAGE INTERFACE AND IMPLEMENTATIONS
# ================================

class ServiceType(str, Enum):
    """Types of AI services that can be tracked"""
    LLM = "llm"
    EMBEDDING = "embedding"
    WEB_SEARCH = "web_search"
    IMAGE_GENERATION = "image_generation"
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"
    VISION = "vision"
    OTHER = "other"

@dataclass
class SystemResource:
    """System resource identifier"""
    resource_type: str
    resource_id: str
    rn: str
    resource_version: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AccountingEvent:
    """Accounting event record"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Context (from contextvars)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    project_id: Optional[str] = None
    tenant_id: Optional[str] = None
    request_id: Optional[str] = None
    component: Optional[str] = None

    # Service details
    service_type: ServiceType = ServiceType.OTHER
    provider: str = ""
    model_or_service: str = ""

    # Caller-provided resources/metadata (from with_accounting)
    seed_system_resources: List[SystemResource] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Usage & status
    usage: ServiceUsage = field(default_factory=ServiceUsage)

    # Status
    success: bool = True
    error_message: Optional[str] = None
    provider_request_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "project_id": self.project_id,
            "tenant_id": self.tenant_id,
            "request_id": self.request_id,
            "component": self.component,
            "service_type": self.service_type.value if hasattr(self.service_type, "value") else str(self.service_type),
            "provider": self.provider,
            "model_or_service": self.model_or_service,
            "seed_system_resources": [
                {
                    "resource_type": res.resource_type,
                    "resource_id": res.resource_id,
                    "rn": res.rn,
                    "resource_version": res.resource_version,
                    "metadata": res.metadata
                } for res in self.seed_system_resources
            ],
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
                "cache_creation_tokens": self.usage.cache_creation_tokens,
                "cache_read_tokens": self.usage.cache_read_tokens,
                "total_tokens": self.usage.total_tokens,
                "embedding_tokens": self.usage.embedding_tokens,
                "embedding_dimensions": self.usage.embedding_dimensions,
                "search_queries": self.usage.search_queries,
                "search_results": self.usage.search_results,
                "image_count": self.usage.image_count,
                "image_pixels": self.usage.image_pixels,
                "audio_seconds": self.usage.audio_seconds,
                "requests": self.usage.requests,
                "cost_usd": self.usage.cost_usd
            },
            "success": self.success,
            "error_message": self.error_message,
            "provider_request_id": self.provider_request_id,
            "metadata": self.metadata
        }

class IAccountingStorage(ABC):
    """Storage interface for accounting events"""

    @abstractmethod
    async def store_event(self, event: AccountingEvent) -> bool: ...

class FileAccountingStorage(IAccountingStorage):
    def __init__(self, storage_backend, base_path: str = "accounting",
                 path_strategy: Optional[Callable[['AccountingEvent'], str]] = None):
        self.storage_backend = storage_backend
        self.base_path = base_path.strip("/")
        self.logger = logging.getLogger(self.__class__.__name__)
        self.path_strategy = path_strategy

    def _default_path(self, event: AccountingEvent) -> str:
        dt = datetime.fromisoformat(event.timestamp.replace("Z","+00:00")) if event.timestamp else datetime.now()
        date_path = f"{dt.year:04d}/{dt.month:02d}/{dt.day:02d}"
        tenant = event.tenant_id or "unknown"
        project = event.project_id or "unknown"
        return f"{self.base_path}/{tenant}/{project}/{date_path}/{event.service_type.value}/{event.event_id}.json"

    async def store_event(self, event: AccountingEvent) -> bool:
        try:
            rel_path = self.path_strategy(event) if self.path_strategy else self._default_path(event)
            content = json.dumps(event.to_dict(), indent=2)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: self.storage_backend.write_text(rel_path, content))
            return True
        except Exception as e:
            self.logger.error(f"Failed to store accounting event {event.event_id}: {e}")
            return False

class NoOpAccountingStorage(IAccountingStorage):
    """No-op storage for when accounting is disabled"""

    async def store_event(self, event: AccountingEvent) -> bool:
        return True

# ================================
# DECORATOR SYSTEM
# ================================

class AccountingTracker:
    """Base class for tracking decorators"""

    def __init__(self,
                 service_type: ServiceType,
                 provider_extractor: Optional[Callable] = None,
                 model_extractor: Optional[Callable] = None,
                 usage_extractor: Optional[Callable] = None,
                 metadata_extractor: Optional[Callable] = None):
        self.service_type = service_type
        self.provider_extractor = provider_extractor
        self.model_extractor = model_extractor
        self.usage_extractor = usage_extractor
        self.metadata_extractor = metadata_extractor

    def _extract_provider(self, *a, **kw) -> str:
        """Extract provider name"""
        if self.provider_extractor: return self.provider_extractor(*a, **kw)
        for arg in a:
            if hasattr(arg, "provider") and hasattr(arg.provider, "provider"):
                pv = arg.provider.provider
                return pv.value if hasattr(pv, "value") else str(pv)
        if "model" in kw and hasattr(kw["model"], "provider"):
            pv = kw["model"].provider.provider
            return pv.value if hasattr(pv, "value") else str(pv)
        return "unknown"

    def _extract_model(self, *args, **kwargs) -> str:
        """Extract model name"""
        if self.model_extractor:
            return self.model_extractor(*args, **kwargs)

        if 'model' in kwargs:
            model_record = kwargs['model']
            if hasattr(model_record, 'systemName'):
                return model_record.systemName

        return "unknown"

    def _extract_usage(self, result: Any, *args, **kwargs) -> ServiceUsage:
        """Extract usage from result"""
        if self.usage_extractor:
            return self.usage_extractor(result, *args, **kwargs)

        # Default extraction based on result type
        if hasattr(result, 'usage') and result.usage:
            usage_obj = result.usage
            return ServiceUsage(
                input_tokens=getattr(usage_obj, 'input_tokens', 0),
                output_tokens=getattr(usage_obj, 'output_tokens', 0),
                cache_creation_tokens=getattr(usage_obj, 'cache_creation_tokens', 0),
                cache_read_tokens=getattr(usage_obj, 'cache_read_tokens', 0),
                total_tokens=getattr(usage_obj, 'total_tokens', 0),
                embedding_tokens=getattr(usage_obj, 'embedding_tokens', 0),
                embedding_dimensions=getattr(usage_obj, 'embedding_dimensions', 0),
                requests=1
            )

        # For embedding results (list of floats)
        if isinstance(result, list) and result and isinstance(result[0], (int, float)):
            return ServiceUsage(
                embedding_dimensions=len(result),
                requests=1
            )

        return ServiceUsage(requests=1)

    def _extract_metadata(self, *args, **kwargs) -> Dict[str, Any]:
        """Extract additional metadata"""
        if self.metadata_extractor:
            return self.metadata_extractor(*args, **kwargs)
        return {}

    def _create_event(self, result: Any, exception: Optional[Exception],
                      start_time: datetime, *args, **kwargs) -> AccountingEvent:
        """Create accounting event"""

        context = _get_context()

        enrich = context.event_enrichment or {}

        # Extract information
        provider = self._extract_provider(*args, **kwargs)
        model = self._extract_model(*args, **kwargs)
        usage = self._extract_usage(result, *args, **kwargs)
        meta = self._extract_metadata(*args, **kwargs)

        # merge in enrichment metadata (caller-provided data)
        extra_meta = dict(enrich.get("metadata") or {})
        extra_meta["processing_time_ms"] = (datetime.now() - start_time).total_seconds() * 1000
        meta.update(extra_meta)

        # caller-provided resources (or none)
        seeds: List[SystemResource] = enrich.get("seed_system_resources") or []

        # Determine success and error
        success = exception is None
        error_message = str(exception) if exception else None

        # Provider request ID
        provider_request_id = None
        if hasattr(result, 'provider_message_id'):
            provider_request_id = result.provider_message_id

        return AccountingEvent(
            user_id=context.user_id,
            session_id=context.session_id,
            project_id=context.project_id,
            tenant_id=context.tenant_id,
            request_id=context.request_id,
            component=context.component,
            service_type=self.service_type,
            provider=provider,
            model_or_service=model,
            seed_system_resources=seeds,
            usage=usage,
            success=success,
            error_message=error_message,
            provider_request_id=provider_request_id,
            metadata=meta
        )

    def __call__(self, func: Callable) -> Callable:
        """Decorator implementation"""

        if asyncio.iscoroutinefunction(func):
            import functools
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                storage = _get_storage()
                if not storage:
                    # No storage configured, just call function
                    return await func(*args, **kwargs)

                start_time = datetime.now()
                result = None
                exception = None

                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    exception = e
                    raise
                finally:
                    try:
                        event = self._create_event(result, exception, start_time, *args, **kwargs)
                        # store_coro = storage.store_event(event)
                        # asyncio.create_task(store_coro)
                        await storage.store_event(event)
                    except Exception as e:
                        # Log but don't fail the original function
                        logging.getLogger("accounting").error(f"Failed to create accounting event: {e}")

            return async_wrapper
        else:
            import functools
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                storage = _get_storage()
                if not storage:
                    return func(*args, **kwargs)

                start_time = datetime.now()
                result = None
                exception = None

                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    exception = e
                    raise
                finally:
                    try:
                        event = self._create_event(result, exception, start_time, *args, **kwargs)

                        try:
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                loop.create_task(storage.store_event(event))
                            else:
                                loop.run_until_complete(storage.store_event(event))
                        except RuntimeError:
                            # No event loop running, create new one
                            asyncio.run(storage.store_event(event))
                    except Exception as e:
                        logging.getLogger("accounting").error(f"Failed to create accounting event: {e}")

            return sync_wrapper

# ================================
# PREDEFINED TRACKERS
# ================================

def track_llm(provider_extractor=None, model_extractor=None,
              usage_extractor=None, metadata_extractor=None):
    """Decorator for tracking LLM usage"""
    return AccountingTracker(
        ServiceType.LLM, provider_extractor, model_extractor,
        usage_extractor, metadata_extractor
    )

def track_embedding(provider_extractor=None, model_extractor=None,
                    usage_extractor=None, metadata_extractor=None):
    """Decorator for tracking embedding usage"""
    return AccountingTracker(
        ServiceType.EMBEDDING, provider_extractor, model_extractor,
        usage_extractor, metadata_extractor
    )

def track_web_search(provider_extractor=None, model_extractor=None,
                     usage_extractor=None, metadata_extractor=None):
    """Decorator for tracking web search usage"""
    return AccountingTracker(
        ServiceType.WEB_SEARCH, provider_extractor, model_extractor,
        usage_extractor, metadata_extractor
    )

# ================================
# ACCOUNTING SYSTEM INTERFACE
# ================================

class AccountingSystem:
    """Main accounting system interface"""

    @staticmethod
    def init_storage(storage_backend,
                     enabled: bool = True,
                     *,
                     base_path: str = "accounting",
                     path_strategy: Optional[Callable[['AccountingEvent'], str]] = None):
        """Initialize accounting storage in context"""
        global _default_storage
        if enabled:
            if not path_strategy:
                path_strategy = grouped_by_component_and_seed()
            _default_storage = FileAccountingStorage(storage_backend, base_path=base_path, path_strategy=path_strategy)
        else:
            _default_storage = NoOpAccountingStorage()
        _set_storage(_default_storage)

    @staticmethod
    def set_context(**kwargs):
        """Set accounting context"""
        set_context(**kwargs)

    @staticmethod
    def set_component(component: str):
        """Set current component context"""
        set_component(component)

    @staticmethod
    def get_context() -> Dict[str, Any]:
        """Get current context"""
        return get_context()

    @staticmethod
    def clear_context():
        """Clear context"""
        clear_context()

# ================================
# USAGE HELPERS
# ================================

class with_accounting:
    """Context manager for setting component - async safe"""

    def __init__(self, component: str, **kwargs):
        self.component = component
        self.previous_component = None
        self._prev_enrichment = None
        self._new_enrichment = kwargs or {}

    def __enter__(self):
        ctx = _get_context()
        self.previous_component = ctx.component
        self._prev_enrichment = dict(ctx.event_enrichment or {})
        ctx.component = self.component
        # shallow-merge: inner scope can override keys
        merged = dict(self._prev_enrichment)
        merged.update(self._new_enrichment)
        ctx.event_enrichment = merged
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        ctx = _get_context()
        ctx.component = self.previous_component
        ctx.event_enrichment = self._prev_enrichment or {}

    async def __aenter__(self): return self.__enter__()
    async def __aexit__(self, *a): return self.__exit__(*a)

def grouped_by_component_and_seed() -> "callable":
    """
    Returns a function(event) -> relative path like:
      <tenant>/<project>/<YYYY.MM.DD>/<component|type|id|version>/usage_<timestamp>.json
    or, if no seeds:
      <tenant>/<project>/<YYYY.MM.DD>/<component>/usage_<timestamp>.json
    """
    def _strategy(event) -> str:
        # date folder: 2025.07.28
        dt = datetime.fromisoformat(event.timestamp.replace('Z', '+00:00')) if event.timestamp else datetime.now()
        date_folder = f"{dt.year:04d}.{dt.month:02d}.{dt.day:02d}"

        # group folder
        component = event.component or "unknown"
        service_type = event.service_type.value if hasattr(event.service_type, "value") else str(event.service_type)
        if event.seed_system_resources:
            r = event.seed_system_resources[0]  # pick the primary seed
            rtype = (r.resource_type or "res").strip()
            source_id = r.metadata.get("source_id") if r.metadata else None
            rid = (source_id or "unknown").strip()
            rver = str(r.resource_version) if r.resource_version is not None else "unknown"
            group = f"{component}|{rtype}|{rid}|{rver}"
        else:
            group = component

        # filename with ms; add short id suffix to avoid rare collisions
        ts = dt.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"usage_{ts}-{event.event_id[:8]}.json"

        tenant = event.tenant_id or "unknown"
        project = event.project_id or "unknown"

        # final path UNDER your base_path
        return f"{tenant}/{project}/{date_folder}/{service_type}/{group}/{filename}"
    return _strategy
# ================================
# EXPORT API
# ================================

__all__ = [
    # Core system
    'AccountingSystem',

    # Decorators
    'track_llm',
    'track_embedding',
    'track_web_search',

    # Context management
    'with_accounting',
    'set_context',
    'set_component',
    'get_context',

    # Data classes
    'AccountingEvent',
    'ServiceUsage',
    'ServiceType',
    'SystemResource',

    # Storage
    'IAccountingStorage',
    'FileAccountingStorage',
    'NoOpAccountingStorage'
]