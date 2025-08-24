# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/inventory.py
import asyncio
import json
import os
import logging
from datetime import datetime
from uuid import uuid4

import aiohttp
import requests
import time
from typing import Optional, Any, Dict, List, AsyncIterator, Callable, Awaitable

from langchain_core.embeddings import Embeddings
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from pydantic import BaseModel

from kdcube_ai_app.apps.chat.reg import MODEL_CONFIGS, EMBEDDERS
from kdcube_ai_app.infra.accounting import track_llm
from kdcube_ai_app.infra.accounting.usage import _structured_usage_extractor, \
    _norm_usage_dict, _approx_tokens_by_chars, ServiceUsage, ClientConfigHint

def _mid(prefix: str = "m") -> str:
    return f"{prefix}-{uuid4().hex}"

class Config:
    """Configuration for the application"""
    def __init__(self,
                 selected_model: str = "gpt-4o",
                 openai_api_key: Optional[str] = None,
                 claude_api_key: Optional[str] = None,
                 embedding_model: Optional[str] = None):

        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        self.claude_api_key = claude_api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.selected_model = selected_model
        self.model_config = MODEL_CONFIGS.get(selected_model, MODEL_CONFIGS["gpt-4o"])

        # New declarative embedding configuration
        self.selected_embedder = "openai-text-embedding-3-small"  # Default
        self.embedder_config = EMBEDDERS.get(self.selected_embedder, EMBEDDERS["openai-text-embedding-3-small"])


        self.embedding_model = embedding_model or "text-embedding-3-small"
        self.custom_embedding_endpoint = None
        self.custom_embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.custom_embedding_size = 384

        # Enhanced logging configuration
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.enable_performance_logging = True
        self.enable_model_call_logging = True

        # Independent models
        self.format_fixer_model = "claude-3-haiku-20240307"

        # Set models based on selected model
        self.has_classifier = self.model_config["has_classifier"]
        self.classifier_model = self.model_config["model_name"]
        self.query_writer_model = self.model_config["model_name"]
        self.reranker_model = self.model_config["model_name"]
        self.answer_generator_model = self.model_config["model_name"]
        self.provider = self.model_config["provider"]

        # Custom endpoint support
        self.custom_model_endpoint = os.getenv("CUSTOM_MODEL_ENDPOINT", "")
        self.custom_model_api_key = os.getenv("CUSTOM_MODEL_API_KEY", "")
        self.custom_model_name = os.getenv("CUSTOM_MODEL_NAME", "custom-model")
        self.use_custom_endpoint = bool(self.custom_model_endpoint)

        self.kb_search_url = os.getenv("KB_SEARCH_URL", None)

    def set_embedder(self, embedder_id: str, custom_endpoint: str = None):
        """Set the embedder declaratively"""
        if embedder_id not in EMBEDDERS:
            raise ValueError(f"Unknown embedder: {embedder_id}")

        self.selected_embedder = embedder_id
        self.embedder_config = EMBEDDERS[embedder_id]

        # If it's a custom embedder, require endpoint
        if self.embedder_config["provider"] == "custom":
            if not custom_endpoint:
                raise ValueError("Custom embedders require an endpoint")
            self.custom_embedding_endpoint = custom_endpoint
            self.custom_embedding_model = self.embedder_config["model_name"]
            self.custom_embedding_size = self.embedder_config["dim"]
        else:
            # OpenAI embedder
            self.custom_embedding_endpoint = None
            self.embedding_model = self.embedder_config["model_name"]

    def set_custom_embedding_endpoint(self, endpoint: str, model: str = None, size: int = None):
        """Set custom embedding endpoint"""
        self.custom_embedding_endpoint = endpoint
        if model:
            self.custom_embedding_model = model
        if size:
            self.custom_embedding_size = size

    def set_kb_search_endpoint(self, endpoint: str):
        self.kb_search_url = endpoint

class ConfigRequest(BaseModel):
    openai_api_key: Optional[str] = None
    claude_api_key: Optional[str] = None
    custom_model_endpoint: Optional[str] = None
    custom_model_api_key: Optional[str] = None
    custom_model_name: Optional[str] = None
    kb_search_endpoint: Optional[str] = None

    # Embeddings
    selected_embedder: str = "openai-text-embedding-3-small"
    custom_embedding_endpoint: Optional[str] = None

    # Legacy
    custom_embedding_model: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2"
    custom_embedding_size: Optional[int] = 384

    selected_model: str = "gpt-4o"

    # Bundle selection
    agentic_bundle_id: Optional[str] = None

class AgentLogger:
    """Enhanced logging for agents with structured output"""

    def __init__(self, name: str, log_level: str = "INFO"):
        self.logger = logging.getLogger(f"agent.{name}")
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Create formatter for structured logs
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Console handler
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        self.start_time = None
        self.execution_logs = []

    def start_operation(self, operation: str, **kwargs):
        """Start timing an operation"""
        self.start_time = time.time()
        log_data = {
            "operation": operation,
            "timestamp": datetime.now().isoformat(),
            "inputs": kwargs
        }
        self.logger.info(f"🚀 Starting {operation} - {json.dumps(log_data, indent=2)}")
        return log_data

    def log_step(self, step: str, data: Any = None, level: str = "INFO"):
        """Log a processing step"""
        log_entry = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "data": data if data is not None else "No data"
        }

        if self.start_time:
            log_entry["elapsed_time"] = f"{time.time() - self.start_time:.2f}s"

        self.execution_logs.append(log_entry)

        log_level = getattr(self.logger, level.lower())
        log_level(f"📋 {step} - {json.dumps(log_entry, indent=2, default=str)}")

    def log_model_call(self, model_name: str, prompt_length: int, response_length: int = None, success: bool = True):
        """Log model API calls"""
        log_data = {
            "model": model_name,
            "prompt_length": prompt_length,
            "response_length": response_length,
            "success": success,
            "timestamp": datetime.now().isoformat()
        }

        if self.start_time:
            log_data["elapsed_time"] = f"{time.time() - self.start_time:.2f}s"

        status_emoji = "✅" if success else "❌"
        self.logger.info(f"{status_emoji} Model Call - {json.dumps(log_data, indent=2)}")

    def log_error(self, error: Exception, context: str = None):
        """Log errors with context"""
        error_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "timestamp": datetime.now().isoformat()
        }

        if self.start_time:
            error_data["elapsed_time"] = f"{time.time() - self.start_time:.2f}s"

        self.logger.error(f"💥 Error - {json.dumps(error_data, indent=2)}")

    def finish_operation(self, success: bool = True, result_summary: str = None):
        """Finish timing an operation"""
        if self.start_time:
            total_time = time.time() - self.start_time
            status_emoji = "🎉" if success else "💥"

            summary_data = {
                "success": success,
                "total_time": f"{total_time:.2f}s",
                "result_summary": result_summary,
                "total_steps": len(self.execution_logs),
                "timestamp": datetime.now().isoformat()
            }

            self.logger.info(f"{status_emoji} Operation Complete - {json.dumps(summary_data, indent=2)}")

            # Reset for next operation
            self.start_time = None
            self.execution_logs = []

            return summary_data

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of current execution"""
        return {
            "total_steps": len(self.execution_logs),
            "execution_logs": self.execution_logs,
            "elapsed_time": f"{time.time() - self.start_time:.2f}s" if self.start_time else None
        }

def ms_provider_extractor(model_service, client, *args, **kw) -> str:
    """First arg is 'self' (ModelService instance)."""
    cfg = kw.get("client_cfg") or model_service.describe_client(client)
    return getattr(cfg, "provider", "unknown")

def ms_model_extractor(model_service, client, *args, **kw) -> str:
    cfg = kw.get("client_cfg") or model_service.describe_client(client)
    return getattr(cfg, "model_name", "unknown")

def ms_structured_meta_extractor(model_service, _client, system_prompt: str, user_message: str, response_format, **kw):
    client_cfg = kw.get("client_cfg")
    return {
        "selected_model": (client_cfg.model_name if client_cfg else getattr(model_service.config, "selected_model", None)),
        "provider": (client_cfg.provider if client_cfg else getattr(model_service.config, "provider", None)),
        "expected_format": getattr(response_format, "__name__", str(response_format)),
        "prompt_chars": len(system_prompt or "") + len(user_message or ""),
        "temperature": kw.get("temperature"),
        "max_tokens": kw.get("max_tokens"),
    }

def ms_freeform_meta_extractor(model_service, _client, messages, *a, **kw):
    try:
        prompt_chars = sum(len(getattr(m, "content", "") or "") for m in (messages or []))
    except Exception:
        prompt_chars = 0
    client_cfg = kw.get("client_cfg")
    return {
        "selected_model": (client_cfg.model_name if client_cfg else getattr(model_service.config, "selected_model", None)),
        "provider": (client_cfg.provider if client_cfg else getattr(model_service.config, "provider", None)),
        "prompt_chars": prompt_chars,
        "temperature": kw.get("temperature"),
        "max_tokens": kw.get("max_tokens"),
    }

class ModelServiceBase:
    """Handles interactions with different models"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = AgentLogger("ModelServiceBase", config.log_level)

        # Create clients based on provider
        if config.use_custom_endpoint:
            self.classifier_client = CustomModelClient(
                endpoint=config.custom_model_endpoint,
                api_key=config.custom_model_api_key,
                model_name=config.custom_model_name,
                temperature=0.1
            )
            self.query_writer_client = CustomModelClient(
                endpoint=config.custom_model_endpoint,
                api_key=config.custom_model_api_key,
                model_name=config.custom_model_name,
                temperature=0.3
            )
            self.reranker_client = CustomModelClient(
                endpoint=config.custom_model_endpoint,
                api_key=config.custom_model_api_key,
                model_name=config.custom_model_name,
                temperature=0.1
            )
            self.answer_generator_client = CustomModelClient(
                endpoint=config.custom_model_endpoint,
                api_key=config.custom_model_api_key,
                model_name=config.custom_model_name,
                temperature=0.3
            )
        elif config.provider == "anthropic":
            try:
                import anthropic
                self.classifier_client = anthropic.Anthropic(api_key=config.claude_api_key)
                self.query_writer_client = anthropic.Anthropic(api_key=config.claude_api_key)
                self.reranker_client = anthropic.Anthropic(api_key=config.claude_api_key)
                self.answer_generator_client = anthropic.Anthropic(api_key=config.claude_api_key)
            except ImportError:
                self.logger.log_error(ImportError("anthropic package not available"), "Anthropic client initialization")
                # Fallback to OpenAI
                self._create_openai_clients(config)
        else:  # OpenAI
            self._create_openai_clients(config)

        self.logger.log_step("model_service_initialized", {
            "selected_model": config.selected_model,
            "provider": config.provider,
            "has_classifier": config.has_classifier,
            "model_name": config.model_config["model_name"]
        })

    # chat/inventory.py (inside ModelServiceBase)
    def describe_client(self, client, role: Optional[str] = None) -> ClientConfigHint:
        # Custom endpoint
        if isinstance(client, CustomModelClient):
            return ClientConfigHint(provider="custom", model_name=client.model_name)

        # OpenAI via LangChain
        if isinstance(client, ChatOpenAI):
            return ClientConfigHint(provider="openai", model_name=getattr(client, "model", self.config.model_config.get("model_name", self.config.selected_model)))

        # Anthropic SDK client (has .messages)
        if hasattr(client, "messages"):
            # pick role-specific model if we can
            if client is getattr(self, "classifier_client", None):
                return ClientConfigHint(provider="anthropic", model_name=self.config.classifier_model)
            if client is getattr(self, "query_writer_client", None):
                return ClientConfigHint(provider="anthropic", model_name=self.config.query_writer_model)
            if client is getattr(self, "reranker_client", None):
                return ClientConfigHint(provider="anthropic", model_name=self.config.reranker_model)
            if client is getattr(self, "answer_generator_client", None):
                return ClientConfigHint(provider="anthropic", model_name=self.config.answer_generator_model)
            # fallback
            return ClientConfigHint(provider="anthropic", model_name=self.config.model_config.get("model_name", self.config.selected_model))

        # ultimate fallback
        return ClientConfigHint(provider=self.config.provider or "unknown",
                                model_name=self.config.model_config.get("model_name", self.config.selected_model))

    def _create_openai_clients(self, config):
        """Create OpenAI clients"""
        self.classifier_client = ChatOpenAI(
            model=config.classifier_model,
            api_key=config.openai_api_key,
            temperature=0.1,
            stream_usage=True,
        )
        self.query_writer_client = ChatOpenAI(
            model=config.query_writer_model,
            api_key=config.openai_api_key,
            temperature=0.3,
            stream_usage=True,
        )
        self.reranker_client = ChatOpenAI(
            model=config.reranker_model,
            api_key=config.openai_api_key,
            temperature=0.1,
            stream_usage=True,
        )
        self.answer_generator_client = ChatOpenAI(
            model=config.answer_generator_model,
            api_key=config.openai_api_key,
            temperature=0.3,
            stream_usage=True,
        )

    # --- usage extractor for freeform (make it static so no 'self' needed) ---
    @staticmethod
    def _freeform_usage_extractor(result, *_a, **_kw) -> ServiceUsage:
        try:
            u = _norm_usage_dict(result.get("usage") or {})
            return ServiceUsage(
                input_tokens=u["prompt_tokens"],
                output_tokens=u["completion_tokens"],
                total_tokens=u["total_tokens"],
                requests=1,
            )
        except Exception:
            return ServiceUsage(requests=1)

    @track_llm(
        provider_extractor=ms_provider_extractor,
        model_extractor=ms_model_extractor,
        usage_extractor=_structured_usage_extractor,         # reuse your shared normalizer
        metadata_extractor=ms_structured_meta_extractor,
    )
    async def call_model_with_structure(self,
                                        client, system_prompt: str, user_message: str,
                                        response_format: BaseModel,
                                        *, client_cfg: Optional[ClientConfigHint] = None) -> Dict[str, Any]:
        operation_start = self.logger.start_operation(
            "model_call_structured",
            system_prompt_length=len(system_prompt),
            user_message_length=len(user_message),
            expected_format=response_format.__name__
        )

        usage: Dict[str, int] = {}
        provider_message_id = None
        # model_name = self.config.model_config.get("model_name", self.config.selected_model)

        cfg = client_cfg or self.describe_client(client)
        provider_name = cfg.provider
        model_name = cfg.model_name

        try:
            self.logger.log_step("sending_request", {
                "system_prompt_preview": system_prompt[:200] + "..." if len(system_prompt) > 200 else system_prompt,
                "user_message_preview": user_message[:200] + "..." if len(user_message) > 200 else user_message
            })

            # if self.config.provider == "anthropic" and hasattr(client, 'messages'):
            if provider_name == "anthropic" and hasattr(client, "messages"):
                # Anthropic SDK
                resp = client.messages.create(
                    model=model_name,
                    max_tokens=1000,
                    messages=[{"role": "user", "content": f"{system_prompt}\n\n{user_message}"}]
                )
                response_content = resp.content[0].text if getattr(resp, "content", None) else ""
                try:
                    u = getattr(resp, "usage", None)
                    if u:
                        usage = {
                            "input_tokens": getattr(u, "input_tokens", 0),
                            "output_tokens": getattr(u, "output_tokens", 0),
                            "total_tokens": (getattr(u, "input_tokens", 0) or 0) + (getattr(u, "output_tokens", 0) or 0),
                        }
                    provider_message_id = getattr(resp, "id", None)
                except Exception:
                    pass
            else:
                # OpenAI (LangChain ChatOpenAI) or CustomModelClient
                ai_msg = await client.ainvoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_message)
                ])
                response_content = ai_msg.content

                # Try to collect usage from several places
                usage = (
                    getattr(ai_msg, "usage_metadata", None)
                    or (getattr(ai_msg, "response_metadata", {}) or {}).get("token_usage")
                    or (getattr(ai_msg, "additional_kwargs", {}) or {}).get("usage")
                    or {}
                )
                provider_message_id = (
                    (getattr(ai_msg, "response_metadata", {}) or {}).get("id")
                    or (getattr(ai_msg, "additional_kwargs", {}) or {}).get("provider_message_id")
                )

            self.logger.log_model_call(
                self.config.selected_model,
                len(system_prompt) + len(user_message),
                len(response_content),
                True
            )

            # Parse as JSON -> validate
            try:
                self.logger.log_step("parsing_response", {
                    "response_length": len(response_content),
                    "response_preview": response_content[:300] + "..." if len(response_content) > 300 else response_content
                })

                parsed = json.loads(response_content)
                validated = response_format.model_validate(parsed)

                self.logger.log_step("validation_successful", {
                    "parsed_fields": list(parsed.keys()) if isinstance(parsed, dict) else "non-dict",
                    "validation_success": True
                })

                result = {
                    "success": True,
                    "data": validated.model_dump(),
                    "raw": response_content,
                    "usage": _norm_usage_dict(usage) if usage else _approx_tokens_by_chars(system_prompt + user_message),
                    "provider_message_id": provider_message_id,
                    "model_name": model_name
                }
                self.logger.finish_operation(True, f"Successfully parsed {response_format.__name__}")
                return result

            except (json.JSONDecodeError, Exception) as e:
                self.logger.log_error(e, "JSON parsing or validation failed")
                result = {
                    "success": False,
                    "error": str(e),
                    "raw": response_content,
                    "usage": _norm_usage_dict(usage) if usage else _approx_tokens_by_chars(system_prompt + user_message),
                    "provider_message_id": provider_message_id,
                    "model_name": model_name
                }
                self.logger.finish_operation(False, f"Parsing failed: {str(e)}")
                return result

        except Exception as e:
            self.logger.log_error(e, "Model API call failed")
            self.logger.log_model_call(self.config.selected_model, len(system_prompt) + len(user_message), success=False)
            result = {
                "success": False,
                "error": str(e),
                "raw": None,
                "usage": _approx_tokens_by_chars(system_prompt + user_message),
                "provider_message_id": provider_message_id,
                "model_name": model_name
            }
            self.logger.finish_operation(False, f"API call failed: {str(e)}")
            return result

    @track_llm(
        provider_extractor=ms_provider_extractor,
        model_extractor=ms_model_extractor,
        usage_extractor=_structured_usage_extractor,         # same normalizer works fine here
        metadata_extractor=ms_freeform_meta_extractor,
    )
    async def call_model_text(
            self,
            client,
            messages: List[BaseMessage],
            *,
            temperature: Optional[float] = 0.3,
            max_tokens: Optional[int] = 1200,
            client_cfg: ClientConfigHint | None = None
    ) -> Dict[str, Any]:
        """Free-form chat with usage + @track_llm. Returns {'text','usage','provider_message_id','model_name'}."""
        usage = {}
        provider_message_id = None
        # model_name = self.config.model_config.get("model_name", self.config.selected_model)
        cfg = client_cfg or self.describe_client(client)
        provider_name = cfg.provider
        model_name = cfg.model_name

        try:
            # Custom endpoint?
            if isinstance(client, CustomModelClient):
                ai_msg = await client.ainvoke(messages, **{
                    "temperature": temperature,
                    "max_new_tokens": max_tokens
                })
                text = ai_msg.content
                usage = (ai_msg.additional_kwargs or {}).get("usage") or {}
                provider_message_id = (ai_msg.additional_kwargs or {}).get("provider_message_id")
                model_name = (ai_msg.additional_kwargs or {}).get("model_name", model_name)
            else:
                # OpenAI (LangChain) or Anthropic (SDK)
                # if self.config.provider == "anthropic" and hasattr(client, "messages"):
                if provider_name == "anthropic" and hasattr(client, "messages"):
                    # convert LC messages
                    sys_prompt = None
                    convo = []
                    for m in messages:
                        if isinstance(m, SystemMessage):
                            sys_prompt = (sys_prompt + "\n" + m.content) if sys_prompt else m.content
                        elif isinstance(m, HumanMessage):
                            convo.append({"role": "user", "content": m.content})
                        elif isinstance(m, AIMessage):
                            convo.append({"role": "assistant", "content": m.content})
                        else:
                            convo.append({"role": "user", "content": str(getattr(m, "content", ""))})
                    resp = client.messages.create(
                        model=self.config.answer_generator_model,
                        system=sys_prompt,
                        messages=convo,
                        max_tokens=max_tokens or 1200,
                        temperature=temperature if temperature is not None else 0.3,
                    )
                    parts = []
                    for c in getattr(resp, "content", []) or []:
                        if getattr(c, "type", "") == "text":
                            parts.append(getattr(c, "text", ""))
                    text = "".join(parts)
                    u = getattr(resp, "usage", None)
                    if u:
                        usage = {
                            "input_tokens": getattr(u, "input_tokens", 0),
                            "output_tokens": getattr(u, "output_tokens", 0),
                            "total_tokens": (getattr(u, "input_tokens", 0) or 0) + (getattr(u, "output_tokens", 0) or 0),
                        }
                    provider_message_id = getattr(resp, "id", None)
                    # model_name = self.config.answer_generator_model

                else:
                    # OpenAI via LangChain
                    ai_msg = await client.ainvoke(messages)
                    text = ai_msg.content
                    usage = (
                            getattr(ai_msg, "usage_metadata", None)
                            or (getattr(ai_msg, "response_metadata", {}) or {}).get("token_usage")
                            or {}
                    )
                    provider_message_id = (getattr(ai_msg, "response_metadata", {}) or {}).get("id")
                    # model_name = (getattr(ai_msg, "response_metadata", {}) or {}).get("model_name") or model_name

            if not usage:
                approx = _approx_tokens_by_chars("".join((getattr(m, "content", "") or "") for m in messages))
                usage = approx

            return {
                "text": text,
                "usage": _norm_usage_dict(usage),
                "provider_message_id": provider_message_id,
                "model_name": model_name
            }

        except Exception as e:
            self.logger.log_error(e, "freeform model call failed")
            return {
                "text": f"Model call failed: {e}",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "provider_message_id": None,
                "model_name": model_name
            }

    async def stream_model_text(
            self,
            client,
            messages: List[BaseMessage],
            *,
            temperature: float = 0.3,
            max_tokens: int = 1200,
            client_cfg: ClientConfigHint | None = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Async generator that yields streaming chunks:
          {"delta": "<text>"} for each piece, and finally
          {"final": True, "usage": {...}, "model_name": "<name>"}.
        Falls back to non-streaming if provider doesn’t support it.
        """
        cfg = client_cfg or self.describe_client(client)
        provider_name = cfg.provider
        model_name = cfg.model_name

        # --- Anthropic streaming (official SDK) ---
        if provider_name == "anthropic" and hasattr(client, "messages"):
            import anthropic
            # Convert LC messages to Anthropic format
            sys_prompt = None
            convo = []
            for m in messages:
                if isinstance(m, SystemMessage):
                    sys_prompt = (sys_prompt + "\n" + m.content) if sys_prompt else m.content
                elif isinstance(m, HumanMessage):
                    convo.append({"role": "user", "content": m.content})
                elif isinstance(m, AIMessage):
                    convo.append({"role": "assistant", "content": m.content})
                else:
                    convo.append({"role": "user", "content": str(getattr(m, "content", ""))})

            # Stream
            with client.messages.stream(
                    model=model_name,
                    system=sys_prompt,
                    messages=convo,
                    max_tokens=max_tokens,
                    temperature=temperature,
            ) as stream:
                for text in stream.text_stream:
                    if text:
                        yield {"delta": text}

                # When stream closes, get final response (usage, id, etc.)
                resp = stream.get_final_response()
                usage = {}
                u = getattr(resp, "usage", None)
                if u:
                    usage = {
                        "input_tokens": getattr(u, "input_tokens", 0),
                        "output_tokens": getattr(u, "output_tokens", 0),
                        "total_tokens": (getattr(u, "input_tokens", 0) or 0) + (getattr(u, "output_tokens", 0) or 0),
                    }
                yield {"final": True, "usage": _norm_usage_dict(usage), "model_name": model_name}
            return

        # --- OpenAI (LangChain ChatOpenAI) streaming ---
        # ChatOpenAI supports .astream returning AIMessageChunk’s content
        from langchain_core.messages import AIMessageChunk
        if isinstance(client, ChatOpenAI):
            combined_text = []
            usage = {}  # will be filled on the final chunk when include_usage=True

            async for chunk in client.astream(messages, stream_usage=True):
                # 1) text deltas
                if isinstance(chunk, AIMessageChunk):
                    piece = chunk.content or ""
                else:
                    piece = getattr(chunk, "content", "") or ""
                if piece:
                    combined_text.append(piece)
                    yield {"delta": piece}

                # 2) try to harvest usage from the chunk (only present on the final one)
                #   a) newer LC sets .usage_metadata on AIMessageChunk
                u = getattr(chunk, "usage_metadata", None)
                #   b) other builds may tuck it into response_metadata
                if not u:
                    rm = getattr(chunk, "response_metadata", None)
                    if isinstance(rm, dict):
                        u = rm.get("usage") or rm.get("token_usage")

                if u:
                    # Normalize a few common shapes/keys
                    input_tokens  = u.get("input_tokens",  u.get("prompt_tokens",  0)) or 0
                    output_tokens = u.get("output_tokens", u.get("completion_tokens", 0)) or 0
                    total_tokens  = u.get("total_tokens")
                    if total_tokens is None:
                        total_tokens = input_tokens + output_tokens
                    usage = {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens,
                    }

            # If your LC build didn’t surface usage at all, fallback to a quick estimate
            if not usage:
                usage = _approx_tokens_by_chars(
                    "".join((getattr(m, "content", "") or "") for m in messages + [AIMessage(content="".join(combined_text))])
                )
            yield {"final": True, "usage": usage, "model_name": model_name}
            return

        # --- Custom endpoint streaming (via our client) ---
        if isinstance(client, CustomModelClient):
            async for ev in client.astream(messages, temperature=temperature, max_new_tokens=max_tokens):
                # expect {"delta": "..."} pieces and at the end {"final": True, "usage": {...}}
                yield ev
            return

        # --- Fallback (no streaming) ---
        res = await self.call_model_text(client, messages, temperature=temperature, max_tokens=max_tokens, client_cfg=cfg)
        text = res.get("text", "") or ""
        # fake-stream the final result in small slices so client still gets a live feel
        for i in range(0, len(text), 30):
            yield {"delta": text[i:i+30]}
        yield {"final": True, "usage": res.get("usage", {}), "model_name": res.get("model_name", model_name)}

    @track_llm(
        provider_extractor=ms_provider_extractor,
        model_extractor=ms_model_extractor,
        usage_extractor=_freeform_usage_extractor,   # expects {'usage': {...}}
        metadata_extractor=ms_freeform_meta_extractor,
    )
    async def stream_model_text_tracked(
            self,
            client,
            messages: List[BaseMessage],
            *,
            on_delta: Callable[[str], Awaitable[None]],
            temperature: float = 0.3,
            max_tokens: int = 1200,
            client_cfg: ClientConfigHint | None = None,
    ) -> Dict[str, Any]:
        """
        Streams tokens (via on_delta) and returns a dict suitable for @track_llm:
          {'text': str, 'usage': {...}, 'provider_message_id': None, 'model_name': str}
        """
        final_chunks: list[str] = []
        usage_out: Dict[str, Any] = {}
        cfg = client_cfg or self.describe_client(client)

        # Use the non-decorated, pure streaming generator
        async for ev in self.stream_model_text(
                client,
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                client_cfg=cfg,
        ):
            if "delta" in ev:
                delta = ev["delta"] or ""
                if delta:
                    final_chunks.append(delta)
                    # forward to caller (e.g., socket emitter)
                    await on_delta(delta)
            if ev.get("final"):
                usage_out = ev.get("usage") or {}
                break
            if ev.get("usage"):
                usage_out = ev.get("usage")

        return {
            "text": "".join(final_chunks),
            "usage": _norm_usage_dict(usage_out),
            "provider_message_id": None,                 # fill if you have one
            "model_name": cfg.model_name,
        }

class CustomModelClient:
    def __init__(self, endpoint: str, api_key: str, model_name: str, temperature: float = 0.7):
        self.endpoint = endpoint
        self.api_key = api_key
        self.model_name = model_name
        self.logger = AgentLogger("CustomModelClient")

        # Default parameters - can be overridden in requests
        self.default_params = {
            "max_new_tokens": 1024,
            "temperature": temperature,
            "top_p": 0.9,
            "min_p": None,
            "skip_cot": True,
            "fabrication_awareness": False,
            "prompt_mode": "default"  # or "simple", "with_reference_material"
        }

    def _convert_langchain_to_conversation(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """
        Convert LangChain messages to the conversation format expected by your handler.
        Your handler expects: [{"role": "system|user|assistant", "content": "..."}]
        """
        conversation = []

        for i, message in enumerate(messages):
            if isinstance(message, SystemMessage):
                conversation.append({"role": "system", "content": message.content})
            elif isinstance(message, HumanMessage):
                conversation.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                conversation.append({"role": "assistant", "content": message.content})
            else:
                # Handle other message types by converting to user message
                self.logger.log_step(
                    f"Unknown message type conversion",
                    {"message_index": i, "type": type(message).__name__, "treated_as": "user"},
                    level="WARNING"
                )
                conversation.append({"role": "user", "content": str(message.content)})

        self.logger.log_step(
            "Message conversion completed",
            {"total_messages": len(messages), "conversation_length": len(conversation)}
        )

        return conversation

    def _prepare_payload(self, messages: List[BaseMessage], **kwargs) -> Dict[str, Any]:
        """Prepare the payload for the HuggingFace endpoint"""
        conversation = self._convert_langchain_to_conversation(messages)

        # Merge default parameters with any provided kwargs
        parameters = {**self.default_params, **kwargs}

        payload = {
            "inputs": conversation,
            "parameters": parameters
        }

        self.logger.log_step(
            "Payload preparation completed",
            {
                "conversation_turns": len(conversation),
                "parameters": parameters,
                "total_input_length": sum(len(msg["content"]) for msg in conversation)
            }
        )

        return payload

    def _prepare_headers(self) -> Dict[str, str]:
        """Prepare headers for the HTTP request"""
        return {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    async def ainvoke(self, messages: List[BaseMessage], **kwargs) -> AIMessage:
        """
        Async method to invoke the model

        Args:
            messages: List of LangChain messages
            **kwargs: Additional parameters to override defaults
                     (max_new_tokens, temperature, top_p, etc.)

        Returns:
            AIMessage with the model's response
        """
        # Start operation logging
        operation_data = self.logger.start_operation(
            "async_model_invocation",
            model_name=self.model_name,
            endpoint=self.endpoint,
            message_count=len(messages),
            parameters=kwargs
        )

        try:
            payload = self._prepare_payload(messages, **kwargs)
            headers = self._prepare_headers()

            # Calculate total input length for logging
            total_input_length = sum(len(str(msg.content)) for msg in messages)

            self.logger.log_step("Initiating HTTP request", {"endpoint": self.endpoint})

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300)  # 5 minute timeout
                ) as response:

                    if response.status != 200:
                        error_text = await response.text()
                        error_msg = f"HTTP {response.status}: {error_text}"
                        self.logger.log_error(
                            Exception(error_msg),
                            context="HTTP request failed"
                        )
                        raise Exception(error_msg)

                    result = await response.json()

                    # Check if the response contains an error
                    if "error" in result:
                        error_msg = f"Model error: {result['error']}"
                        self.logger.log_error(
                            Exception(error_msg),
                            context="Model returned error response"
                        )
                        raise Exception(error_msg)

                    # Extract the response text
                    response_text = result.get("response", "") or result.get("text", "")
                    usage = result.get("usage") or {}

                    try:
                        hdr = {k.lower(): v for k, v in dict(response.headers).items()}
                        pt = int(hdr.get("x-prompt-tokens", 0))
                        ct = int(hdr.get("x-completion-tokens", 0))
                        tt = int(hdr.get("x-total-tokens", pt + ct))
                        if pt or ct or tt:
                            usage = {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": tt}
                    except Exception:
                        pass

                    provider_message_id = result.get("id") or result.get("message_id")

                    if not response_text:
                        self.logger.log_step("Empty response received", level="WARNING")
                        response_text = "No response generated"

                    # Log successful model call
                    self.logger.log_model_call(
                        model_name=self.model_name,
                        prompt_length=total_input_length,
                        response_length=len(response_text),
                        success=True
                    )

                    # Finish operation logging
                    self.logger.finish_operation(
                        success=True,
                        result_summary=f"Generated {len(response_text)} characters"
                    )

                    return AIMessage(content=response_text,
                                     additional_kwargs={
                                         "usage": usage,
                                         "provider_message_id": provider_message_id,
                                         "model_name": self.model_name
                                     })

        except asyncio.TimeoutError as e:
            self.logger.log_error(e, context="Request timeout")
            self.logger.finish_operation(success=False, result_summary="Request timed out")
            raise Exception("Request to model endpoint timed out")
        except aiohttp.ClientError as e:
            self.logger.log_error(e, context="HTTP client error")
            self.logger.finish_operation(success=False, result_summary="HTTP client error")
            raise Exception(f"HTTP client error: {e}")
        except Exception as e:
            self.logger.log_error(e, context="Unexpected error during model invocation")
            self.logger.finish_operation(success=False, result_summary="Unexpected error")
            raise

    def invoke(self, messages: List[BaseMessage], **kwargs) -> AIMessage:
        """
        Synchronous method to invoke the model (wrapper around ainvoke)

        Args:
            messages: List of LangChain messages
            **kwargs: Additional parameters to override defaults

        Returns:
            AIMessage with the model's response
        """
        try:
            # Run the async method in a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.ainvoke(messages, **kwargs))
            finally:
                loop.close()
        except RuntimeError:
            # If we're already in an event loop, use the existing one
            return asyncio.create_task(self.ainvoke(messages, **kwargs))

    async def astream(self, messages: List[BaseMessage], **kwargs) -> AsyncIterator[Dict[str, Any]]:
        """
        Yields {"delta": "..."} pieces and finally {"final": True, "usage": {...}, "model_name": self.model_name}
        Supports SSE and JSONL. Falls back to non-streaming.
        """
        # Start operation logging
        self.logger.start_operation(
            "async_model_stream",
            model_name=self.model_name,
            endpoint=self.endpoint,
            message_count=len(messages),
            parameters=kwargs
        )

        payload = self._prepare_payload(messages, **{**kwargs, "stream": True})
        headers = self._prepare_headers()
        total_input_length = sum(len(str(m.content)) for m in messages)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=600)
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise Exception(f"HTTP {resp.status}: {error_text}")

                ctype = (resp.headers.get("content-type") or "").lower()

                # Try SSE: text/event-stream
                if "text/event-stream" in ctype:
                    usage = {}
                    async for raw in resp.content:
                        line = raw.decode("utf-8", errors="ignore").strip()
                        if not line:
                            continue
                        if line.startswith("data:"):
                            data_s = line[len("data:"):].strip()
                            if data_s == "[DONE]":
                                break
                            try:
                                evt = json.loads(data_s)
                            except Exception:
                                continue
                            # expected forms: {"delta": "..."} or {"response": "..."} or {"usage": {...}, "final": true}
                            if "delta" in evt:
                                yield {"delta": evt["delta"]}
                            elif "response" in evt:
                                yield {"delta": evt["response"]}
                            if evt.get("final"):
                                usage = evt.get("usage") or {}
                                yield {"final": True, "usage": _norm_usage_dict(usage), "model_name": self.model_name}
                                return

                    # if stream closed w/o final, emit final anyway
                    yield {"final": True, "usage": {}, "model_name": self.model_name}
                    return

                # Try JSON lines
                text_buffer = []
                async for raw in resp.content:
                    chunk = raw.decode("utf-8", errors="ignore")
                    for line in chunk.splitlines():
                        s = line.strip()
                        if not s:
                            continue
                        try:
                            obj = json.loads(s)
                        except Exception:
                            # not JSONL → accumulate and fake-stream later
                            text_buffer.append(s)
                            continue
                        if "delta" in obj:
                            yield {"delta": obj["delta"]}
                        elif "response" in obj:
                            yield {"delta": obj["response"]}
                        if obj.get("final"):
                            usage = obj.get("usage") or {}
                            yield {"final": True, "usage": _norm_usage_dict(usage), "model_name": self.model_name}
                            return

                # If we got non-stream JSON (single response) → fallback
                try:
                    full = await resp.json()
                    out = full.get("response", "") or full.get("text", "")
                    if out:
                        for i in range(0, len(out), 30):
                            yield {"delta": out[i:i+30]}
                    usage = full.get("usage") or {}
                    yield {"final": True, "usage": _norm_usage_dict(usage), "model_name": self.model_name}
                except Exception:
                    # last fallback: raw text
                    out = "".join(text_buffer)
                    if out:
                        for i in range(0, len(out), 30):
                            yield {"delta": out[i:i+30]}
                    yield {"final": True, "usage": {}, "model_name": self.model_name}

class CustomEmbeddings(Embeddings):
    """Custom embeddings that work with your embedding service"""

    def __init__(self, endpoint: str, model: str = "sentence-transformers/all-MiniLM-L6-v2", size: int = 384):
        self.endpoint = endpoint
        self.model = model
        self.size = size
        self.logger = AgentLogger("CustomEmbeddings")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        embeddings = []
        for text in texts:
            try:
                response = requests.post(
                    self.endpoint,
                    json={
                        "inputs": text,
                        "model": self.model,
                        "size": self.size
                    },
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()
                embedding = data.get("embedding", [])
                embeddings.append(embedding)

                self.logger.log_step("document_embedded", {
                    "text_length": len(text),
                    "embedding_dim": len(embedding),
                    "model": data.get("model", self.model)
                })

            except Exception as e:
                self.logger.log_error(e, f"Failed to embed document: {text[:100]}...")
                # Return zero vector as fallback
                embeddings.append([0.0] * self.size)

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        try:
            response = requests.post(
                self.endpoint,
                json={
                    "inputs": text,
                    "model": self.model,
                    "size": self.size,
                },
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            embedding = data.get("embedding", [])

            self.logger.log_step("query_embedded", {
                "text_length": len(text),
                "embedding_dim": len(embedding),
                "model": data.get("model", self.model)
            })

            return embedding

        except Exception as e:
            self.logger.log_error(e, f"Failed to embed query: {text[:100]}...")
            # Return zero vector as fallback
            # return [0.0] * self.size
            return None

def setup_logging():
    """Setup global logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('agent_execution.log', mode='a')
        ]
    )


def export_execution_logs(execution_data: Dict[str, Any], filename: str = None):
    """Export execution logs to JSON file"""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"execution_logs_{timestamp}.json"

    try:
        with open(filename, 'w') as f:
            json.dump(execution_data, f, indent=2, default=str)
        return f"Logs exported to {filename}"
    except Exception as e:
        return f"Failed to export logs: {str(e)}"


def probe_embeddings(config_request: ConfigRequest) -> Dict[str, Any]:
    """Test embedding configuration"""
    config = create_workflow_config(config_request)

    embedder_config = config.embedder_config

    if embedder_config["provider"] == "openai":
        # Test OpenAI embeddings
        embeddings = OpenAIEmbeddings(
            model=embedder_config["model_name"],
            openai_api_key=config.openai_api_key
        )

        test_text = "This is a test embedding query"
        embedding = embeddings.embed_query(test_text)

        return {
            "status": "success",
            "embedder_id": config.selected_embedder,
            "provider": "openai",
            "model": embedder_config["model_name"],
            "embedding_size": len(embedding),
            "test_text": test_text,
            "embedding_preview": embedding[:5] if embedding and len(embedding) > 5 else embedding
        }

    elif embedder_config["provider"] == "custom":
        # Test custom embeddings
        if not config_request.custom_embedding_endpoint:
            raise Exception("Custom embedder requires an endpoint")

        embeddings = CustomEmbeddings(
            endpoint=config_request.custom_embedding_endpoint,
            model=embedder_config["model_name"],
            size=embedder_config["dim"]
        )

        test_text = "This is a test embedding query"
        embedding = embeddings.embed_query(test_text)

        return {
            "status": "success" if embedding else "failed",
            "embedder_id": config.selected_embedder,
            "provider": "custom",
            "endpoint": config_request.custom_embedding_endpoint,
            "model": embedder_config["model_name"],
            "embedding_size": len(embedding) if embedding else 0,
            "test_text": test_text,
            "embedding_preview": embedding[:5] if embedding and len(embedding) > 5 else embedding
        }
    else:
        raise Exception(f"Unknown embedding provider: {embedder_config['provider']}")


def create_workflow_config(config_request: ConfigRequest) -> Config:
    """Create workflow configuration based on request"""

    config = Config(selected_model=config_request.selected_model)

    if config_request.openai_api_key:
        config.openai_api_key = config_request.openai_api_key
    if config_request.claude_api_key:
        config.claude_api_key = config_request.claude_api_key

    # Handle declarative embedding configuration
    try:
        config.set_embedder(
            config_request.selected_embedder,
            config_request.custom_embedding_endpoint
        )
    except ValueError as e:
        # Fallback to legacy configuration if declarative fails
        if config_request.custom_embedding_endpoint:
            config.set_custom_embedding_endpoint(
                config_request.custom_embedding_endpoint,
                config_request.custom_embedding_model,
                config_request.custom_embedding_size
            )

    # Handle custom model configuration
    if config_request.custom_model_endpoint:
        config.custom_model_endpoint = config_request.custom_model_endpoint
        config.custom_model_api_key = config_request.custom_model_api_key or ""
        config.custom_model_name = config_request.custom_model_name or "custom-model"
        config.use_custom_endpoint = True

    if config_request.kb_search_endpoint:
        config.set_kb_search_endpoint(config_request.kb_search_endpoint)

    return config

if __name__ == "__main__":

    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())

    async def streaming():

        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        model_name = "gpt-4o-mini"
        client = ChatOpenAI(model=model_name, stream_usage=True)
        msgs = [SystemMessage(content="You are concise."), HumanMessage(content="Say hi!")]

        base_service = ModelServiceBase(Config(selected_model=model_name))
        async for evt in base_service.stream_model_text(client, msgs):
            print(evt)
        print()

    asyncio.run(streaming())


