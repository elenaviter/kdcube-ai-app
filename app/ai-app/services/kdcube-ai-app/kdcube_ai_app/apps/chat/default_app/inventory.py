# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/default_inventory.py
from typing import Optional, Dict, Any

from langchain_openai import ChatOpenAI

from kdcube_ai_app.apps.chat.inventory import Config, ConfigRequest, CustomModelClient, ModelServiceBase, AgentLogger
from kdcube_ai_app.infra.accounting.usage import ClientConfigHint
from kdcube_ai_app.tools.serialization import json_safe

BUNDLE_ID = "kdcube.demo.1"

class ThematicBotModelService(ModelServiceBase):
    """Handles interactions with different models"""

    def __init__(self, config: Config):
        super().__init__(config)
        self.config = config
        self.logger = AgentLogger("ThematicBotModelService", config.log_level)

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

    # chat/inventory.py (inside ModelService)
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

APP_STATE_KEYS = [
    "context",
    "user_message",
    "is_our_domain",
    "classification_reasoning",
    "rag_queries",
    "retrieved_docs",
    "reranked_docs",
    "final_answer",
    "error_message",
    "format_fix_attempts",
    "search_hits",
    "execution_id",
    "start_time",
    "step_logs",
    "performance_metrics",
]

def project_app_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only app-specified keys and make them JSON-safe."""
    out = {}
    for k in APP_STATE_KEYS:
        if k == "context":
            # ensure bundle id present
            ctx = dict(state.get("context") or {})
            ctx.setdefault("bundle", BUNDLE_ID)
            out["context"] = json_safe(ctx)
        else:
            out[k] = json_safe(state.get(k))
    return out
