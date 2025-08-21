# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

MODEL_CONFIGS = {
    "gpt-4.1-nano": {
        "model_name": "gpt-4.1-nano-2025-04-14",
        "provider": "openai",
        "has_classifier": True,
        "description": "GPT-4.1 Nano"
    },
    "gpt-4o": {
        "model_name": "gpt-4o",
        "provider": "openai",
        "has_classifier": True,
        "description": "GPT-4 Optimized - OpenAI model"
    },
    "gpt-4o-mini": {
        "model_name": "gpt-4o-mini",
        "provider": "openai",
        "has_classifier": True,
        "description": "GPT-4 Optimized Mini - High performance, cost-effective"
    },
    "claude-3-5-sonnet": {
        "model_name": "claude-3-5-sonnet-20241022",
        "provider": "anthropic",
        "has_classifier": True,
        "description": "Claude 3.5 Sonnet - Latest Anthropic model"
    },
    "claude-3-haiku": {
        "model_name": "claude-3-haiku-20240307",
        "provider": "anthropic",
        "has_classifier": False,
        "description": "Claude 3 Haiku - Fast and efficient"
    }
}
EMBEDDERS = {
    # OpenAI Embeddings
    "openai-text-embedding-3-small": {
        "provider": "openai",
        "model_name": "text-embedding-3-small",
        "dim": 1536,
        "description": "OpenAI Text Embedding 3 Small - High performance, cost-effective"
    },
    "openai-text-embedding-3-large": {
        "provider": "openai",
        "model_name": "text-embedding-3-large",
        "dim": 3072,
        "description": "OpenAI Text Embedding 3 Large - Highest quality"
    },
    "openai-text-embedding-ada-002": {
        "provider": "openai",
        "model_name": "text-embedding-ada-002",
        "dim": 1536,
        "description": "OpenAI Ada 002 - Previous generation"
    },

    # Custom/Sentence Transformer Embeddings
    "sentence-transformers/all-MiniLM-L6-v2": {
        "provider": "custom",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": 384,
        "description": "All MiniLM L6 v2 - Lightweight and fast"
    },
    "sentence-transformers/distiluse-base-multilingual-cased": {
        "provider": "custom",
        "model_name": "sentence-transformers/distiluse-base-multilingual-cased",
        "dim": 512,
        "description": "DistilUSE Multilingual - Good for multilingual content"
    },
    "sentence-transformers/all-mpnet-base-v2": {
        "provider": "custom",
        "model_name": "sentence-transformers/all-mpnet-base-v2",
        "dim": 768,
        "description": "All MPNet Base v2 - High quality general purpose"
    }
}
